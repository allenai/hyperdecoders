import copy

from transformers.models.t5.configuration_t5 import T5Config
from transformers.models.t5.modeling_t5 import (
    T5Block,
    T5LayerFF,
    T5Stack,
    T5Model,
    T5ForConditionalGeneration,
    T5EncoderModel,
    T5LayerSelfAttention,
    T5LayerCrossAttention,
    T5Attention,
)
import torch
from torch import nn

from modeling.adapter_generators import ParameterGenerator
from modeling.adapter_layer import AdapterLayer


class T5WithAdapterConfig(T5Config):
    def __init__(
        self,
        use_adapters=True,
        use_manual_adapters=False,
        adapter_hidden_param=64,
        hypernetwork_bottleneck=128,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.adapter_dim = adapter_hidden_param
        self.generator_hdim = hypernetwork_bottleneck
        self.use_adapters = use_adapters
        self.use_manual_adapters = use_manual_adapters


class T5AttentionWithAdapter(T5Attention):
    def __init__(self, config, has_relative_attention_bias=False):
        super().__init__(config, has_relative_attention_bias=has_relative_attention_bias)
        self.adapter_layer = AdapterLayer(
            config.d_kv, config.adapter_dim,
        )

    def forward(
        self,
        hidden_states,
        mask=None,
        key_value_states=None,
        position_bias=None,
        past_key_value=None,
        layer_head_mask=None,
        query_length=None,
        use_cache=False,
        output_attentions=False,
    ):
        """
        Self-attention (if key_value_states is None) or attention over source sentence (provided by key_value_states).
        """
        # Input is (batch_size, seq_length, dim)
        # Mask is (batch_size, key_length) (non-causal) or (batch_size, key_length, key_length)
        # past_key_value[0] is (batch_size, n_heads, q_len - 1, dim_per_head)
        batch_size, seq_length = hidden_states.shape[:2]

        real_seq_length = seq_length

        if past_key_value is not None:
            assert (
                len(past_key_value) == 2
            ), f"past_key_value should have 2 past states: keys and values. Got { len(past_key_value)} past states"
            real_seq_length += past_key_value[0].shape[2] if query_length is None else query_length

        key_length = real_seq_length if key_value_states is None else key_value_states.shape[1]

        def shape(states):
            """projection"""
            return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)

        def unshape(states):
            """reshape"""
            return states.transpose(1, 2).contiguous().view(batch_size, -1, self.inner_dim)

        def project(hidden_states, proj_layer, key_value_states, past_key_value):
            """projects hidden states correctly to key/query states"""
            if key_value_states is None:
                # self-attn
                # (batch_size, n_heads, seq_length, dim_per_head)
                hidden_states = shape(proj_layer(hidden_states))
            elif past_key_value is None:
                # cross-attn
                # (batch_size, n_heads, seq_length, dim_per_head)
                hidden_states = shape(proj_layer(key_value_states))

            if past_key_value is not None:
                if key_value_states is None:
                    # self-attn
                    # (batch_size, n_heads, key_length, dim_per_head)
                    hidden_states = torch.cat([past_key_value, hidden_states], dim=2)
                else:
                    # cross-attn
                    hidden_states = past_key_value
            return hidden_states

        # get query states
        query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)

        

        # get key/value states
        key_states = project(
            hidden_states, self.k, key_value_states, past_key_value[0] if past_key_value is not None else None
        )
        value_states = project(
            hidden_states, self.v, key_value_states, past_key_value[1] if past_key_value is not None else None
        )

        # compute scores
        scores = torch.matmul(
            query_states, key_states.transpose(3, 2)
        )  # equivalent of torch.einsum("bnqd,bnkd->bnqk", query_states, key_states), compatible with onnx op>9

        if position_bias is None:
            if not self.has_relative_attention_bias:
                position_bias = torch.zeros(
                    (1, self.n_heads, real_seq_length, key_length), device=scores.device, dtype=scores.dtype
                )
                if self.gradient_checkpointing and self.training:
                    position_bias.requires_grad = True
            else:
                position_bias = self.compute_bias(real_seq_length, key_length)

            # if key and values are already calculated
            # we want only the last query position bias
            if past_key_value is not None:
                position_bias = position_bias[:, :, -hidden_states.size(1) :, :]

            if mask is not None:
                position_bias = position_bias + mask  # (batch_size, n_heads, seq_length, key_length)

        scores += position_bias
        attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(
            scores
        )  # (batch_size, n_heads, seq_length, key_length)
        attn_weights = nn.functional.dropout(
            attn_weights, p=self.dropout, training=self.training
        )  # (batch_size, n_heads, seq_length, key_length)

        # Mask heads if we want to
        if layer_head_mask is not None:
            attn_weights = attn_weights * layer_head_mask

        # we apply adapters to the attention output *per-head*
        

        attn_results = torch.matmul(attn_weights, value_states)
        attn_output = unshape(attn_results + self.adapter_layer(value_states))  # (batch_size, seq_length, dim)
        attn_output = self.o(attn_output)

        present_key_value_state = (key_states, value_states) if (self.is_decoder and use_cache) else None
        outputs = (attn_output,) + (present_key_value_state,) + (position_bias,)

        if output_attentions:
            outputs = outputs + (attn_weights,)
        return outputs

class T5LayerFFWithAdapter(T5LayerFF):
    def __init__(self, config):
        super().__init__(config)
        self.adapter_layer = AdapterLayer(config.hidden_size, config.adapter_dim)

    def forward(self, hidden_states):
        normed_states = self.layer_norm(hidden_states)
        forwarded_states = self.DenseReluDense(normed_states)
        hidden_states = (
            hidden_states
            + self.dropout(forwarded_states)
            + self.adapter_layer(normed_states)
        )
        return hidden_states


class T5LayerSelfAttentionWithAdapter(T5LayerSelfAttention):
    def __init__(self, config, has_relative_attention_bias=False):
        super().__init__(
            config, has_relative_attention_bias=has_relative_attention_bias
        )
        self.SelfAttention = T5AttentionWithAdapter(config, has_relative_attention_bias=has_relative_attention_bias)
        #self.adapter_layer = AdapterLayer(config.hidden_size, config.adapter_dim)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_bias=None,
        layer_head_mask=None,
        past_key_value=None,
        use_cache=False,
        output_attentions=False,
    ):
        normed_hidden_states = self.layer_norm(hidden_states)
        attention_output = self.SelfAttention(
            normed_hidden_states,
            mask=attention_mask,
            position_bias=position_bias,
            layer_head_mask=layer_head_mask,
            past_key_value=past_key_value,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )
        hidden_states = (
            hidden_states
            + self.dropout(attention_output[0])
            #+ self.adapter_layer(normed_hidden_states)
        )
        outputs = (hidden_states,) + attention_output[
            1:
        ]  # add attentions if we output them
        return outputs


class T5LayerCrossAttentionWithAdapter(T5LayerCrossAttention):
    def __init__(self, config):
        super().__init__(config)
        self.adapter_layer = AdapterLayer(config.hidden_size, config.adapter_dim)

    def forward(
        self,
        hidden_states,
        key_value_states,
        attention_mask=None,
        position_bias=None,
        layer_head_mask=None,
        past_key_value=None,
        use_cache=False,
        query_length=None,
        output_attentions=False,
    ):
        normed_hidden_states = self.layer_norm(hidden_states)
        attention_output = self.EncDecAttention(
            normed_hidden_states,
            mask=attention_mask,
            key_value_states=key_value_states,
            position_bias=position_bias,
            layer_head_mask=layer_head_mask,
            past_key_value=past_key_value,
            use_cache=use_cache,
            query_length=query_length,
            output_attentions=output_attentions,
        )
        layer_output = (
            hidden_states
            + self.dropout(attention_output[0])
            + self.adapter_layer(normed_hidden_states)
        )
        outputs = (layer_output,) + attention_output[
            1:
        ]  # add attentions if we output them
        return outputs


class T5BlockWithAdapter(T5Block):
    def __init__(self, config, has_relative_attention_bias=False):
        super().__init__(
            config, has_relative_attention_bias=has_relative_attention_bias
        )
        self.layer[0] = T5LayerSelfAttentionWithAdapter(
            config, has_relative_attention_bias=has_relative_attention_bias
        )
        #if self.is_decoder:
        #    self.layer[1] = T5LayerCrossAttentionWithAdapter(config)
        #self.layer[-1] = T5LayerFFWithAdapter(config)


class T5StackWithAdapter(T5Stack):
    def __init__(self, config, embed_tokens=None):
        super().__init__(config, embed_tokens=embed_tokens)
        self.block = torch.nn.ModuleList(
            [
                T5BlockWithAdapter(config, has_relative_attention_bias=bool(i == 0))
                for i in range(config.num_layers)
            ]
        )
        self.param_gen = ParameterGenerator(config, config.d_kv)
        self.mlp = nn.Sequential(
            nn.Linear(config.d_model, config.d_model),
            nn.ReLU(),
            nn.Linear(config.d_model, config.d_model),
            nn.ReLU(),
        )

    def forward(
        self,
        input_ids=None,
        encoder_hidden_states=None,
        **kwargs,
    ):
        # using input ids to determine whats going
        self.clear_adapters()
        if self.is_decoder and self.config.use_adapters:
            x = self.mlp(encoder_hidden_states).mean(dim=1)  # mean over sentence
            self.apply_params_to_adapters(
                encoder_hidden_states.size(0), self.param_gen(x)
            )
        return super().forward(
            input_ids=input_ids, encoder_hidden_states=encoder_hidden_states, **kwargs
        )

    def clear_adapters(self):
        for block in self.block:
            for layer in block.layer:
                if isinstance(layer, T5LayerSelfAttentionWithAdapter):
                    layer.SelfAttention.adapter_layer.clear_adapter()
                elif (
                    isinstance(layer, T5LayerSelfAttentionWithAdapter)
                    or isinstance(layer, T5LayerCrossAttentionWithAdapter)
                    or isinstance(layer, T5LayerFFWithAdapter)
                ):
                    layer.adapter_layer.clear_adapter()

    def apply_params_to_adapters(self, batch_size, generated_params):
        for param, block in zip(generated_params, self.block):
            for p, l in zip(param, block.layer):
                if isinstance(l, T5LayerSelfAttentionWithAdapter):
                    l.SelfAttention.adapter_layer.apply_adapter_params(batch_size, *p)
                elif (
                    isinstance(l, T5LayerSelfAttentionWithAdapter)
                    or isinstance(l, T5LayerCrossAttentionWithAdapter)
                    or isinstance(l, T5LayerFFWithAdapter)
                ):
                    l.adapter_layer.apply_adapter_params(batch_size, *p)


class T5ModelWithAdapter(T5Model):
    def __init__(self, config: T5Config):
        super().__init__(config)

        encoder_config = copy.deepcopy(config)
        encoder_config.is_decoder = False
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False
        self.encoder = T5StackWithAdapter(encoder_config, self.shared)

        decoder_config = copy.deepcopy(config)
        decoder_config.is_decoder = True
        decoder_config.is_encoder_decoder = False
        decoder_config.num_layers = config.num_decoder_layers
        self.decoder = T5StackWithAdapter(decoder_config, self.shared)

        self.init_weights()


class T5ForConditionalGenerationWithAdapter(T5ForConditionalGeneration):
    def __init__(self, config):
        super().__init__(config)
        encoder_config = copy.deepcopy(config)
        encoder_config.is_decoder = False
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False
        self.encoder = T5StackWithAdapter(encoder_config, self.shared)

        decoder_config = copy.deepcopy(config)
        decoder_config.is_decoder = True
        decoder_config.is_encoder_decoder = False
        decoder_config.num_layers = config.num_decoder_layers
        self.decoder = T5StackWithAdapter(decoder_config, self.shared)

        self.init_weights()


class T5EncoderModelWithAdapter(T5EncoderModel):
    def __init__(self, config: T5Config):
        super().__init__(config)

        encoder_config = copy.deepcopy(config)
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False
        self.encoder = T5StackWithAdapter(encoder_config, self.shared)

        self.init_weights()
