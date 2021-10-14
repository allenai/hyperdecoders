import copy
import warnings

from transformers.models.t5.configuration_t5 import T5Config
from transformers.models.t5.modeling_t5 import T5Block, T5LayerFF, T5Stack, T5Model, T5ForConditionalGeneration, T5EncoderModel, __HEAD_MASK_WARNING_MSG
from transformers.activations import ACT2FN
from transformers.modeling_outputs import BaseModelOutput, Seq2SeqModelOutput, Seq2SeqLMOutput

import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from adapter_generators import ParameterGenerator



class T5WithAdapterConfig(T5Config):
    def __init__(self, train_adapters=False, **kwargs):
        super().__init__(**kwargs)
        self.adapter_dim = 64
        self.generator_hdim = self.d_model * 0.25


class T5LayerFFWithAdapter(T5LayerFF):
    def __init__(self, config):
        super().__init__(config)
        self.adapter_dim = config.adapter_dim
        self.adapter_down_weight = torch.zeros(config.hidden_size, self.adapter_dim)
        self.adapter_down_bias = torch.zeros(1, 1, self.adapter_dim)

        self.adapter_up_weight = torch.zeros(self.adapter_dim, config.hidden_size)
        self.adapter_up_bias = torch.zeros(1, 1, config.hidden_size)
        self.hidden_act = ACT2FN[config.feed_forward_proj]

    def adapter_down(self, x):
        return (x @ self.adapter_down_weight) + self.adapter_down_bias.unsqueeze(1)

    def adapter_up(self, x):
        return (x @ self.adapter_up_weight) + self.adapter_up_bias.unsqueeze(1)

    def apply_adapter(self, x):
        # downsample, activate, upsample, skip connection
        y = self.adapter_down(x)
        y = self.hidden_act(y)
        y = self.adapter_up(y)
        return y + x

    def forward(self, hidden_states):
        forwarded_states = self.layer_norm(hidden_states)
        forwarded_states = self.dropout(self.DenseReluDense(forwarded_states))
        forwarded_states = self.apply_adapter(forwarded_states)
        hidden_states = hidden_states + forwarded_states
        return hidden_states

class T5BlockWithAdapter(T5Block):
    def __init__(self, config, has_relative_attention_bias=False):
        super().__init__(config, has_relative_attention_bias=has_relative_attention_bias)
        self.layer[-1] = T5LayerFFWithAdapter(config)

class T5StackWithAdapter(T5Stack):
    def __init__(self, config, embed_tokens=None):
        super().__init__(config, embed_tokens=embed_tokens)
        self.block = torch.nn.ModuleList(
            [T5BlockWithAdapter(config, has_relative_attention_bias=bool(i == 0)) for i in range(config.num_layers)]
        )

    def apply_params_to_adapters(self, batch_size, generated_params):
        hidden_size = self.config.hidden_size
        d_adapter = self.config.adapter_dim

        for p, layer in zip(generated_params, self.block):
            adapter_layer = layer.layer[-1]
            # dw, db: down weight, down bias
            # uw, ub: up weight, up bias
            dw, uw, db, ub = p[:, 0:hidden_size*d_adapter], \
                            p[:, hidden_size*d_adapter:hidden_size*d_adapter*2], \
                            p[:, hidden_size*d_adapter*2:hidden_size*d_adapter*2+d_adapter], \
                            p[:, hidden_size*d_adapter*2+d_adapter:hidden_size*d_adapter*2+d_adapter+hidden_size]
            adapter_layer.adapter_down_weight = dw.view(batch_size, hidden_size, d_adapter)
            adapter_layer.adapter_down_bias = db.view(batch_size, d_adapter)
            adapter_layer.adapter_up_weight = uw.view(batch_size, d_adapter, hidden_size)
            adapter_layer.adapter_up_bias = ub.view(batch_size, hidden_size)

class T5ModelWithAdapter(T5Model):
    def __init__(self, config: T5Config):
        super().__init__(config)

        self.param_gen = ParameterGenerator(config, config.feed_forward_proj, config.d_model)
        self.param_gen_decoder = ParameterGenerator(config, config.feed_forward_proj, config.d_model)
        self.init_state = nn.Parameter(torch.randn(config.hidden_size))
        self.mlp = nn.Sequential(nn.Linear(config.d_model, config.d_model), nn.ReLU(), nn.Linear(config.d_model, config.d_model), nn.ReLU())

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

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        encoder_outputs=None,
        past_key_values=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # FutureWarning: head_mask was separated into two input args - head_mask, decoder_head_mask
        if head_mask is not None and decoder_head_mask is None:
            if self.config.num_layers == self.config.num_decoder_layers:
                warnings.warn(__HEAD_MASK_WARNING_MSG, FutureWarning)
                decoder_head_mask = head_mask

        # Encode if needed (training, first prediction pass)
        if encoder_outputs is None:
            # generate params.
            generated_params = self.param_gen(self.init_state.view(1, -1).expand(input_ids.size(0), -1))
            self.encoder.apply_params_to_adapters(input_ids.size(0), generated_params)
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        hidden_states = encoder_outputs[0]
        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)
        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)
            hidden_states = hidden_states.to(self.decoder.first_device)
            if decoder_input_ids is not None:
                decoder_input_ids = decoder_input_ids.to(self.decoder.first_device)
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.decoder.first_device)
            if decoder_attention_mask is not None:
                decoder_attention_mask = decoder_attention_mask.to(self.decoder.first_device)
        # pool and generate decoder adapters
        x = self.mlp(hidden_states).mean(dim=1) # mean over sentence
        self.decoder.apply_params_to_adapters(input_ids.size(0), self.param_gen_decoder(x))

        # Decode
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            past_key_values=past_key_values,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        if not return_dict:
            return decoder_outputs + encoder_outputs

        return Seq2SeqModelOutput(
            last_hidden_state=decoder_outputs.last_hidden_state,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )


class T5ForConditionalGenerationWithAdapter(T5ForConditionalGeneration):
    def __init__(self, config):
        super().__init__(config)

        self.param_gen = ParameterGenerator(config, config.feed_forward_proj, config.d_model)
        self.param_gen_decoder = ParameterGenerator(config, config.feed_forward_proj, config.d_model)
        self.init_state = nn.Parameter(torch.randn(config.hidden_size))
        self.mlp = nn.Sequential(nn.Linear(config.d_model, config.d_model), nn.ReLU(), nn.Linear(config.d_model, config.d_model), nn.ReLU())

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

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        encoder_outputs=None,
        past_key_values=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        task=None,
        task_embedding=None,
        t5_block_adapters=None
    ):
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # FutureWarning: head_mask was separated into two input args - head_mask, decoder_head_mask
        if head_mask is not None and decoder_head_mask is None:
            if self.config.num_layers == self.config.num_decoder_layers:
                warnings.warn(__HEAD_MASK_WARNING_MSG, FutureWarning)
                decoder_head_mask = head_mask

        # Encode if needed (training, first prediction pass)
        if encoder_outputs is None:
             # generate params.
            generated_params = self.param_gen(self.init_state.view(1, -1).expand(input_ids.size(0), -1))
            self.encoder.apply_params_to_adapters(input_ids.size(0), generated_params)
            # Convert encoder inputs in embeddings if needed
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        hidden_states = encoder_outputs[0]

        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)

        if labels is not None and decoder_input_ids is None and decoder_inputs_embeds is None:
            # get decoder inputs from shifting lm labels to the right
            decoder_input_ids = self._shift_right(labels)

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)
            hidden_states = hidden_states.to(self.decoder.first_device)
            if decoder_input_ids is not None:
                decoder_input_ids = decoder_input_ids.to(self.decoder.first_device)
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.decoder.first_device)
            if decoder_attention_mask is not None:
                decoder_attention_mask = decoder_attention_mask.to(self.decoder.first_device)

        # pool and generate decoder adapters
        x = self.mlp(hidden_states).mean(dim=1) # mean over sentence
        self.decoder.apply_params_to_adapters(input_ids.size(0), self.param_gen_decoder(x))
        # Decode
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            past_key_values=past_key_values,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = decoder_outputs[0]

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.encoder.first_device)
            self.lm_head = self.lm_head.to(self.encoder.first_device)
            sequence_output = sequence_output.to(self.lm_head.weight.device)

        if self.config.tie_word_embeddings:
            # Rescale output before projecting on vocab
            # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/transformer.py#L586
            sequence_output = sequence_output * (self.model_dim ** -0.5)

        lm_logits = self.lm_head(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))
            # TODO(thom): Add z_loss https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/layers.py#L666

        if not return_dict:
            output = (lm_logits,) + decoder_outputs[1:] + encoder_outputs
            return ((loss,) + output) if loss is not None else output

        return Seq2SeqLMOutput(
            loss=loss,
            logits=lm_logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )

class T5EncoderModelWithAdapter(T5EncoderModel):
    def __init__(self, config: T5Config):
        super().__init__(config)

        encoder_config = copy.deepcopy(config)
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False
        self.encoder = T5StackWithAdapter(encoder_config, self.shared)

        self.init_weights()

