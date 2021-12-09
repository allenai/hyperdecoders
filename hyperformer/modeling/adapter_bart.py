from transformers.models.bart.modeling_bart import (
    BartEncoderLayer,
    BartForConditionalGeneration,
    BartDecoderLayer,
    BartEncoder,
    BartDecoder,
    BartModel,
)
from transformers.models.bart.configuration_bart import BartConfig

import torch
import torch.nn as nn

from modeling.adapter_generators import ParameterGenerator
from modeling.adapter_layer import AdapterLayer

from typing import Optional, Tuple


class BartWithAdapterConfig(BartConfig):
    def __init__(self, adapter_dim=64, generator_dim=128, **kwargs):
        super().__init__(
            **kwargs,
        )
        self.adapter_dim = adapter_dim
        self.generator_hdim = generator_dim


class BartEncoderLayerWithAdapter(BartEncoderLayer):
    def __init__(self, config):
        super().__init__(config)
        self.adapter = AdapterLayer(config.hidden_size, config.adapter_dim)
        self.attn_adapter = AdapterLayer(config.hidden_size, config.adapter_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        layer_head_mask: torch.Tensor,
        output_attentions: bool = False,
    ):

        residual = hidden_states
        hidden_states, attn_weights, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            layer_head_mask=layer_head_mask,
            output_attentions=output_attentions,
        )
        hidden_states = nn.functional.dropout(
            hidden_states, p=self.dropout, training=self.training
        )
        hidden_states = residual + hidden_states + self.attn_adapter(residual)
        hidden_states = self.self_attn_layer_norm(hidden_states)

        residual = hidden_states
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = nn.functional.dropout(
            hidden_states, p=self.activation_dropout, training=self.training
        )
        hidden_states = self.fc2(hidden_states)
        hidden_states = nn.functional.dropout(
            hidden_states, p=self.dropout, training=self.training
        )
        hidden_states = residual + hidden_states + self.adapter(residual)

        hidden_states = self.final_layer_norm(hidden_states)

        if hidden_states.dtype == torch.float16 and (
            torch.isinf(hidden_states).any() or torch.isnan(hidden_states).any()
        ):
            clamp_value = torch.finfo(hidden_states.dtype).max - 1000
            hidden_states = torch.clamp(
                hidden_states, min=-clamp_value, max=clamp_value
            )

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attn_weights,)

        return outputs


class BartDecoderLayerWithAdapter(BartDecoderLayer):
    def __init__(self, config):
        super().__init__(config)
        self.adapter = AdapterLayer(config.hidden_size, config.adapter_dim)
        self.attn_adapter = AdapterLayer(config.hidden_size, config.adapter_dim)
        self.cross_attn_adapter = AdapterLayer(config.hidden_size, config.adapter_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        cross_attn_layer_head_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = True,
    ):
        residual = hidden_states

        # Self Attention
        # decoder uni-directional self-attention cached key/values tuple is at positions 1,2
        self_attn_past_key_value = (
            past_key_value[:2] if past_key_value is not None else None
        )
        # add present self-attn cache to positions 1,2 of present_key_value tuple
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            past_key_value=self_attn_past_key_value,
            attention_mask=attention_mask,
            layer_head_mask=layer_head_mask,
            output_attentions=output_attentions,
        )
        hidden_states = nn.functional.dropout(
            hidden_states, p=self.dropout, training=self.training
        )
        hidden_states = residual + hidden_states + self.attn_adapter(residual)
        hidden_states = self.self_attn_layer_norm(hidden_states)

        # Cross-Attention Block
        cross_attn_present_key_value = None
        cross_attn_weights = None
        if encoder_hidden_states is not None:
            residual = hidden_states

            # cross_attn cached key/values tuple is at positions 3,4 of present_key_value tuple
            cross_attn_past_key_value = (
                past_key_value[-2:] if past_key_value is not None else None
            )
            (
                hidden_states,
                cross_attn_weights,
                cross_attn_present_key_value,
            ) = self.encoder_attn(
                hidden_states=hidden_states,
                key_value_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
                layer_head_mask=cross_attn_layer_head_mask,
                past_key_value=cross_attn_past_key_value,
                output_attentions=output_attentions,
            )
            hidden_states = nn.functional.dropout(
                hidden_states, p=self.dropout, training=self.training
            )
            hidden_states = residual + hidden_states + self.cross_attn_adapter(residual)
            hidden_states = self.encoder_attn_layer_norm(hidden_states)

            # add cross-attn to positions 3,4 of present_key_value tuple
            present_key_value = present_key_value + cross_attn_present_key_value

        # Fully Connected
        residual = hidden_states
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = nn.functional.dropout(
            hidden_states, p=self.activation_dropout, training=self.training
        )
        hidden_states = self.fc2(hidden_states)
        hidden_states = nn.functional.dropout(
            hidden_states, p=self.dropout, training=self.training
        )
        hidden_states = residual + hidden_states + self.adapter(residual)
        hidden_states = self.final_layer_norm(hidden_states)

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights, cross_attn_weights)

        if use_cache:
            outputs += (present_key_value,)

        return outputs


class BartEncoderWithAdapter(BartEncoder):
    def __init__(self, config: BartConfig, embed_tokens: Optional[nn.Embedding] = None):
        super().__init__(config, embed_tokens)
        self.layers = nn.ModuleList(
            [BartEncoderLayerWithAdapter(config) for _ in range(config.encoder_layers)]
        )


class BartDecoderWithAdapter(BartDecoder):
    def __init__(self, config: BartConfig, embed_tokens: Optional[nn.Embedding] = None):
        super().__init__(config, embed_tokens)
        self.layers = nn.ModuleList(
            [BartDecoderLayerWithAdapter(config) for _ in range(config.encoder_layers)]
        )
        self.param_gen = ParameterGenerator(config)
        self.mlp = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.ReLU(),
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.ReLU(),
        )

    def forward(self, input_ids=None, encoder_hidden_states=None, **kwargs):
        x = self.mlp(encoder_hidden_states).mean(dim=1)
        params = self.param_gen(x)
        for layer, param in zip(self.layers, params):
            layer.adapter.apply_adapter_params(encoder_hidden_states.size(0), *param[0])
            layer.attn_adapter.apply_adapter_params(
                encoder_hidden_states.size(0), *param[1]
            )
            layer.cross_attn_adapter.apply_adapter_params(
                encoder_hidden_states.size(0), *param[2]
            )
        return super().forward(
            input_ids=input_ids, encoder_hidden_states=encoder_hidden_states, **kwargs
        )


class BartModelWithAdapter(BartModel):
    def __init__(self, config: BartConfig):
        super().__init__(config)
        self.encoder = BartEncoderWithAdapter(config, self.shared)
        self.decoder = BartDecoderWithAdapter(config, self.shared)


class BartForConditionalGenerationWithAdapter(BartForConditionalGeneration):
    def __init__(self, config: BartConfig):
        super().__init__(config)
        self.model = BartModelWithAdapter(config)
