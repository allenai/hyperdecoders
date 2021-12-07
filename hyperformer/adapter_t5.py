import copy

from transformers.models.t5.configuration_t5 import T5Config
from transformers.models.t5.modeling_t5 import (
    T5Block,
    T5LayerFF,
    T5Stack,
    T5Model,
    T5ForConditionalGeneration,
    T5EncoderModel,
    T5DenseGatedGeluDense,
    T5DenseReluDense,
)
from transformers.activations import ACT2FN
import torch
from torch import nn

from adapter_generators import ParameterGenerator
from adapter_layer import AdapterLayer


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

class T5DenseReluDenseWithAdapter(T5DenseReluDense):
    def __init__(self, config):
        super().__init__(config)
        self.wi_adapter = AdapterLayer(config.d_model, config.adapter_dim, config.d_ff)
        self.wo_adapter = AdapterLayer(config.d_ff, config.adapter_dim, config.d_model)

    def forward(self, hidden_states):
        hidden_states = self.wi(hidden_states) + self.wi_adapter(hidden_states)
        hidden_states = nn.functional.relu(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.wo(hidden_states) + self.wo_adapter(hidden_states)
        return hidden_states


class T5DenseGatedGeluDenseWithAdapter(T5DenseGatedGeluDense):
    def __init__(self, config):
        super().__init__(config)
        # imma be big and chuck adapters for *every* linear layer
        self.wi_0_adapter = AdapterLayer(config.d_model, config.adapter_dim, config.d_ff)
        self.wi_1_adapter = AdapterLayer(config.d_model, config.adapter_dim, config.d_ff)
        self.wo_adapter = AdapterLayer(config.d_ff, config.adapter_dim, config.d_model)

    def forward(self, hidden_states):
        hidden_gelu = self.gelu_act(self.wi_0(hidden_states) + self.wi_0_adapter(hidden_states))
        hidden_linear = self.wi_1(hidden_states) + self.wi_1_adapter(hidden_states)
        hidden_states = hidden_gelu * hidden_linear
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.wo(hidden_states) + self.wo_adapter(hidden_states)
        return hidden_states


class T5LayerFFWithAdapter(T5LayerFF):
    def __init__(self, config):
        super().__init__(config)
        if config.feed_forward_proj == "relu":
            self.DenseReluDense = T5DenseReluDenseWithAdapter(config)
        elif config.feed_forward_proj == "gated-gelu":
            self.DenseReluDense = T5DenseGatedGeluDenseWithAdapter(config)
        else:
            raise ValueError(
                f"{self.config.feed_forward_proj} is not supported. Choose between `relu` and `gated-gelu`"
            )


class T5BlockWithAdapter(T5Block):
    def __init__(self, config, has_relative_attention_bias=False):
        super().__init__(
            config, has_relative_attention_bias=has_relative_attention_bias
        )
        self.layer[-1] = T5LayerFFWithAdapter(config)


class T5StackWithAdapter(T5Stack):
    def __init__(self, config, embed_tokens=None):
        super().__init__(config, embed_tokens=embed_tokens)
        self.block = torch.nn.ModuleList(
            [
                T5BlockWithAdapter(config, has_relative_attention_bias=bool(i == 0))
                for i in range(config.num_layers)
            ]
        )
        self.param_gen = ParameterGenerator(config)
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
        for layer in self.block:
            adapter_block = layer.layer[-1].DenseReluDense
            if isinstance(adapter_block, T5DenseGatedGeluDenseWithAdapter):
                adapter_block.wi_0_adapter.clear_adapter()
                adapter_block.wi_1_adapter.clear_adapter()
                adapter_block.wo_adapter.clear_adapter()
            elif isinstance(adapter_block, T5DenseReluDenseWithAdapter):
                adapter_block.wi_adapter.clear_adapter()
                adapter_block.wo_adapter.clear_adapter()

    def apply_params_to_adapters(self, batch_size, generated_params):
        for p, layer in zip(generated_params, self.block):
            adapter_layer = layer.layer[-1].DenseReluDense
            if isinstance(adapter_layer, T5DenseGatedGeluDenseWithAdapter):
                adapter_layer.wi_0_adapter.apply_adapter_params(batch_size, *p[0])
                adapter_layer.wi_1_adapter.apply_adapter_params(batch_size, *p[1])
                #adapter_layer.wo_adapter.apply_adapter_params(batch_size, *p[2])
            elif isinstance(adapter_layer, T5DenseReluDenseWithAdapter):
                adapter_layer.wi_adapter.apply_adapter_params(batch_size, *p[0])
                #adapter_layer.wo_adapter.apply_adapter_params(batch_size, *p[1])

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
