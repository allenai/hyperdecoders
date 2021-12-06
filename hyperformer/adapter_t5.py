import copy

from transformers.models.t5.configuration_t5 import T5Config
from transformers.models.t5.modeling_t5 import T5Block, T5LayerFF, T5Stack, T5Model, T5ForConditionalGeneration, T5EncoderModel
from transformers.activations import ACT2FN
import torch
from torch import nn

from adapter_generators import ParameterGenerator



class T5WithAdapterConfig(T5Config):
    def __init__(self, use_adapters=True, use_manual_adapters=False, adapter_hidden_param=64, hypernetwork_bottleneck=128, **kwargs):
        super().__init__(**kwargs)
        self.adapter_dim = adapter_hidden_param
        self.generator_hdim = hypernetwork_bottleneck
        self.use_adapters = use_adapters
        self.use_manual_adapters = use_manual_adapters

class T5LayerFFWithAdapter(T5LayerFF):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.adapter_dim = config.adapter_dim
        self.adapter_down_weight = None
        self.adapter_down_bias = None
        self.adapter_up_weight = None
        self.adapter_up_bias = None
        # manual adapters
        self.adapter_down_manual = nn.Linear(config.hidden_size, self.adapter_dim)
        self.adapter_up_manual = nn.Linear(self.adapter_dim, config.hidden_size)
        self.hidden_act = nn.ReLU()

    def adapter_down(self, x):
        return (x @ self.adapter_down_weight) + self.adapter_down_bias.unsqueeze(1)

    def adapter_up(self, x):
        return (x @ self.adapter_up_weight) + self.adapter_up_bias.unsqueeze(1)

    def apply_adapter(self, x):
        if self.config.use_manual_adapters or self.adapter_down_weight is None:
            y = self.adapter_down_manual(x)
            y = self.hidden_act(y)
            y = self.adapter_up_manual(y)
        else:
            y = self.adapter_down(x)
            y = self.hidden_act(y)
            y = self.adapter_up(y)
        return y + x

    def forward(self, hidden_states):
        forwarded_states = self.layer_norm(hidden_states)
        forwarded_states = self.dropout(self.DenseReluDense(forwarded_states))
        if self.config.use_adapters:
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
        self.param_gen = ParameterGenerator(config)
        self.mlp = nn.Sequential(nn.Linear(config.d_model, config.d_model), nn.ReLU(), nn.Linear(config.d_model, config.d_model), nn.ReLU())

    def forward(
        self,
        input_ids=None,
        encoder_hidden_states=None,
        **kwargs,
    ):
        # using input ids to determine whats going
        if self.is_decoder and self.config.use_adapters:
            # from transformers import T5TokenizerFast
            # tok = T5TokenizerFast.from_pretrained('t5-small')
            # print(input_ids[0])
            # print(tok.decode(input_ids[0].cpu().numpy().tolist()))
            x = self.mlp(encoder_hidden_states).mean(dim=1) # mean over sentence
            self.apply_params_to_adapters(encoder_hidden_states.size(0), self.param_gen(x))
        return super().forward(input_ids=input_ids, encoder_hidden_states=encoder_hidden_states, **kwargs)


    def apply_params_to_adapters(self, batch_size, generated_params):
        hidden_size = self.config.hidden_size
        d_adapter = self.config.adapter_dim

        for p, layer in zip(generated_params, self.block):
            adapter_layer = layer.layer[-1]
            # dw, db: down weight, down bias
            # uw, ub: up weight, up bias
            uw, dw, ub, db = p
            adapter_layer.adapter_down_weight = dw.view(batch_size, hidden_size, d_adapter)
            adapter_layer.adapter_down_bias = db.view(batch_size, d_adapter)
            adapter_layer.adapter_up_weight = uw.view(batch_size, d_adapter, hidden_size)
            adapter_layer.adapter_up_bias = ub.view(batch_size, hidden_size)

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

