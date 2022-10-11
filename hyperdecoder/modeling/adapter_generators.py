import math

import torch
import torch.nn as nn


def hyperfanin_init_weight(linear_layer, hypernet_in, mainnet_in):
    bound = 1e-3 * math.sqrt(3 / (hypernet_in * mainnet_in))
    nn.init.uniform_(linear_layer.weight, -bound, bound)
    nn.init.constant_(linear_layer.bias, 0.0)


def hyperfanin_init_bias(linear_layer, hypernet_in):
    bound = 1e-3 * math.sqrt(3 / (hypernet_in))
    nn.init.uniform_(linear_layer.weight, -bound, bound)
    nn.init.constant_(linear_layer.bias, 0.0)


class SimpleGenerator(nn.Module):
    def __init__(self, config, input_dim, hidden_size, is_encoder=False):
        super().__init__()
        adapter_dim = (
            config.encoder_adapter_dim if is_encoder else config.decoder_adapter_dim
        )
        self.input_dim = input_dim
        self.hidden_dim = config.hypernetwork_bottleneck
        self.linear1 = nn.Linear(self.input_dim, self.hidden_dim)
        self.activation_fn = nn.ReLU()
        # output weights
        self.weight_up = nn.Linear(self.hidden_dim, hidden_size * adapter_dim)
        self.weight_down = nn.Linear(self.hidden_dim, hidden_size * adapter_dim)
        self.bias_up = nn.Linear(self.hidden_dim, hidden_size)
        self.bias_down = nn.Linear(self.hidden_dim, adapter_dim)
        # init weights
        hyperfanin_init_weight(self.weight_up, self.hidden_dim, adapter_dim)
        hyperfanin_init_weight(self.weight_down, self.hidden_dim, hidden_size)
        hyperfanin_init_bias(self.bias_up, self.hidden_dim)
        hyperfanin_init_bias(self.bias_down, self.hidden_dim)

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation_fn(x)
        return (
            self.weight_up(x),
            self.weight_down(x),
            self.bias_up(x),
            self.bias_down(x),
        )


class ParameterGenerator(nn.Module):
    def __init__(self, config, hidden_size, is_encoder=False):
        super().__init__()
        self.config = config
        self.layer_embed = nn.Embedding(config.num_hidden_layers, 10)
        self.decoder = SimpleGenerator(
            config, config.hidden_size + 10, hidden_size, is_encoder=is_encoder
        )

    def forward(self, hidden_inputs):
        layers = []
        # setup idxs we need
        layers_idxs = torch.arange(
            0,
            self.config.num_hidden_layers,
            dtype=torch.long,
            device=hidden_inputs.device,
        )
        layers_idxs = layers_idxs.repeat(hidden_inputs.size(0), 1)
        for i in range(self.config.num_hidden_layers):
            layer_embed = self.layer_embed(layers_idxs[:, i])
            hidden_input = torch.cat([hidden_inputs, layer_embed], dim=1)
            layers.append(self.decoder(hidden_input))
        return layers
