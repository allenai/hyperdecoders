import math

import torch
import torch.nn as nn


def linear(i, o):
    l = nn.Linear(i, o)
    nn.init.xavier_uniform_(l.weight)
    nn.init.constant_(l.bias, 0.0)
    return l


def hyperfanin_init_weight(linear_layer, hypernet_in, mainnet_in):
    bound = 1e-3 * math.sqrt(3 / (hypernet_in * mainnet_in))
    nn.init.uniform_(linear_layer.weight, -bound, bound)
    nn.init.constant_(linear_layer.bias, 0.0)


def hyperfanin_init_bias(linear_layer, hypernet_in):
    bound = 1e-3 * math.sqrt(3 / (hypernet_in))
    nn.init.uniform_(linear_layer.weight, -bound, bound)
    nn.init.constant_(linear_layer.bias, 0.0)


class SimpleGenerator(nn.Module):
    # takes in a encoded task description and generates parameters of an adapter
    def __init__(self, config, input_dim):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = config.generator_hdim
        self.output_dim = (
            config.hidden_size * config.adapter_dim * 2
            + config.hidden_size
            + config.adapter_dim
        )
        self.linear1 = linear(self.input_dim, self.hidden_dim)
        self.activation_fn = nn.ReLU()
        # output weights
        self.weight_up = nn.Linear(
            self.hidden_dim, config.d_ff * config.adapter_dim
        )
        self.weight_down = nn.Linear(
            self.hidden_dim, config.d_model * config.adapter_dim
        )
        self.bias_up = nn.Linear(self.hidden_dim, config.d_ff)
        self.bias_down = nn.Linear(self.hidden_dim, config.adapter_dim)
        # init weights
        hyperfanin_init_weight(self.weight_up, self.hidden_dim, config.adapter_dim)
        hyperfanin_init_weight(self.weight_down, self.hidden_dim, config.d_model)
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
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.ffn_embed = nn.Embedding(2, 10)
        self.embed = nn.Embedding(config.num_hidden_layers, 10)
        self.decoder = SimpleGenerator(config, config.d_model + 20)

    def forward(self, hidden_inputs):
        layers = []
        # setup idxs we need
        layers_idxs = torch.arange(0, self.config.num_hidden_layers, dtype=torch.long, device=hidden_inputs.device)
        layers_idxs = layers_idxs.repeat(hidden_inputs.size(0), 1)
        ffn_idxs = torch.arange(0, 2, dtype=torch.long, device=hidden_inputs.device)
        ffn_idxs = ffn_idxs.repeat(hidden_inputs.size(0), 1)
        for i in range(self.config.num_hidden_layers):
            layer_embed = self.embed(layers_idxs[:, i])
            ffn_params = []
            for j in range(2):
                ffn_embed = self.ffn_embed(ffn_idxs[:, j])
                hidden_input = torch.cat([hidden_inputs, layer_embed, ffn_embed], dim=1)
                ffn_params.append(self.decoder(hidden_input))
            layers.append(ffn_params)
        return layers
