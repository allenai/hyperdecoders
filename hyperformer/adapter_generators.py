import torch
import torch.nn as nn
from transformers.activations import ACT2FN
import math

# todo: hidden should use xaiver init
def linear(i, o):
    l = nn.Linear(i, o)
    nn.init.xavier_uniform_(l.weight)
    nn.init.constant_(l.bias, 0.0)
    return l

# assumptions: using relu and xavier init, and data input has variance 1 (😬)
# times by a constant to 'init' the generated adapters near 0. This is required
# for stable training, from og adapter paper.
def hyperfanin_init_weight(linear_layer, hypernet_in, mainnet_in):
  bound = 1e-3 * math.sqrt(3 / (hypernet_in * mainnet_in))
  nn.init.uniform_(linear_layer.weight, -bound, bound)
  nn.init.constant_(linear_layer.bias, 0.0)

def hyperfanin_init_bias(linear_layer,  hypernet_in):
  bound = 1e-3 * math.sqrt(3 / (hypernet_in))
  nn.init.uniform_(linear_layer.weight, -bound, bound)
  nn.init.constant_(linear_layer.bias, 0.0)

class SimpleGenerator(nn.Module):
    # takes in a encoded task description and generates parameters of an adapter
    def __init__(self, config, activation_func, input_dim=768):
        super().__init__()

        self.input_dim = config.hidden_size + 10
        self.hidden_dim = config.generator_hdim
        self.output_dim = config.hidden_size * config.adapter_dim * 2 + config.hidden_size + config.adapter_dim
        self.linear1 = linear(self.input_dim, self.hidden_dim)
        self.activation_fn = ACT2FN['relu']
        # output weights
        self.weight_up = nn.Linear(self.hidden_dim, config.hidden_size * config.adapter_dim)
        self.weight_down = nn.Linear(self.hidden_dim, config.hidden_size * config.adapter_dim)
        self.bias_up = nn.Linear(self.hidden_dim, config.hidden_size)
        self.bias_down = nn.Linear(self.hidden_dim, config.adapter_dim)
        # init weights
        # technically wrong init but ah well
        hyperfanin_init_weight(self.weight_up, self.hidden_dim, config.adapter_dim)
        hyperfanin_init_weight(self.weight_down, self.hidden_dim, config.hidden_size)
        hyperfanin_init_bias(self.bias_up, self.hidden_dim)
        hyperfanin_init_bias(self.bias_down, self.hidden_dim)

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation_fn(x)
        return (
            self.weight_up(x),
            self.weight_down(x),
            self.bias_up(x),
            self.bias_down(x)
        )


class ParameterGenerator(nn.Module):
    def __init__(self, config, activation_func):
        super().__init__()

        self.config = config
        self.activation_function = activation_func

        self.embed = nn.Embedding(config.num_hidden_layers, 10)
        self.decoder = SimpleGenerator(config, self.activation_function)

    def forward(self, hidden_inputs):
        layers = []
        for i in range(self.config.num_hidden_layers):
            embed = self.embed(torch.tensor([i], device=hidden_inputs.device))
            embed = embed.repeat(hidden_inputs.size(0), 1)
            layers.append(self.decoder(torch.cat([hidden_inputs, embed], dim=1)))
        return layers