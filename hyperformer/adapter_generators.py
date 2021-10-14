import torch.nn as nn
from transformers.activations import ACT2FN

def Linear(in_features, out_features, bias=True):
    m = nn.Linear(in_features, out_features, bias)
    nn.init.xavier_uniform_(m.weight, gain=0.0000001)
    if bias:
        nn.init.constant_(m.bias, 0.0)
    return m

class SimpleGenerator(nn.Module):
    # takes in a encoded task description and generates parameters of an adapter
    def __init__(self, config, activation_func, input_dim=768):
        super().__init__()

        self.input_dim = input_dim # config.hidden_size
        self.hidden_dim = 128
        self.output_dim = config.hidden_size * config.adapter_dim * 2 + config.hidden_size + config.adapter_dim
        self.linear1 = Linear(self.input_dim, self.hidden_dim)
        self.activation_fn = ACT2FN[activation_func]
        self.linear2 = Linear(self.hidden_dim, self.output_dim)

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation_fn(x)
        x = self.linear2(x)
        return x.view(x.size(0), -1)


class ParameterGenerator(nn.Module):
    def __init__(self, config, activation_func, input_dim):
        super().__init__()

        self.config = config
        self.activation_function = activation_func

        self.decoders = nn.ModuleList([
            SimpleGenerator(config, self.activation_function, input_dim) for _ in range(config.num_hidden_layers * 2) # two per layer
        ])

    def decode(self, sr):
        return [one_decoder(sr) for one_decoder in self.decoders]

    def forward(self, hidden_inputs):
        params = self.decode(hidden_inputs)
        return params