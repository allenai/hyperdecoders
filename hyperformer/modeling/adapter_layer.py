import torch
from torch import nn
import math


class AdapterLayer(nn.Module):
    def __init__(self, config, is_encoder=False):
        super().__init__()
        self.adapter_dim = (
            config.encoder_adapter_dim if is_encoder else config.decoder_adapter_dim
        )
        hidden_size = config.hidden_size
        self.input_dim = config.hidden_size
        self.output_dim = config.hidden_size
        # insertion weights
        self.adapter_down_weight = None
        self.adapter_down_bias = None
        self.adapter_up_weight = None
        self.adapter_up_bias = None
        self.hidden_act = nn.ReLU()
        # learnt adapter + inits for it
        self.adapter_down_manual = nn.Linear(hidden_size, self.adapter_dim)
        self.adapter_up_manual = nn.Linear(self.adapter_dim, hidden_size)
        nn.init.xavier_uniform_(self.adapter_up_manual.weight, gain=1e-4)
        nn.init.xavier_uniform_(self.adapter_down_manual.weight, gain=1e-4)
        nn.init.constant_(self.adapter_up_manual.bias, 0.0)
        nn.init.constant_(self.adapter_down_manual.bias, 0.0)

    def clear_adapter(self):
        self.adapter_down_weight = None
        self.adapter_down_bias = None
        self.adapter_up_weight = None
        self.adapter_up_bias = None

    def apply_adapter_params(self, bsz, uw, dw, ub, db):
        self.adapter_down_weight = dw.view(bsz, self.input_dim, self.adapter_dim)
        self.adapter_down_bias = db.view(bsz, self.adapter_dim)
        self.adapter_up_weight = uw.view(bsz, self.adapter_dim, self.output_dim)
        self.adapter_up_bias = ub.view(bsz, self.output_dim)

    def forward(self, x):
        if self.adapter_down_weight is not None:
            x = (x @ self.adapter_down_weight) + self.adapter_down_bias.unsqueeze(1)
            x = self.hidden_act(x)
            x = (x @ self.adapter_up_weight) + self.adapter_up_bias.unsqueeze(1)
        else:
            x = self.adapter_down_manual(x)
            x = self.hidden_act(x)
            x = self.adapter_up_manual(x)
        return x  # no residual connection - we let the user of this layer decide that


class TaskSpecificAdapterLayer(nn.Module):
    def __init__(self, config, task_list, is_encoder=False):
        super().__init__()
        self.adapter_dim = (
            config.encoder_adapter_dim if is_encoder else config.decoder_adapter_dim
        )
        hidden_size = config.hidden_size
        task_list = config.tasks
        self.input_dim = hidden_size
        self.output_dim = hidden_size
        self.hidden_act = nn.ReLU()
        # learnt adapter + inits for it
        self.adapter_down_manual_weight = nn.Parameter(
            torch.randn(len(task_list), hidden_size, self.adapter_dim)
        )
        self.adapter_down_manual_bias = nn.Parameter(
            torch.randn(len(task_list), 1, self.adapter_dim)
        )
        self.adapter_up_manual_weight = nn.Parameter(
            torch.randn(len(task_list), self.adapter_dim, hidden_size)
        )
        self.adapter_up_manual_bias = nn.Parameter(
            torch.randn(len(task_list), 1, hidden_size)
        )

        nn.init.xavier_uniform_(self.adapter_down_manual_weight, gain=1e-4)
        nn.init.constant_(self.adapter_down_manual_bias, 0.0)
        nn.init.xavier_uniform_(self.adapter_up_manual_weight, gain=1e-4)
        nn.init.constant_(self.adapter_up_manual_bias, 0.0)
        # hacky method for setting task specific adapters
        self.adapter_down_weight_holder = None
        self.adapter_down_bias_holder = None
        self.adapter_up_weight_holder = None
        self.adapter_up_bias_holder = None

    def clear_adapter(self):
        self.adapter_down_weight_holder = None
        self.adapter_down_bias_holder = None
        self.adapter_up_weight_holder = None
        self.adapter_up_bias_holder = None

    def set_indices(self, indices):
        self.adapter_down_weight_holder = self.adapter_down_manual_weight[indices]
        self.adapter_down_bias_holder = self.adapter_down_manual_bias[indices]
        self.adapter_up_weight_holder = self.adapter_up_manual_weight[indices]
        self.adapter_up_bias_holder = self.adapter_up_manual_bias[indices]

    def forward(self, x):
        x = (
            torch.bmm(x, self.adapter_down_weight_holder)
            + self.adapter_down_bias_holder
        )
        x = self.hidden_act(x)
        x = torch.bmm(x, self.adapter_up_weight_holder) + self.adapter_up_bias_holder
        return x
