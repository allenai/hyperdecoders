from torch import nn


class AdapterLayer(nn.Module):
    def __init__(self, hidden_size, adapter_dim):
        super().__init__()
        self.adapter_dim = adapter_dim
        self.input_dim = hidden_size
        self.output_dim = hidden_size
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
