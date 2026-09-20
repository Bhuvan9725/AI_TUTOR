import torch.nn as nn
import torch.nn.functional as F


class SwiGLU(nn.Module):

    def __init__(self, config):

        super().__init__()

        hidden_dim = config.d_model * 4

        self.gate_proj = nn.Linear(
            config.d_model,
            hidden_dim,
            bias=False
        )

        self.up_proj = nn.Linear(
            config.d_model,
            hidden_dim,
            bias=False
        )

        self.down_proj = nn.Linear(
            hidden_dim,
            config.d_model,
            bias=False
        )

    def forward(self, x):

        gate = F.silu(
            self.gate_proj(x)
        )

        up = self.up_proj(x)

        return self.down_proj(
            gate * up
        )