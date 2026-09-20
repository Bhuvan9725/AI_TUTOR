import torch
import torch.nn as nn


class RMSNorm(nn.Module):

    def __init__(self, dim, eps=1e-6):

        super().__init__()

        self.weight = nn.Parameter(
            torch.ones(dim)
        )

        self.eps = eps

    def forward(self, x):

        rms = torch.rsqrt(
            x.pow(2).mean(
                dim=-1,
                keepdim=True
            ) + self.eps
        )

        return x * rms * self.weight