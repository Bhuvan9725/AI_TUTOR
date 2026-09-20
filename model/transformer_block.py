import torch.nn as nn

from .rmsnorm import RMSNorm
from .attention import CausalSelfAttention
from .swiglu import SwiGLU


class TransformerBlock(nn.Module):

    def __init__(self, config):

        super().__init__()

        self.attention_norm = RMSNorm(
            config.d_model
        )

        self.attention = CausalSelfAttention(
            config
        )

        self.ffn_norm = RMSNorm(
            config.d_model
        )

        self.ffn = SwiGLU(
            config
        )

    def forward(self, x):

        # Attention + residual connection
        x = x + self.attention(
            self.attention_norm(x)
        )

        # Feed-forward + residual connection
        x = x + self.ffn(
            self.ffn_norm(x)
        )

        return x