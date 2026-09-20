import torch
import torch.nn as nn
import torch.nn.functional as F

from .rope import (
    precompute_rope,
    apply_rope
)


class CausalSelfAttention(nn.Module):

    def __init__(self, config):

        super().__init__()

        self.n_heads = config.n_heads

        self.d_model = config.d_model

        self.head_dim = (
            config.d_model // config.n_heads
        )

        self.q_proj = nn.Linear(
            config.d_model,
            config.d_model,
            bias=False
        )

        self.k_proj = nn.Linear(
            config.d_model,
            config.d_model,
            bias=False
        )

        self.v_proj = nn.Linear(
            config.d_model,
            config.d_model,
            bias=False
        )

        self.out_proj = nn.Linear(
            config.d_model,
            config.d_model,
            bias=False
        )

        cos, sin = precompute_rope(
            config.max_seq_len,
            self.head_dim
        )

        self.register_buffer(
            "rope_cos",
            cos,
            persistent=False
        )

        self.register_buffer(
            "rope_sin",
            sin,
            persistent=False
        )

        self.dropout = config.dropout

    def forward(self, x):

        batch_size, seq_len, _ = x.shape

        q = self.q_proj(x)

        k = self.k_proj(x)

        v = self.v_proj(x)

        q = q.view(
            batch_size,
            seq_len,
            self.n_heads,
            self.head_dim
        ).transpose(1, 2)

        k = k.view(
            batch_size,
            seq_len,
            self.n_heads,
            self.head_dim
        ).transpose(1, 2)

        v = v.view(
            batch_size,
            seq_len,
            self.n_heads,
            self.head_dim
        ).transpose(1, 2)

        q = apply_rope(
            q,
            self.rope_cos,
            self.rope_sin
        )

        k = apply_rope(
            k,
            self.rope_cos,
            self.rope_sin
        )

        output = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=(
                self.dropout
                if self.training
                else 0.0
            ),
            is_causal=True
        )

        output = output.transpose(
            1,
            2
        ).contiguous()

        output = output.view(
            batch_size,
            seq_len,
            self.d_model
        )

        return self.out_proj(output)