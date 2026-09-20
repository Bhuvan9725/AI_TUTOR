import torch
import torch.nn as nn

from .config import ModelConfig
from .rmsnorm import RMSNorm
from .transformer_block import TransformerBlock


class AITransformer(nn.Module):

    def __init__(self, config=None):

        super().__init__()

        if config is None:
            config = ModelConfig()

        self.config = config

        # Token embedding
        self.token_embedding = nn.Embedding(
            config.vocab_size,
            config.d_model
        )

        # Transformer blocks
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(config)
                for _ in range(config.n_layers)
            ]
        )

        # Final normalization
        self.final_norm = RMSNorm(
            config.d_model
        )

        # Language model head
        self.lm_head = nn.Linear(
            config.d_model,
            config.vocab_size,
            bias=False
        )

        # Weight tying
        self.lm_head.weight = (
            self.token_embedding.weight
        )

        self.apply(self._init_weights)

    def _init_weights(self, module):

        if isinstance(
            module,
            nn.Linear
        ):

            nn.init.normal_(
                module.weight,
                mean=0.0,
                std=0.02
            )

            if module.bias is not None:
                nn.init.zeros_(
                    module.bias
                )

        elif isinstance(
            module,
            nn.Embedding
        ):

            nn.init.normal_(
                module.weight,
                mean=0.0,
                std=0.02
            )

    def forward(self, input_ids):

        x = self.token_embedding(
            input_ids
        )

        for block in self.blocks:

            x = block(x)

        x = self.final_norm(x)

        logits = self.lm_head(x)

        return logits

    @torch.no_grad()
    def generate(
        self,
        input_ids,
        max_new_tokens=50,
        temperature=0.8,
        top_k=50
    ):

        self.eval()

        for _ in range(max_new_tokens):

            # Keep only the latest context
            input_ids_cond = input_ids[
                :, -self.config.max_seq_len:
            ]

            logits = self(
                input_ids_cond
            )

            # Last token
            logits = logits[:, -1, :]

            # Temperature
            logits = logits / temperature

            # Top-k filtering
            if top_k is not None:

                values, _ = torch.topk(
                    logits,
                    min(
                        top_k,
                        logits.size(-1)
                    )
                )

                min_value = values[:, -1].unsqueeze(-1)

                logits = torch.where(
                    logits < min_value,
                    torch.full_like(
                        logits,
                        float("-inf")
                    ),
                    logits
                )

            probabilities = torch.softmax(
                logits,
                dim=-1
            )

            next_token = torch.multinomial(
                probabilities,
                num_samples=1
            )

            input_ids = torch.cat(
                [
                    input_ids,
                    next_token
                ],
                dim=1
            )

        return input_ids