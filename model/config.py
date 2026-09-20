from dataclasses import dataclass


@dataclass
class ModelConfig:

    vocab_size: int = 8000

    max_seq_len: int = 256

    d_model: int = 256

    n_heads: int = 4

    n_layers: int = 4

    dropout: float = 0.1

    multiple_of: int = 256