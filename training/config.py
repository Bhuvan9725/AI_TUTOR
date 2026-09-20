from dataclasses import dataclass


@dataclass
class TrainingConfig:

    train_file: str = (
        "dataset/processed/train.jsonl"
    )

    validation_file: str = (
        "dataset/processed/validation.jsonl"
    )

    tokenizer_file: str = (
        "tokenizer/tokenizer.json"
    )

    checkpoint_dir: str = (
        "checkpoints"
    )

    batch_size: int = 4

    learning_rate: float = 3e-4

    min_learning_rate: float = 3e-5

    weight_decay: float = 0.1

    epochs: int = 5

    max_seq_len: int = 256

    gradient_clip: float = 1.0

    warmup_steps: int = 100

    log_every: int = 50

    save_every_epoch: bool = True