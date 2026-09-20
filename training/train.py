import math
import os

import torch
import torch.nn as nn

from torch.utils.data import DataLoader

from model.config import ModelConfig
from model.transformer import AITransformer

from training.config import TrainingConfig
from training.dataset import TutorDataset


def calculate_perplexity(loss):

    try:
        return math.exp(loss)
    except OverflowError:
        return float("inf")


def evaluate(
    model,
    loader,
    device
):

    model.eval()

    total_loss = 0.0
    batches = 0

    criterion = nn.CrossEntropyLoss(
        ignore_index=-100
    )

    with torch.no_grad():

        for batch in loader:

            input_ids = batch[
                "input_ids"
            ].to(device)

            targets = batch[
                "targets"
            ].to(device)

            logits = model(
                input_ids
            )

            loss = criterion(
                logits.reshape(
                    -1,
                    logits.size(-1)
                ),
                targets.reshape(-1)
            )

            total_loss += loss.item()

            batches += 1

    model.train()

    if batches == 0:
        return 0.0

    return total_loss / batches


def get_learning_rate(
    step,
    total_steps,
    config
):

    if step < config.warmup_steps:

        return (
            config.learning_rate
            * step
            / max(
                config.warmup_steps,
                1
            )
        )

    progress = (
        step - config.warmup_steps
    ) / max(
        total_steps - config.warmup_steps,
        1
    )

    progress = min(
        max(progress, 0.0),
        1.0
    )

    cosine = (
        0.5
        * (
            1.0
            + math.cos(
                math.pi * progress
            )
        )
    )

    return (
        config.min_learning_rate
        + (
            config.learning_rate
            - config.min_learning_rate
        )
        * cosine
    )


def set_learning_rate(
    optimizer,
    learning_rate
):

    for param_group in optimizer.param_groups:
        param_group["lr"] = learning_rate


def main():

    config = TrainingConfig()

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("=" * 60)
    print("AI TUTOR TRAINING - VERSION 2")
    print("=" * 60)

    print("\nDevice:", device)

    train_dataset = TutorDataset(
        config.train_file,
        config.tokenizer_file,
        config.max_seq_len
    )

    validation_dataset = TutorDataset(
        config.validation_file,
        config.tokenizer_file,
        config.max_seq_len
    )

    print(
        "Training examples:",
        len(train_dataset)
    )

    print(
        "Validation examples:",
        len(validation_dataset)
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=config.batch_size,
        shuffle=False
    )

    model_config = ModelConfig()

    model = AITransformer(
        model_config
    ).to(device)

    parameters = sum(
        p.numel()
        for p in model.parameters()
    )

    print(
        f"Model parameters: "
        f"{parameters:,}"
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )

    criterion = nn.CrossEntropyLoss(
        ignore_index=-100
    )

    total_steps = (
        len(train_loader)
        * config.epochs
    )

    global_step = 0

    best_val_loss = float("inf")

    os.makedirs(
        config.checkpoint_dir,
        exist_ok=True
    )

    for epoch in range(
        config.epochs
    ):

        model.train()

        total_loss = 0.0

        print(
            "\n"
            + "=" * 60
        )

        print(
            f"EPOCH "
            f"{epoch + 1}/"
            f"{config.epochs}"
        )

        print(
            "=" * 60
        )

        for step, batch in enumerate(
            train_loader,
            start=1
        ):

            global_step += 1

            learning_rate = (
                get_learning_rate(
                    global_step,
                    total_steps,
                    config
                )
            )

            set_learning_rate(
                optimizer,
                learning_rate
            )

            input_ids = batch[
                "input_ids"
            ].to(device)

            targets = batch[
                "targets"
            ].to(device)

            optimizer.zero_grad()

            logits = model(
                input_ids
            )

            loss = criterion(
                logits.reshape(
                    -1,
                    logits.size(-1)
                ),
                targets.reshape(-1)
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                config.gradient_clip
            )

            optimizer.step()

            total_loss += loss.item()

            if (
                step % config.log_every == 0
                or step == 1
            ):

                average_loss = (
                    total_loss / step
                )

                print(
                    f"Step "
                    f"{step}/"
                    f"{len(train_loader)} "
                    f"| Loss: "
                    f"{average_loss:.4f} "
                    f"| LR: "
                    f"{learning_rate:.7f}"
                )

        train_loss = (
            total_loss
            / max(
                len(train_loader),
                1
            )
        )

        val_loss = evaluate(
            model,
            validation_loader,
            device
        )

        train_ppl = calculate_perplexity(
            train_loss
        )

        val_ppl = calculate_perplexity(
            val_loss
        )

        print(
            "\nTrain Loss:",
            f"{train_loss:.4f}"
        )

        print(
            "Validation Loss:",
            f"{val_loss:.4f}"
        )

        print(
            "Train Perplexity:",
            f"{train_ppl:.2f}"
        )

        print(
            "Validation Perplexity:",
            f"{val_ppl:.2f}"
        )

        checkpoint = {
            "model_state_dict":
                model.state_dict(),

            "model_config":
                model_config.__dict__,

            "training_config":
                config.__dict__,

            "validation_loss":
                val_loss,

            "train_loss":
                train_loss,

            "epoch":
                epoch + 1,

            "global_step":
                global_step
        }

        # Save latest checkpoint
        latest_path = os.path.join(
            config.checkpoint_dir,
            "latest_model.pt"
        )

        torch.save(
            checkpoint,
            latest_path
        )

        print(
            "\n✓ Latest checkpoint saved:"
        )

        print(latest_path)

        # Save best checkpoint
        if val_loss < best_val_loss:

            best_val_loss = val_loss

            best_path = os.path.join(
                config.checkpoint_dir,
                "best_model.pt"
            )

            torch.save(
                checkpoint,
                best_path
            )

            print(
                "✓ New best model saved:"
            )

            print(best_path)

    print(
        "\n"
        + "=" * 60
    )

    print(
        "TRAINING COMPLETED"
    )

    print(
        "=" * 60
    )

    print(
        "\nBest validation loss:",
        f"{best_val_loss:.4f}"
    )


if __name__ == "__main__":
    main()