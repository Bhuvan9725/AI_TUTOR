import torch

from model.config import ModelConfig
from model.transformer import AITransformer

from tokenizers import Tokenizer


CHECKPOINT = "checkpoints/best_model.pt"
TOKENIZER_FILE = "tokenizer/tokenizer.json"

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


def load_model():

    checkpoint = torch.load(
        CHECKPOINT,
        map_location=DEVICE
    )

    config = ModelConfig(
        **checkpoint["model_config"]
    )

    model = AITransformer(config)

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.to(DEVICE)
    model.eval()

    return model


def generate_answer(
    model,
    tokenizer,
    question,
    max_new_tokens=80,
    temperature=0.7,
    top_k=40
):

    prompt = (
        "[USER] "
        + question
        + " "
        + "[ASSISTANT]"
    )

    encoded = tokenizer.encode(prompt)

    input_ids = torch.tensor(
        [encoded.ids],
        dtype=torch.long,
        device=DEVICE
    )

    generated = model.generate(
        input_ids,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k
    )

    generated_ids = generated[0].tolist()

    text = tokenizer.decode(
        generated_ids
    )

    # Extract assistant response
    if "[ASSISTANT]" in text:

        text = text.split(
            "[ASSISTANT]",
            1
        )[1]

    if "[EOS]" in text:

        text = text.split(
            "[EOS]",
            1
        )[0]

    return text.strip()


def main():

    print("=" * 60)
    print("AI TUTOR - INFERENCE TEST")
    print("=" * 60)

    print("\nDevice:", DEVICE)

    tokenizer = Tokenizer.from_file(
        TOKENIZER_FILE
    )

    model = load_model()

    print("Model loaded successfully.")

    questions = [
        "What is artificial intelligence?",
        "Explain machine learning in simple words.",
        "What is a stack in data structures?",
        "What is a variable in Python?"
    ]

    for question in questions:

        print("\n" + "=" * 60)

        print("USER:")
        print(question)

        answer = generate_answer(
            model,
            tokenizer,
            question
        )

        print("\nAI TUTOR:")
        print(answer)


if __name__ == "__main__":
    main()