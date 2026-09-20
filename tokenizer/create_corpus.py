import json
from pathlib import Path


INPUT_FILE = Path("dataset/processed/train.jsonl")
OUTPUT_FILE = Path("tokenizer/corpus.txt")


def main():

    print("=" * 60)
    print("CREATING TOKENIZER CORPUS")
    print("=" * 60)

    if not INPUT_FILE.exists():
        print("Dataset not found:")
        print(INPUT_FILE)
        return

    count = 0

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        INPUT_FILE,
        "r",
        encoding="utf-8"
    ) as input_file, open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as output_file:

        for line in input_file:

            line = line.strip()

            if not line:
                continue

            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue

            question = item.get("question", "").strip()
            answer = item.get("answer", "").strip()

            if not question or not answer:
                continue

            # Conversation format
            text = (
                "[USER] "
                + question
                + " "
                + "[ASSISTANT] "
                + answer
                + " [EOS]"
            )

            output_file.write(text + "\n")

            count += 1

    print("\nExamples written:", count)

    print("\nCorpus saved to:")
    print(OUTPUT_FILE)

    print("\nTokenizer corpus creation complete!")


if __name__ == "__main__":
    main()