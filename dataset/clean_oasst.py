import json
from pathlib import Path

INPUT_FILE = Path("dataset/raw/public/oasst1_ready.jsonl")
OUTPUT_FILE = Path("dataset/raw/public/oasst_clean.jsonl")


MIN_QUALITY = 0.60
MIN_LENGTH = 10
MAX_LENGTH = 4000


def good_message(item):
    if item.get("deleted", False):
        return False

    if item.get("lang") != "en":
        return False

    text = item.get("text", "").strip()

    if len(text) < MIN_LENGTH:
        return False

    if len(text) > MAX_LENGTH:
        return False

    return True


def good_labels(item):
    labels = item.get("labels", {})

    # Quality
    quality = labels.get("quality", {}).get("value", 0)

    if quality < MIN_QUALITY:
        return False

    # Remove undesirable categories
    blocked = [
        "spam",
        "pii",
        "not_appropriate",
        "hate_speech",
        "sexual_content"
    ]

    for label in blocked:
        value = labels.get(label, {}).get("value", 0)

        if value > 0.5:
            return False

    return True


def load_messages():

    messages = {}

    print("Loading OASST1...")

    with open(INPUT_FILE, "r", encoding="utf-8") as file:

        for line in file:

            line = line.strip()

            if not line:
                continue

            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue

            messages[item["message_id"]] = item

    print("Messages loaded:", len(messages))

    return messages


def create_pairs(messages):

    pairs = []

    for message_id, assistant in messages.items():

        if assistant.get("role") != "assistant":
            continue

        if not good_message(assistant):
            continue

        if not good_labels(assistant):
            continue

        parent_id = assistant.get("parent_id")

        if not parent_id:
            continue

        parent = messages.get(parent_id)

        if parent is None:
            continue

        if parent.get("role") != "prompter":
            continue

        if not good_message(parent):
            continue

        if not good_labels(parent):
            continue

        question = parent["text"].strip()
        answer = assistant["text"].strip()

        pairs.append({
            "question": question,
            "answer": answer,
            "category": "general",
            "language": "en",
            "level": "intermediate",
            "type": "conversation"
        })

    return pairs


def remove_duplicates(records):

    unique = {}

    for item in records:

        question = item["question"].lower().strip()
        answer = item["answer"].lower().strip()

        key = (question, answer)

        if key not in unique:
            unique[key] = item

    return list(unique.values())


def save(records):

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        for item in records:

            file.write(
                json.dumps(
                    item,
                    ensure_ascii=False
                ) + "\n"
            )


def main():

    print("=" * 60)
    print("OASST DATASET CLEANING")
    print("=" * 60)

    messages = load_messages()

    pairs = create_pairs(messages)

    print("Pairs after quality filtering:", len(pairs))

    pairs = remove_duplicates(pairs)

    print("Pairs after duplicate removal:", len(pairs))

    save(pairs)

    print("\nClean dataset saved to:")
    print(OUTPUT_FILE)

    print("\nDone!")


if __name__ == "__main__":
    main()