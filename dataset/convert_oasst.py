import json
from pathlib import Path

INPUT_FILE = Path("dataset/raw/public/oasst1_ready.jsonl")
OUTPUT_FILE = Path("dataset/raw/public/oasst_pairs.jsonl")


def load_messages():
    messages = {}

    print("=" * 60)
    print("Reading OASST1...")
    print("=" * 60)

    with open(INPUT_FILE, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()

            if not line:
                continue

            try:
                item = json.loads(line)
                messages[item["message_id"]] = item
            except (json.JSONDecodeError, KeyError):
                continue

    print("Messages loaded:", len(messages))
    return messages


def is_good_message(item):
    # Remove deleted messages
    if item.get("deleted", False):
        return False

    # For this first version, keep English
    if item.get("lang") != "en":
        return False

    text = item.get("text", "").strip()

    # Remove extremely short messages
    if len(text) < 10:
        return False

    # Remove extremely long messages
    if len(text) > 5000:
        return False

    return True


def convert(messages):
    pairs = []

    print("\nConverting conversation messages into Q&A pairs...")

    for message_id, assistant in messages.items():

        # We only want assistant responses
        if assistant.get("role") != "assistant":
            continue

        if not is_good_message(assistant):
            continue

        # Find the message that the assistant replied to
        parent_id = assistant.get("parent_id")

        if not parent_id:
            continue

        parent = messages.get(parent_id)

        if parent is None:
            continue

        # Parent must be the user's question
        if parent.get("role") != "prompter":
            continue

        if not is_good_message(parent):
            continue

        question = parent["text"].strip()
        answer = assistant["text"].strip()

        pair = {
            "question": question,
            "answer": answer,
            "category": "general",
            "language": "en",
            "level": "intermediate",
            "type": "conversation"
        }

        pairs.append(pair)

    return pairs


def remove_duplicates(records):
    unique = {}

    for item in records:

        key = (
            item["question"].lower().strip(),
            item["answer"].lower().strip()
        )

        if key not in unique:
            unique[key] = item

    return list(unique.values())


def save(records):
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as file:

        for item in records:
            file.write(
                json.dumps(
                    item,
                    ensure_ascii=False
                ) + "\n"
            )


def main():

    messages = load_messages()

    pairs = convert(messages)

    print("Pairs before duplicate removal:", len(pairs))

    pairs = remove_duplicates(pairs)

    print("Pairs after duplicate removal:", len(pairs))

    save(pairs)

    print("\n" + "=" * 60)
    print("Conversion complete!")
    print("=" * 60)

    print("Saved to:")
    print(OUTPUT_FILE)


if __name__ == "__main__":
    main()