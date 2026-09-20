import json
import random
from pathlib import Path
from collections import Counter


OASST_FILE = Path("dataset/raw/public/oasst_clean.jsonl")
CUSTOM_FILE = Path("dataset/raw/custom/custom_dataset.jsonl")

MASTER_FILE = Path("dataset/master_dataset.jsonl")

TRAIN_FILE = Path("dataset/processed/train.jsonl")
VAL_FILE = Path("dataset/processed/validation.jsonl")
TEST_FILE = Path("dataset/processed/test.jsonl")

SEED = 42


def load_jsonl(file_path):
    records = []

    if not file_path.exists():
        print("File not found:", file_path)
        return records

    with open(file_path, "r", encoding="utf-8") as file:

        for line in file:

            line = line.strip()

            if not line:
                continue

            try:
                item = json.loads(line)
                records.append(item)

            except json.JSONDecodeError:
                continue

    return records


def normalize_record(item):

    question = item.get("question", "").strip()
    answer = item.get("answer", "").strip()

    if not question or not answer:
        return None

    return {
        "question": question,
        "answer": answer,
        "category": item.get("category", "general"),
        "language": item.get("language", "en"),
        "level": item.get("level", "intermediate"),
        "type": item.get("type", "conversation")
    }


def remove_duplicates(records):

    unique = {}

    for item in records:

        key = (
            item["question"].lower().strip(),
            item["answer"].lower().strip(),
            item["language"]
        )

        if key not in unique:
            unique[key] = item

    return list(unique.values())


def save_jsonl(records, file_path):

    file_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        file_path,
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


def print_statistics(records):

    print("\n" + "=" * 60)
    print("DATASET STATISTICS")
    print("=" * 60)

    language_counts = Counter(
        item["language"]
        for item in records
    )

    category_counts = Counter(
        item["category"]
        for item in records
    )

    level_counts = Counter(
        item["level"]
        for item in records
    )

    print("\nLanguages:")

    for language, count in language_counts.items():
        print(f"  {language}: {count}")

    print("\nCategories:")

    for category, count in category_counts.most_common():
        print(f"  {category}: {count}")

    print("\nLevels:")

    for level, count in level_counts.items():
        print(f"  {level}: {count}")


def split_dataset(records):

    random.seed(SEED)

    random.shuffle(records)

    total = len(records)

    train_end = int(total * 0.80)
    val_end = int(total * 0.90)

    train = records[:train_end]

    validation = records[train_end:val_end]

    test = records[val_end:]

    return train, validation, test


def main():

    print("=" * 60)
    print("AI TUTOR DATASET PREPARATION")
    print("=" * 60)

    print("\nLoading OASST cleaned dataset...")

    oasst = load_jsonl(OASST_FILE)

    print("OASST records:", len(oasst))

    print("\nLoading custom dataset...")

    custom = load_jsonl(CUSTOM_FILE)

    print("Custom records:", len(custom))

    # Normalize
    oasst = [
        normalize_record(item)
        for item in oasst
    ]

    custom = [
        normalize_record(item)
        for item in custom
    ]

    # Remove invalid
    oasst = [
        item for item in oasst
        if item is not None
    ]

    custom = [
        item for item in custom
        if item is not None
    ]

    print("\nCombining datasets...")

    combined = oasst + custom

    print("Combined records:", len(combined))

    print("\nRemoving duplicates...")

    combined = remove_duplicates(combined)

    print("Final unique records:", len(combined))

    # Statistics
    print_statistics(combined)

    # Save master dataset
    save_jsonl(
        combined,
        MASTER_FILE
    )

    print("\nMaster dataset saved:")
    print(MASTER_FILE)

    # Split
    train, validation, test = split_dataset(
        combined
    )

    save_jsonl(
        train,
        TRAIN_FILE
    )

    save_jsonl(
        validation,
        VAL_FILE
    )

    save_jsonl(
        test,
        TEST_FILE
    )

    print("\n" + "=" * 60)
    print("DATASET SPLIT")
    print("=" * 60)

    print("Training:", len(train))
    print("Validation:", len(validation))
    print("Testing:", len(test))

    print("\nFiles created:")

    print(TRAIN_FILE)
    print(VAL_FILE)
    print(TEST_FILE)

    print("\nDataset preparation complete!")


if __name__ == "__main__":
    main()