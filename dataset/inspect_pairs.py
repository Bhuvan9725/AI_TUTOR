import json

FILE = "dataset/raw/public/oasst_pairs.jsonl"

with open(FILE, "r", encoding="utf-8") as file:

    for i in range(5):

        line = file.readline()

        if not line:
            break

        item = json.loads(line)

        print("\n" + "=" * 70)
        print(f"EXAMPLE {i + 1}")
        print("=" * 70)

        print("QUESTION:")
        print(item["question"])

        print("\nANSWER:")
        print(item["answer"])

        print("\nLANGUAGE:", item["language"])
        print("CATEGORY:", item["category"])
        print("LEVEL:", item["level"])