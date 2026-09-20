import json

FILE = "dataset/raw/public/oasst1_ready.jsonl"


with open(
    FILE,
    "r",
    encoding="utf-8"
) as f:

    for i in range(3):

        line = f.readline()

        if not line:
            break

        item = json.loads(line)

        print("\n" + "=" * 70)
        print("RECORD", i + 1)
        print("=" * 70)

        print(json.dumps(
            item,
            indent=2,
            ensure_ascii=False
        ))