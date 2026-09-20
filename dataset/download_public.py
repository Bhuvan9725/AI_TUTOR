import urllib.request
import gzip
import shutil
from pathlib import Path


URL = (
    "https://huggingface.co/datasets/"
    "OpenAssistant/oasst1/resolve/main/"
    "2023-04-12_oasst_ready.messages.jsonl.gz"
)

RAW_DIR = Path("dataset/raw/public")
COMPRESSED_FILE = RAW_DIR / "oasst1_ready.jsonl.gz"
OUTPUT_FILE = RAW_DIR / "oasst1_ready.jsonl"


def download():

    RAW_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    print("=" * 60)
    print("Downloading OpenAssistant OASST1")
    print("=" * 60)

    print("\nDownloading file...")

    urllib.request.urlretrieve(
        URL,
        COMPRESSED_FILE
    )

    print("Download completed.")
    print("File:", COMPRESSED_FILE)

    print("\nExtracting JSONL file...")

    with gzip.open(
        COMPRESSED_FILE,
        "rb"
    ) as source:

        with open(
            OUTPUT_FILE,
            "wb"
        ) as target:

            shutil.copyfileobj(
                source,
                target
            )

    print("Extraction completed.")
    print("JSONL:", OUTPUT_FILE)

    print("\nDone!")


if __name__ == "__main__":
    download()