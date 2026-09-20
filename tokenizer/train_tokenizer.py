from pathlib import Path

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.trainers import BpeTrainer


CORPUS_FILE = "tokenizer/corpus.txt"
TOKENIZER_FILE = "tokenizer/tokenizer.json"


VOCAB_SIZE = 8000


def main():

    print("=" * 60)
    print("TRAINING CUSTOM BPE TOKENIZER")
    print("=" * 60)

    tokenizer = Tokenizer(
        BPE(
            unk_token="[UNK]"
        )
    )

    tokenizer.pre_tokenizer = ByteLevel(
        add_prefix_space=True
    )

    tokenizer.decoder = ByteLevelDecoder()

    trainer = BpeTrainer(
        vocab_size=VOCAB_SIZE,

        special_tokens=[
            "[PAD]",
            "[UNK]",
            "[BOS]",
            "[EOS]",
            "[USER]",
            "[ASSISTANT]"
        ],

        min_frequency=2,

        show_progress=True
    )

    print("\nTraining tokenizer...")

    tokenizer.train(
        files=[CORPUS_FILE],
        trainer=trainer
    )

    tokenizer.save(TOKENIZER_FILE)

    print("\n" + "=" * 60)
    print("TOKENIZER TRAINING COMPLETE")
    print("=" * 60)

    print("Vocabulary size:", tokenizer.get_vocab_size())

    print("\nSaved to:")
    print(TOKENIZER_FILE)


if __name__ == "__main__":
    main()