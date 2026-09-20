from tokenizers import Tokenizer


TOKENIZER_FILE = "tokenizer/tokenizer.json"


tokenizer = Tokenizer.from_file(
    TOKENIZER_FILE
)


def encode(text):

    output = tokenizer.encode(text)

    return output.ids


def decode(ids):

    return tokenizer.decode(ids)


if __name__ == "__main__":

    examples = [
        "What is artificial intelligence?",
        "Explain machine learning in simple words.",
        "కృత్రిమ మేధస్సు అంటే ఏమిటి?",
        "मशीन लर्निंग क्या है?"
    ]

    for text in examples:

        ids = encode(text)

        decoded = decode(ids)

        print("\n" + "=" * 60)

        print("TEXT:")
        print(text)

        print("\nTOKEN IDS:")
        print(ids)

        print("\nNUMBER OF TOKENS:")
        print(len(ids))

        print("\nDECODED:")
        print(decoded)