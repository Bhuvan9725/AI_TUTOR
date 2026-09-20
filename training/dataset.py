import json
import torch
from torch.utils.data import Dataset
from tokenizers import Tokenizer


class TutorDataset(Dataset):

    def __init__(
        self,
        data_file,
        tokenizer_file,
        max_seq_len=256
    ):

        self.max_seq_len = max_seq_len

        self.tokenizer = Tokenizer.from_file(
            tokenizer_file
        )

        self.records = []

        with open(
            data_file,
            "r",
            encoding="utf-8"
        ) as file:

            for line in file:

                line = line.strip()

                if not line:
                    continue

                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    continue

                question = item.get(
                    "question",
                    ""
                ).strip()

                answer = item.get(
                    "answer",
                    ""
                ).strip()

                if not question or not answer:
                    continue

                text = (
                    "[USER] "
                    + question
                    + " "
                    + "[ASSISTANT] "
                    + answer
                    + " [EOS]"
                )

                self.records.append(text)

    def __len__(self):

        return len(self.records)

    def __getitem__(self, index):

        text = self.records[index]

        encoded = self.tokenizer.encode(text)

        token_ids = encoded.ids

        # Keep sequence within context length
        token_ids = token_ids[
            :self.max_seq_len
        ]

        # Need at least two tokens
        if len(token_ids) < 2:

            token_ids = token_ids + [0]

        input_ids = token_ids[:-1]

        target_ids = token_ids[1:]

        # Padding
        padding_length = (
            self.max_seq_len - 1 - len(input_ids)
        )

        if padding_length > 0:

            input_ids += [0] * padding_length

            target_ids += [-100] * padding_length

        return {
            "input_ids": torch.tensor(
                input_ids,
                dtype=torch.long
            ),

            "targets": torch.tensor(
                target_ids,
                dtype=torch.long
            )
        }