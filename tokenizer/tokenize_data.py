from pathlib import Path

import numpy as np
from tokenizers import Tokenizer


TOKENIZER_FILE = "tokenizer.json"

TRAIN_FILE = "data/processed/train.txt"
VAL_FILE = "data/processed/val.txt"

TRAIN_BIN = "data/processed/train.bin"
VAL_BIN = "data/processed/val.bin"


tokenizer = Tokenizer.from_file(
    TOKENIZER_FILE
)


def tokenize_file(input_file, output_file):

    print(f"Tokenizing {input_file}")

    total_tokens = 0

    with open(
        input_file,
        "r",
        encoding="utf-8"
    ) as f:

        with open(
            output_file,
            "wb"
        ) as out:

            while True:
                lines = []
                for _ in range(1000):
                    line = f.readline()
                    if not line:
                        break
                    lines.append(line)
                if not lines:
                    break

                encoded = tokenizer.encode_batch(
                    lines
                )
                ids = []

                for item in encoded:
                    ids.extend(item.ids)

                array = np.asarray(
                    ids,
                    dtype=np.uint16
                )
                array.tofile(out)
                total_tokens += len(ids)
    print(
        f"Tokens: {total_tokens:,}"
    )
    print(
        f"Saved {output_file}"
    )

tokenize_file(
    TRAIN_FILE,
    TRAIN_BIN
)

tokenize_file(
    VAL_FILE,
    VAL_BIN
)