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


def tokenize_file(
    input_file,
    output_file
):

    print(
        f"Tokenizing {input_file}"
    )

    text = Path(
        input_file
    ).read_text(
        encoding="utf-8"
    )

    ids = tokenizer.encode(
        text
    ).ids

    print(
        f"Tokens: {len(ids):,}"
    )

    array = np.array(
        ids,
        dtype=np.uint16
    )

    array.tofile(
        output_file
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