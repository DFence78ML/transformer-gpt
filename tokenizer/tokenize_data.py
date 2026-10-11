from pathlib import Path

import numpy as np
from tokenizers import Tokenizer
import time

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
    total_words = 0
    encoding_seconds = 0
    lines_processed = 0

    with open(
        input_file,
        "r",
        encoding="utf-8"
    ) as f:

        with open(
            output_file,
            "wb"
        ) as out:
            start_time = time.time()
            while True:
                lines = []
                for i in range(1000):
                    line = f.readline()
                    if not line:
                        break
                    words = line.split()
                    total_words += len(words)
                    lines.append(line)
                lines_processed += i+1
                if not lines:
                    break
                start = time.perf_counter()
                encoded = tokenizer.encode_batch(
                    lines
                )
                end = time.perf_counter()
                encoding_seconds += end - start
                ids = []


                for item in encoded:
                    ids.extend(item.ids)

                array = np.asarray(
                    ids,
                    dtype=np.uint16
                )
                array.tofile(out)
                total_tokens += len(ids)

                if lines_processed % 100000 == 0:
                    print(
                        f"Processed: {lines_processed}"
                    )
                
    end_time = time.time()
    tokens_per_sec = total_tokens / (end_time - start_time)
    tokens_per_word = total_tokens / total_words
    encoding_throughput = total_tokens / encoding_seconds
    print(f"Tokens per word: {tokens_per_word:.4f}")
    print(f"Tokens per sec: {tokens_per_sec:.4f}")
    print(f"Encoding throughput: {encoding_throughput:.4f}")

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

print(max(tokenizer.get_vocab().values()))
print(min(tokenizer.get_vocab().values()))
sentence = "I love cats"
ids = tokenizer.encode(sentence)
words = tokenizer.decode(ids.ids)
print(words)