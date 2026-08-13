from pathlib import Path
import random
import re
import json

SEED = 42

RAW_FILE = Path(
    "data/raw/books/tiny_input.txt"
)

OUTPUT_DIR = Path(
    "data/processed"
)

TRAIN_FILE = (
    OUTPUT_DIR / "train.txt"
)

VAL_FILE = (
    OUTPUT_DIR / "val.txt"
)

STATS_FILE = Path(
    "data/stats.json"
)

VAL_RATIO = 0.05
MIN_CHARS = 100
SEPARATOR = "<|endoftext|>"

def clean_text(text):

    text = text.replace(
        "\r\n",
        "\n"
    )

    text = text.replace(
        "\r",
        "\n"
    )

    text = text.replace(
        "\x00",
        ""
    )

    text = re.sub(
        r"[ \t]+",
        " ",
        text
    )

    text = re.sub(
        r"\n{3,}",
        "\n\n",
        text
    )

    return text.strip()

def read_stories():

    with open(
        RAW_FILE,
        "r",
        encoding="utf-8"
    ) as f:

        buffer = ""

        for line in f:

            buffer += line

            while SEPARATOR in buffer:

                story, buffer = (
                    buffer.split(
                        SEPARATOR,
                        1
                    )
                )

                story = clean_text(
                    story
                )

                if len(story) >= MIN_CHARS:
                    yield story

        buffer = clean_text(buffer)

        if len(buffer) >= MIN_CHARS:
            yield buffer

def prepare_data():

    random.seed(SEED)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    temp_train = (
        OUTPUT_DIR / "train_temp.txt"
    )

    temp_val = (
        OUTPUT_DIR / "val_temp.txt"
    )

    train_count = 0
    val_count = 0

    train_chars = 0
    val_chars = 0

    with open(
        temp_train,
        "w",
        encoding="utf-8"
    ) as train_file, open(
        temp_val,
        "w",
        encoding="utf-8"
    ) as val_file:

        for i, story in enumerate(
            read_stories()
        ):
            if random.random() < VAL_RATIO:

                val_file.write(
                    story
                    + "\n"
                    + SEPARATOR
                    + "\n"
                )

                val_count += 1
                val_chars += len(story)

            else:

                train_file.write(
                    story
                    + "\n"
                    + SEPARATOR
                    + "\n"
                )

                train_count += 1
                train_chars += len(story)

            if (i + 1) % 10000 == 0:

                print(
                    f"Processed "
                    f"{i + 1:,} stories"
                )

    temp_train.replace(TRAIN_FILE)
    temp_val.replace(VAL_FILE)

    stats = {

        "train_documents":
            train_count,

        "validation_documents":
            val_count,

        "train_characters":
            train_chars,

        "validation_characters":
            val_chars,

        "total_characters":
            train_chars + val_chars
    }

    STATS_FILE.write_text(
        json.dumps(
            stats,
            indent=4
        ),
        encoding="utf-8"
    )

    print()
    print("Finished!")
    print()
    print(
        f"Train stories: "
        f"{train_count:,}"
    )

    print(
        f"Validation stories: "
        f"{val_count:,}"
    )

    print(
        f"Train characters: "
        f"{train_chars:,}"
    )

    print(
        f"Validation characters: "
        f"{val_chars:,}"
    )

if __name__ == "__main__":
    prepare_data()