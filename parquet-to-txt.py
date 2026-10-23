
from pathlib import Path
from fastparquet import ParquetFile

INPUT_DIR = Path(
    r"C:\Users\preet\Desktop\Data science projects\Transformer"
    r"\my_dataset\plain_text"
)
OUTPUT_FILE = Path(
    r"C:\Users\preet\Desktop\Data science projects\Transformer"
    r"\corpus.txt"
)

# Set this after checking the printed column names.
TEXT_COLUMN = "text"

files = sorted(INPUT_DIR.rglob("*.parquet"))

if not files:
    raise FileNotFoundError(
        f"No Parquet files found in {INPUT_DIR}"
    )

total_rows = 0
total_chars = 0

with OUTPUT_FILE.open("w", encoding="utf-8") as out:
    for i, path in enumerate(files, start=1):
        print(f"[{i}/{len(files)}] Reading {path.name}")

        pf = ParquetFile(str(path))

        df = pf.to_pandas()

        if TEXT_COLUMN not in pf.columns:
            print("Available columns:", pf.columns)
            raise ValueError(
                f"Column '{TEXT_COLUMN}' not found in {path.name}"
            )

        # Read this file and extract the text column.
        df = pf.to_pandas(columns=[TEXT_COLUMN])

        for value in df[TEXT_COLUMN]:
            if not isinstance(value, str) or not value.strip():
                continue

            text = value.strip()
            out.write(text.strip()  + "\n<|endoftext|>\n")

            total_rows += 1
            total_chars += len(text)

        # Flush output so progress is saved during the run.
        out.flush()

        print(
            f"Rows exported: {total_rows:,} | "
            f"Characters: {total_chars:,}"
        )

print("\nExport finished.")
print("Output:", OUTPUT_FILE.resolve())
print("File size:", OUTPUT_FILE.stat().st_size, "bytes")
print("Rows exported:", total_rows)
