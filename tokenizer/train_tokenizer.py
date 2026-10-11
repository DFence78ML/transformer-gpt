from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config

tokenizer = Tokenizer(BPE(unk_token="[UNK]"))

tokenizer.pre_tokenizer = ByteLevel()
from tokenizers.decoders import ByteLevel
tokenizer.decoder = ByteLevel()
trainer = BpeTrainer(
    vocab_size=config.modelconfig["vocab_size"],
    special_tokens=[
        "[UNK]",
        "[BOS]",
        "[EOS]",
        "<|endoftext|>"
    ]
)

files = [
    "data/processed/train.txt"
]

tokenizer.train(files, trainer)

tokenizer.save("tokenizer.json")