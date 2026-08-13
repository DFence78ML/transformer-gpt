from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer

tokenizer = Tokenizer(BPE(unk_token="[UNK]"))

tokenizer.pre_tokenizer = ByteLevel()

trainer = BpeTrainer(
    vocab_size=8000,
    special_tokens=[
        "[PAD]",
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