from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import BpeTrainer

tokenizer = Tokenizer(BPE(unk_token="[UNK]"))

tokenizer.pre_tokenizer = Whitespace()

trainer = BpeTrainer(
    vocab_size=3000,
    special_tokens=[
        "[PAD]",
        "[UNK]",
        "[BOS]",
        "[EOS]"
    ]
)

files = [
    "data/input.txt"
]

tokenizer.train(files, trainer)

tokenizer.save("tokenizer.json")