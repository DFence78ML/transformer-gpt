
#MODEL
vocab_size = 32000
d_model = 512
layers = 6
heads = 8
d_ff = 2048
dropout = 0.1
seq_len = 128

#TRAINING
batch_size = 128
grad_accum = 8
epochs = 3
max_lr = 3e-4
min_lr = 3e-5
warmup_perc = 0.01
weight_decay = 0.1
grad_clip = 1.0

#DATA
train_file = "/kaggle/input/datasets/preetsidhu20/tokenized-files/train.bin"
val_file = "/kaggle/input/datasets/preetsidhu20/tokenized-files/val.bin"
tokenizer = "tokenizer.json"

#HARDWARE
num_workers = 8