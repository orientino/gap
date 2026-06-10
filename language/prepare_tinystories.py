import os

import numpy as np
import tiktoken
from datasets import load_dataset
from tqdm import tqdm

DIR_DATA = "./data"
N_TOKENS = 10_000_000  # number of tokens to prepare

# download and tokenize TinyStories
enc = tiktoken.get_encoding("gpt2")
ds = load_dataset(
    "roneneldan/TinyStories",
    split="train",
    streaming=True,
)

# encode N_TOKENS with tiktoken gpt2 bpe
tokens = []
for doc in tqdm(ds):
    tokens.extend(enc.encode_ordinary(doc["text"]))
    if len(tokens) >= N_TOKENS:
        break
tokens = np.array(tokens, dtype=np.uint16)
n = len(tokens)
train_ids = tokens[: int(n * 0.99)]
val_ids = tokens[int(n * 0.99) :]
print(f"train has {len(train_ids):,} tokens")
print(f"val has {len(val_ids):,} tokens")

# export to bin files
dir_out = os.path.join(DIR_DATA, "tinystories")
os.makedirs(dir_out, exist_ok=True)
train_ids.tofile(os.path.join(dir_out, "train.bin"))
val_ids.tofile(os.path.join(dir_out, "val.bin"))
