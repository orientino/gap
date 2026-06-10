import os

import numpy as np
import tiktoken
from datasets import load_dataset
from tqdm import tqdm

DIR_DATA = "./data"
N_TOKENS = 1_000_000_000  # number of tokens to prepare
CHUNK_SIZE = 10_000_000

# download and tokenize FineWeb-Edu
enc = tiktoken.get_encoding("gpt2")
ds = load_dataset(
    "HuggingFaceFW/fineweb-edu",
    name="sample-10BT",
    split="train",
    streaming=True,
)

# encode N_TOKENS with tiktoken gpt2 bpe, writing chunks to disk to avoid OOM
dir_out = os.path.join(DIR_DATA, "fineweb")
os.makedirs(dir_out, exist_ok=True)
tmp_path = os.path.join(dir_out, "all_tokens.bin")

total = 0
buf = []
with open(tmp_path, "wb") as f:
    for doc in tqdm(ds):
        buf.extend(enc.encode_ordinary(doc["text"]))
        if len(buf) >= CHUNK_SIZE:
            f.write(np.array(buf, dtype=np.uint16).tobytes())
            total += len(buf)
            buf = []
            if total >= N_TOKENS:
                break
    if buf and total < N_TOKENS:
        f.write(np.array(buf, dtype=np.uint16).tobytes())
        total += len(buf)

# memory-map and split into train/val
tokens = np.memmap(tmp_path, dtype=np.uint16, mode="r")[:N_TOKENS]
n = len(tokens)
split = int(n * 0.99)
print(f"train has {split:,} tokens")
print(f"val has {n - split:,} tokens")

np.array(tokens[:split]).tofile(os.path.join(dir_out, "train.bin"))
np.array(tokens[split:]).tofile(os.path.join(dir_out, "val.bin"))
os.remove(tmp_path)
