import gzip
import os
import urllib.request

import numpy as np
from tqdm import tqdm

DIR_DATA = "./data"
CHUNK_SIZE = 10_000_000

URL = "https://storage.googleapis.com/basenji_barnyard2/hg38.ml.fa.gz"
gz_path = os.path.join(os.path.dirname(__file__), "hg38.ml.fa.gz")

if not os.path.exists(gz_path):
    print(f"Downloading {URL} ...")
    urllib.request.urlretrieve(URL, gz_path)

dir_out = os.path.join(DIR_DATA, "hg38_char")
os.makedirs(dir_out, exist_ok=True)
tmp_path = os.path.join(dir_out, "all_tokens.bin")

char_to_int = {"A": 0, "C": 1, "G": 2, "T": 3, "N": 4}

print("Parsing FASTA with character encoding...")
total = 0
buf = []
with gzip.open(gz_path, "rt") as f, open(tmp_path, "wb") as out:
    for line in tqdm(f):
        if line.startswith(">"):
            continue
        buf.extend(char_to_int[c] for c in line.strip().upper() if c in char_to_int)
        if len(buf) >= CHUNK_SIZE:  # write in chunks to prevent OOM
            out.write(np.array(buf, dtype=np.uint8).tobytes())
            total += len(buf)
            buf = []
    out.write(np.array(buf, dtype=np.uint8).tobytes())
    total += len(buf)

tokens = np.memmap(tmp_path, dtype=np.uint8, mode="r")
n = len(tokens)
split = int(n * 0.9)
print(f"train has {split:,} tokens")
print(f"val has {n - split:,} tokens")

np.array(tokens[:split]).tofile(os.path.join(dir_out, "train.bin"))
np.array(tokens[split:]).tofile(os.path.join(dir_out, "val.bin"))
os.remove(tmp_path)

# train has 2,727,938,175 tokens
# val has 303,104,025 tokens
