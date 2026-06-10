import gzip
import os
import urllib.request

import numpy as np
from tqdm import tqdm

DIR_DATA = "./data"
CHUNK_SIZE = 10_000_000
K = 6
STRIDE = 1  # overlapping k-mers
# STRIDE = K  # non-overlapping k-mers

URL = "https://storage.googleapis.com/basenji_barnyard2/hg38.ml.fa.gz"
gz_path = os.path.join(os.path.dirname(__file__), "hg38.ml.fa.gz")

if not os.path.exists(gz_path):
    print(f"Downloading {URL} ...")
    urllib.request.urlretrieve(URL, gz_path)

dir_out = os.path.join(DIR_DATA, f"hg38_kmer{K}_s{STRIDE}")
os.makedirs(dir_out, exist_ok=True)
tmp_path = os.path.join(dir_out, "all_tokens.bin")

# This code encodes one chromosome at time using the k-mer encoding.
# When using a K=6 the vocabulary size is equal to 4^6=4096.
# Special token for k-mers containing `N` nucleotides, so the final vocab size is 4097.
# Chunking is used to prevent OOM.

char_to_int = {"A": 0, "C": 1, "G": 2, "T": 3}
N_TOKEN = 4**K  # special token for k-mers containing `N` nucleotides
vocab_size = 4**K + 1


def encode(chromosome, buf, out):
    seq = "".join(chromosome)
    for i in range(0, len(seq) - K + 1, STRIDE):
        kmer = seq[i : i + K]
        if any(c not in char_to_int for c in kmer):
            buf.append(N_TOKEN)  # append special token for k-mers containing `N`
        else:
            val = 0
            for c in kmer:
                val = val * 4 + char_to_int[c]  # encode the k-mer as a base-4 integer
            buf.append(val)  # append the encoded k-mer as an integer in [0, 4^K-1]
        if len(buf) >= CHUNK_SIZE:
            out.write(np.array(buf, dtype=np.uint16).tobytes())
            buf = []
    return buf


print(f"Parsing FASTA with k-mers encoding (K={K}, STRIDE={STRIDE})...")
buf = []
chromosome = []
with gzip.open(gz_path, "rt") as f, open(tmp_path, "wb") as out:
    for line in tqdm(f):
        if line.startswith(">"):  # new chromosome starts, encode the previous one
            buf = encode(chromosome, buf, out)
            chromosome = []
            continue
        chromosome.append(line.strip().upper())
    buf = encode(chromosome, buf, out)  # encode and write the last chromosome
    out.write(np.array(buf, dtype=np.uint16).tobytes())

tokens = np.memmap(tmp_path, dtype=np.uint16, mode="r")
n = len(tokens)
split = int(n * 0.9)
print(f"train has {split:,} tokens")
print(f"val has {n - split:,} tokens")
print(f"vocab_size: {vocab_size}")

np.array(tokens[:split]).tofile(os.path.join(dir_out, "train.bin"))
np.array(tokens[split:]).tofile(os.path.join(dir_out, "val.bin"))
os.remove(tmp_path)

# K=6, STRIDE=1 (overlapping k-mers)
# train has 2,727,938,274 tokens
# val has 303,104,253 tokens
# vocab_size: 4097

# K=6, STRIDE=6 (non-overlapping k-mers)
# train has 454,656,353 tokens
# val has 50,517,373 tokens
# vocab_size: 4097
