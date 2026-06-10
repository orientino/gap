import numpy as np

DIR_DATA = "./data/imagenet1k-imbalanced/i1k_imbalanced.npz"
CLASS_COUNTS = np.array([int(np.ceil(1300 / (i + 1))) for i in range(1000)], dtype=np.int64)  # fmt: skip

with np.load(DIR_DATA, mmap_mode="r") as data:
    y = data["Y"]
    counts = np.bincount(y, minlength=1000)

np.testing.assert_array_equal(np.sort(counts)[::-1], CLASS_COUNTS)
assert int(counts.sum()) == int(CLASS_COUNTS.sum())

print("Class counts:", np.sort(counts)[::-1])
