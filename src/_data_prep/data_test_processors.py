"""Manage test data: Create, augment, deduplicate, and prevent data leakage."""

import random
from pathlib import Path

from src.data.data_loaders import load_hf_dataset
from src.utils import check_file_path
from src.utils import constants as const

sample_size: int | None = 200

# Construct the path to raw/test.txt and final/test.txt
raw_test_data_path = check_file_path(const.RAW_TEST_DATASET_TXT, new_ok=True)
final_test_data_path = check_file_path(
    const.FINAL_TEST_DATASET_TXT, new_ok=True, to_create=True
)

# Read the unique terms from test.txt (each term on a new line)
with Path.open(raw_test_data_path, "r") as f:
    my_terms = set(line.strip() for line in f if line.strip())

# Augment with dataset from nbalepur/Mnemonic_Test (only one "train" split)
mnemonic_dataset = load_hf_dataset("nbalepur/Mnemonic_Test", split="train")
mnemonic_terms = set(example["term"] for example in mnemonic_dataset)
union_terms = my_terms.union(mnemonic_terms)
print(f"Total union terms: {len(union_terms)}")

# Prevent data leakage by removing terms that are in the training+validation set
train_val_data = load_hf_dataset("chiffonng/en-vocab-mnemonics", split="train+test")
train_val_terms = set(example["term"] for example in train_val_data)
print(f"Total training+validation terms: {len(train_val_terms)}")
final_test_terms = union_terms - train_val_terms
print(f"Final test terms (after cleaning): {len(final_test_terms)}")

# Sample terms from the final test terms
if sample_size is not None and len(final_test_terms) >= sample_size:
    sampled_test_terms = random.sample(list(final_test_terms), k=sample_size)
else:
    sampled_test_terms = list(final_test_terms)

# Write the sampled terms to final/test.txt
with final_test_data_path.open("w") as f:
    for term in sorted(sampled_test_terms):  # Sort the terms
        f.write(term + "\n")
