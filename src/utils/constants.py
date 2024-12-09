"""Module for constant variables across the project."""

# Extensions
PARQUET_EXT = ".parquet"
CSV_EXT = ".csv"
TXT_EXT = ".txt"

# Data paths
RAW_DATA_DIR = "data/raw"
PROCESSED_DATA_DIR = "data/processed"
FINAL_DATA_DIR = "data/final"

SMART_DATASET_CSV = RAW_DATA_DIR + "/smart.csv"
COMBINED_DATASET_CSV = PROCESSED_DATA_DIR + "/combined.csv"
COMBINED_DATASET_PARQUET = PROCESSED_DATA_DIR + "/combined.parquet"
CLASSIFIED_DATASET_CSV = FINAL_DATA_DIR + "/classified.csv"
CLASSIFIED_DATASET_PARQUET = FINAL_DATA_DIR + "/classified.parquet"

# Data columns
TERM_COL = "term"
MNEMONIC_COL = "mnemonic"
CATEGORY_COL = "category"
SUBCATEGORY_COL = "subcategory"
CATEGORY_NAMES = ["unsure", "shallow-encoding", "deep-encoding", "mixed"]
CATEGORY_DICT = {name: i for i, name in enumerate(CATEGORY_NAMES)}

# Hugging Face datasets
HF_DATASET_REPO = "chiffonng/mnemonic-sft"  # <user>/<dataset_name>

# Model paths
CHECKPOINT_DIR = "ckpt"
