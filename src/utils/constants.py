"""Module for constant variables across this data_prep module."""

# Extensions
PARQUET_EXT = ".parquet"
CSV_EXT = ".csv"
TXT_EXT = ".txt"

# Data paths
DATA_MODULE = "data_prep"
RAW_DATA_DIR = DATA_MODULE + "/raw"
PROCESSED_DATA_DIR = DATA_MODULE + "/processed"
FINAL_DATA_DIR = DATA_MODULE + "/final"

COMBINED_DATASET_CSV = PROCESSED_DATA_DIR + "/combined.csv"
COMBINED_DATASET_PARQUET = PROCESSED_DATA_DIR + "/combined.parquet"

# Data columns
TERM_COL = "term"
MNEMONIC_COL = "mnemonic"

HF_DATASET_NAME = "chiffonng/en-vocab-mnemonics"  # <user>/<dataset_name>
HF_MODEL_NAME = "chiffonng/gemma2-9b-it-mnemonics"  # <user>/<model_name>
