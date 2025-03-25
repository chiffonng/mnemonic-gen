"""Module for constant variables across this data_prep module."""

# File extensions
JSON_EXT = ".json"
JSONL_EXT = ".jsonl"
PARQUET_EXT = ".parquet"
CSV_EXT = ".csv"
TXT_EXT = ".txt"
PROMPT_EXT = ".txt"

# Data paths
DATA_MODULE = "data"
RAW_DATA_DIR = DATA_MODULE + "/raw"
PROCESSED_DATA_DIR = DATA_MODULE + "/processed"
FINAL_DATA_DIR = DATA_MODULE + "/final"

COMBINED_DATASET_CSV = PROCESSED_DATA_DIR + "/combined.csv"
COMBINED_DATASET_PARQUET = PROCESSED_DATA_DIR + "/combined.parquet"
RAW_TEST_DATASET_TXT = RAW_DATA_DIR + "/test.txt"
FINAL_TEST_DATASET_TXT = FINAL_DATA_DIR + "/test.txt"

SEED_IMPROVED_CSV = PROCESSED_DATA_DIR + "/seed_improved.csv"
SEED_IMPROVED_JSON = PROCESSED_DATA_DIR + "/seed_improved_stratified.jsonl"
IMPROVED_CSV = PROCESSED_DATA_DIR + "/improved.csv"
MNEMONIC_DB_URI = "sqlite:///" + PROCESSED_DATA_DIR + "/mnemonics.db"

# Data columns
TERM_COL = "term"
MNEMONIC_COL = "mnemonic"

# Hugging Face constants
HF_DATASET_NAME = "chiffonng/en-vocab-mnemonics"  # <user>/<dataset_name>
HF_TESTSET_NAME = "chiffonng/en-vocab-mnemonics-test"  # <user>/<dataset_name>
HF_MODEL_NAME = "chiffonng/gemma2-9b-it-mnemonics"  # <user>/<model_name>
HF_MNEMONIC_DATASET = "chiffonng/en-vocab-mnemonics-full"

# OpenAI Finetuning API
SFT_IMPROVE_TRAIN = PROCESSED_DATA_DIR + "/sft_improve_train.jsonl"
SFT_IMPROVE_VAL = PROCESSED_DATA_DIR + "/sft_improve_val.jsonl"
SFT_OPENAI_MODEL_ID = "ft:gpt-4o-mini-2024-07-18:personal:improve-sft:B62kPWoy"

# Prompts
DIR_PROMPT = "prompts"
DIR_PROMPT_IMPROVE = DIR_PROMPT + "/improve"
DIR_PROMPT_CLASSIFY = DIR_PROMPT + "/classify"
DIR_PROMPT_FINETUNE = DIR_PROMPT + "/finetune"
DIR_PROMPT_GENERATE = DIR_PROMPT + "/generate"

FILE_PROMPT_PLACEHOLDER_DICT = DIR_PROMPT + "/placeholders.json"

FILE_PROMPT_IMPROVE_SYSTEM = DIR_PROMPT_IMPROVE + "/improve_gen_system.txt"
FILE_PROMPT_IMPROVE_SFT_SYSTEM = DIR_PROMPT_IMPROVE + "/improve_sft_system.txt"
FILE_PROMPT_CLASSIFY_SYSTEM = DIR_PROMPT_CLASSIFY + "/classify_system.txt"
FILE_PROMPT_USER = DIR_PROMPT + "/user.txt"

FILE_PROMPT_FINETUNE_SYSTEM = DIR_PROMPT_FINETUNE + "/system.txt"
FILE_PROMPT_FINETUNE_USER = DIR_PROMPT_FINETUNE + "/user.txt"

# Config files
DIR_CONFIG = "config"
CONF_DEFAULT = DIR_CONFIG + "/default_conf.json"
CONF_HUGGINGFACE = DIR_CONFIG + "/huggingface_conf.json"
CONF_CLAUDE = DIR_CONFIG + "/claude_conf.json"
CONF_OPENAI = DIR_CONFIG + "/openai_conf.json"
CONF_OPENAI_SFT = DIR_CONFIG + "/openai_sft_conf.json"
CONF_OPENAI_SFT_COMPLETION = DIR_CONFIG + "/openai_sftcc_conf.json"
