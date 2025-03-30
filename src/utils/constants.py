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
HF_MODEL_NAME = "chiffonng/gemma3-4b-it-mnemonics"  # <user>/<model_name>
HF_CHAT_DATASET = "chiffonng/en-vocab-mnemonics-chat"

# OpenAI Finetuning API
SFT_IMPROVE_TRAIN = PROCESSED_DATA_DIR + "/sft_improve_train.jsonl"
SFT_IMPROVE_VAL = PROCESSED_DATA_DIR + "/sft_improve_val.jsonl"
SFT_OPENAI_MODEL_ID = "ft:gpt-4o-mini-2024-07-18:personal:improve-sft:B62kPWoy"

# Prompts
DIR_PROMPT = "prompts"
DIR_PROMPT_REASON = DIR_PROMPT + "/reason"
DIR_PROMPT_CLASSIFY = DIR_PROMPT + "/classify"
DIR_PROMPT_FINETUNE = DIR_PROMPT + "/finetune"
DIR_PROMPT_GENERATE = DIR_PROMPT + "/generate"

SYSTEM_PROMPT_NAME = "/system.txt"
USER_PROMPT_NAME = "/user.txt"

FILE_PROMPT_PLACEHOLDER_DICT = DIR_PROMPT + "/placeholders.json"
FILE_PROMPT_USER = DIR_PROMPT + "/user_basic.txt"

FILE_PROMPT_REASON_SYSTEM = DIR_PROMPT_GENERATE + SYSTEM_PROMPT_NAME
FILE_PROMPT_REASON_USER = DIR_PROMPT_GENERATE + USER_PROMPT_NAME

FILE_PROMPT_FINETUNE_SYSTEM = DIR_PROMPT_FINETUNE + SYSTEM_PROMPT_NAME
FILE_PROMPT_FINETUNE_USER = DIR_PROMPT_FINETUNE + USER_PROMPT_NAME

FILE_PROMPT_CLASSIFY_SYSTEM = DIR_PROMPT_CLASSIFY + SYSTEM_PROMPT_NAME

# Config files
DIR_CONFIG = "config"
DIR_CONFIG_API = DIR_CONFIG + "/api"
CONF_DEFAULT_GEN = DIR_CONFIG_API + "/default_generation.json"
CONF_DEFAULT_BACKEND = DIR_CONFIG_API + "/default_backend.json"

CONF_HUGGINGFACE = DIR_CONFIG_API + "/hf.json"
CONF_CLAUDE = DIR_CONFIG_API + "/claude.json"
CONF_OPENAI = DIR_CONFIG_API + "/openai.json"
CONF_OPENAI_SFT_API = DIR_CONFIG_API + "/openai_sft.json"
CONF_DEEPSEEK_REASONER = DIR_CONFIG_API + "/deepseek_reasoner.json"

DIR_CONFIG_FINETUNE = DIR_CONFIG + "/finetune"
CONF_OPENAI_SFT = DIR_CONFIG_FINETUNE + "/openai_sft.json"
