"""Module for structured project constants with automatic path generation."""

from __future__ import annotations

import os
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Literal, Optional


# File extensions as an Enum for type safety
class Extensions(str, Enum):
    """File extensions used throughout the project."""

    JSON = ".json"
    JSONL = ".jsonl"
    PARQUET = ".parquet"
    CSV = ".csv"
    TXT = ".txt"
    PROMPT = ".txt"

    @classmethod
    def get(cls, ext: str) -> str:
        """Get extension by name, with fallback to raw string."""
        try:
            return cls[ext.upper()].value
        except (KeyError, AttributeError):
            return f".{ext.lower().lstrip('.')}"


# Base paths as a dataclass with automatic Path generation
@dataclass(frozen=True)
class BasePaths:
    """Base directory paths for the project."""

    ROOT: Path = Path(os.getenv("PROJECT_ROOT", "."))
    DATA: Path = ROOT / "data"
    DATA_RAW: Path = DATA / "raw"
    DATA_PROCESSED: Path = DATA / "processed"
    DATA_FINAL: Path = DATA / "final"
    PROMPTS: Path = ROOT / "prompts"
    CONFIG: Path = ROOT / "config"

    def __post_init__(self):
        """Ensure all directories exist."""
        for field_name in self.__dataclass_fields__:
            if field_name != "ROOT":
                path = getattr(self, field_name)
                path.mkdir(parents=True, exist_ok=True)


# Path generator for flexible path creation
class PathMaker:
    """Utility for generating paths with consistent structure."""

    def __init__(self, base_paths: BasePaths):
        """Initialize the PathMaker with base paths."""
        self.base = base_paths

    def data_file(
        self, filename: str, data_type: str = "processed", ext: Optional[str] = None
    ) -> Path:
        """Generate a path for a data file.

        Args:
            filename: Base name of the file without extension
            data_type: Type of data (raw, processed, final)
            ext: File extension (defaults to inferring from filename)

        Returns:
            Full path to the file
        """
        if ext is None:
            # Try to extract extension from filename
            parts = filename.split(".")
            if len(parts) > 1:
                ext = f".{parts[-1]}"
                filename = ".".join(parts[:-1])
            else:
                ext = Extensions.CSV  # Default to CSV
        else:
            ext = Extensions.get(ext)

        # Determine the base directory based on the data type
        base_dir = getattr(self.base, data_type.upper(), self.base.DATA_PROCESSED)
        return base_dir / f"{filename}{ext}"

    def prompt_file(
        self, category: Optional[str], name: str = "system", ext: Literal["txt"] = "txt"
    ) -> Path:
        """Generate a path for a prompt file."""
        ext = Extensions.get(ext)
        return self.base.PROMPTS / category / f"{name}{ext}"

    def config_file(
        self, category: Optional[str], name: str, ext: Literal["json", "yaml"] = "json"
    ) -> Path:
        """Generate a path for a config file."""
        ext = Extensions.get(ext)
        return self.base.CONFIG / category / f"{name}{ext}"


# Column names
class Columns:
    """Column names used in datasets."""

    TERM = "term"
    MNEMONIC = "mnemonic"
    IMPROVED_MNEMONIC = "improved_mnemonic"
    LINGUISTIC_REASONING = "linguistic_reasoning"
    MAIN_TYPE = "main_type"
    SUB_TYPE = "sub_type"


# Hugging Face constants
class HFConstants:
    """Hugging Face related constants."""

    USER = "chiffonng"
    DATASET_NAME = f"{USER}/en-vocab-mnemonics"
    TESTSET_NAME = f"{USER}/en-vocab-mnemonics-test"
    MODEL_NAME = f"{USER}/gemma3-4b-it-mnemonics"
    CHAT_DATASET = f"{USER}/en-vocab-mnemonics-chat"


# Initialize base paths
PATHS = BasePaths()
PATH = PathMaker(PATHS)

# Create common file path constants
DATA_FILES = {
    # Data files
    "COMBINED_CSV": PATH.data_file("combined", ext="csv"),
    "RAW_TEST": PATH.data_file("test", data_type="raw", ext="txt"),
    "FINAL_TEST": PATH.data_file("test", data_type="final", ext="txt"),
    "SEED_IMPROVED_CSV": PATH.data_file("seed_improved", ext="csv"),
    "MNEMONIC_DB_URI": f"sqlite:///{PATHS.DATA_PROCESSED}/mnemonics.db",
    # OpenAI finetuning files
    "SFT_IMPROVE_TRAIN": PATH.data_file("sft_improve_train", ext="jsonl"),
    "SFT_IMPROVE_VAL": PATH.data_file("sft_improve_val", ext="jsonl"),
}

# Prompt file paths
PROMPT_FILES = {
    "PLACEHOLDER_DICT": PATH.prompt_file("", "placeholders", "json"),
    "USER_BASIC": PATH.prompt_file("", "user_basic"),
    "REASON_SYSTEM": PATH.prompt_file("generate", "system"),
    "REASON_USER": PATH.prompt_file("generate", "user"),
    "FINETUNE_SYSTEM": PATH.prompt_file("finetune", "system"),
    "FINETUNE_USER": PATH.prompt_file("finetune", "user"),
    "CLASSIFY_SYSTEM": PATH.prompt_file("classify", "system"),
}

# Config file paths
CONFIG_FILES = {
    "DEFAULT_GEN": PATH.config_file("api", "default_generation"),
    "DEFAULT_BACKEND": PATH.config_file("api", "default_backend"),
    "HUGGINGFACE": PATH.config_file("api", "hf"),
    "CLAUDE": PATH.config_file("api", "claude"),
    "OPENAI": PATH.config_file("api", "openai"),
    "OPENAI_SFT_API": PATH.config_file("api", "openai_sft"),
    "DEEPSEEK_REASONER": PATH.config_file("api", "deepseek_reasoner"),
    "OPENAI_SFT": PATH.config_file("finetune", "openai_sft"),
    "GRPO": PATH.config_file("finetune", "grpo", "yaml"),
}

# Other constants
SFT_OPENAI_MODEL_ID = "ft:gpt-4o-mini-2024-07-18:personal:improve-sft:B62kPWoy"

# Export all the constants for easy access
# These allow direct imports like `from src.utils.constants import PATHS, COLUMNS`
COLUMNS = Columns
HFCONST = HFConstants
EXT = Extensions

# For backward compatibility, expose all constants at module level
locals().update(DATA_FILES)
locals().update(PROMPT_FILES)
locals().update(CONFIG_FILES)
