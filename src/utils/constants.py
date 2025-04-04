"""Module for structured project constants with automatic path generation."""

from __future__ import annotations

import os
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Literal, Optional


# File extensions as an Enum for type safety
class Extension(str, Enum):
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
class BasePath:
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


class PathMaker:
    """Utility for generating paths with consistent structure."""

    def __init__(self, base_paths: BasePath):
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
                ext = Extension.CSV  # Default to CSV
        else:
            ext = Extension.get(ext)

        # Determine the base directory based on the data type
        base_dir = getattr(self.base, data_type.upper(), self.base.DATA_PROCESSED)
        return base_dir / f"{filename}{ext}"

    def prompt_file(
        self,
        category: Optional[Literal["reason", "judge", "finetune", "prompt"]] = None,
        name: str = "system",
        ext: str = "txt",
    ) -> Path:
        """Generate a path for a prompt file."""
        ext = Extension.get(ext)

        # Handle special case for placeholder prompts
        if not category:
            prompt_path = self.base.PROMPTS / f"{name}{ext}"
        else:
            prompt_path = self.base.PROMPTS / category / f"{name}{ext}"

        # Create the directory and file if it doesn't exist
        prompt_path.parent.mkdir(parents=True, exist_ok=True)
        if not prompt_path.exists():
            prompt_path.touch()
        return prompt_path

    def config_file(
        self, category: Literal["finetune", "api", ""], name: str, ext: str = "json"
    ) -> Path:
        """Generate a path for a config file."""
        ext = Extension.get(ext)

        if not category:
            config_path = self.base.CONFIG / f"{name}{ext}"
        else:
            config_path = self.base.CONFIG / category / f"{name}{ext}"

        config_path.parent.mkdir(parents=True, exist_ok=True)
        if not config_path.exists():
            config_path.touch()
        return config_path


# Initialize base paths
BASE_PATHS = BasePath()
PATH = PathMaker(BASE_PATHS)


@dataclass
class DataPath:
    """Data paths for different stages of processing."""

    RAW_TEST = PATH.data_file("test", data_type="raw", ext="txt")
    FINAL_TEST = PATH.data_file("test", data_type="final", ext="txt")
    EXAMPLES = PATH.data_file("examples", data_type="processed", ext="csv")
    EXAMPLES_JSONL = PATH.data_file("examples", data_type="processed", ext="jsonl")

    OPENAI_SFT_IMPROVE_TRAIN = PATH.data_file("sft_improve_train", ext="jsonl")
    OPENAI_SFT_IMPROVE_VAL = PATH.data_file("sft_improve_val", ext="jsonl")


@dataclass
class PromptPath:
    """Prompt paths for different stages of processing."""

    # Synthetic data generation. Default is CoT
    REASON_SYSTEM: Path = PATH.prompt_file("reason", "system")
    REASON_USER: Path = PATH.prompt_file("reason", "user")

    JUDGE_SYSTEM: Path = PATH.prompt_file("judge", "system")

    # Fine-tuning
    FINETUNE_SYSTEM: Path = PATH.prompt_file("finetune", "system")
    FINETUNE_USER: Path = PATH.prompt_file("finetune", "user")

    # Placeholder variables in prompts
    PLACEHOLDER_DICT: Path = PATH.prompt_file("", "placeholders", ext="json")


@dataclass
class ConfigPath:
    """Config file paths for various configurations."""

    # API related config files
    DEFAULT_GENERATION: Path = PATH.config_file("api", "default_generation")
    DEFAULT_BACKEND: Path = PATH.config_file("api", "default_backend")
    DEFAULT_BACKEND_BATCH: Path = PATH.config_file("api", "default_backend_batch")
    HUGGINGFACE: Path = PATH.config_file("api", "hf")
    OPENAI: Path = PATH.config_file("api", "openai")
    OPENAI_SFT_API: Path = PATH.config_file("api", "openai_sft")
    DEEPSEEK_REASONER: Path = PATH.config_file("api", "deepseek_reasoner")

    # Fine-tuning related config files
    OPENAI_SFT: Path = PATH.config_file("finetune", "openai_sft")
    GRPO: Path = PATH.config_file("finetune", "grpo", "yaml")
    PEFT: Path = PATH.config_file("finetune", "peft", "yaml")


class Column:
    """Column names used in datasets."""

    TERM = "term"
    MNEMONIC = "mnemonic"
    REASONING = "reasoning"
    MAIN_TYPE = "main_type"
    SUB_TYPE = "sub_type"


# Hugging Face constants
class HfConst:
    """Hugging Face related constants."""

    USER: str = "chiffonng"
    MODEL_NAME: str = f"{USER}/gemma-3-4b-it-mnemonics"

    DATASET_NAME: str = f"{USER}/en-vocab-en-mnemonics-cot"
    TESTSET_NAME: str = f"{USER}/en-vocab-mnemonics-test"
    CHAT_DATASET_NAME: str = f"{USER}/en-vocab-mnemonics-chat"
    RL_DATASET_NAME: str = f"{USER}/en-vocab-mnemonics-rl"


HF_CONST = HfConst()
DATA_PATH = DataPath()
CONFIG_PATH = ConfigPath()
PROMPT_PATH = PromptPath()
