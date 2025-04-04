# src/data_prep/data_converters.py
"""Module to format mnemonic dataset for HuggingFace and upload to the hub.

Main functions:
- create_hf_mnemonic_dataset: Create a HuggingFace dataset from mnemonic data.
- create_hf_chat_dataset: Create a HuggingFace dataset in chat format.
- create_hf_grpo_dataset: Create a HuggingFace dataset in GRPO format.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from datasets import ClassLabel, Dataset, DatasetDict, Features, Value, load_dataset
from structlog import getLogger

from src.data_gen.models import LinguisticFeature, Mnemonic
from src.data_prep.data_hf import (
    load_local_dataset,
    push_data_to_hf,
)
from src.utils import constants as const
from src.utils.common import read_prompt

if TYPE_CHECKING:
    from typing import Any, Optional

    from structlog.stdlib import BoundLogger

    from src.utils.types import PathLike

# Set up logging
logger: BoundLogger = getLogger(__name__)


def create_hf_chat_dataset(
    input_repo_id: str,
    output_repo_id: Optional[str] = None,
    system_prompt_path: Optional[PathLike] = None,
    private: bool = False,
) -> DatasetDict:
    """Convert a dataset to the chat format with role and content fields.

    Args:
        input_repo_id: HuggingFace dataset ID to convert
        output_repo_id: HuggingFace repository ID to push the converted dataset to
        system_prompt_path: Path to the system prompt file
        private: Whether to make the repository private

    Returns:
        DatasetDict containing the converted dataset
    """
    logger.info("Loading dataset", input_repo_id=input_repo_id)
    dataset = load_dataset(input_repo_id)

    # Load system prompt
    if system_prompt_path is None:
        system_prompt = read_prompt(const.PROMPT_PATH.REASON_SYSTEM)
    else:
        system_prompt = read_prompt(system_prompt_path)

    # Function to convert a row to chat format
    def convert_to_chat_format(example):
        """Convert a single example to chat format."""
        think_block = f"<think>\n\n{example['reasoning']}\n</think>"
        solution_block = f"<solution>\n\n{example['solution']}\n</solution>"
        assistant_response = f"{think_block}\n\n{solution_block}"
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": example["instruction"]},
            {"role": "assistant", "content": assistant_response},
        ]
        return {"messages": messages, "term": example["term"]}

    # Convert each split in the dataset
    chat_dataset = DatasetDict()
    for split_name, split_data in dataset.items():
        chat_dataset[split_name] = split_data.map(
            convert_to_chat_format,
            remove_columns=["instruction", "reasoning", "solution"],
        )
        logger.info(
            f"Converted {split_name} split to chat format",
            num_rows=len(chat_dataset[split_name]),
        )

    # Push to HuggingFace if output_repo_id is provided
    if output_repo_id:
        push_data_to_hf(chat_dataset, output_repo_id, private=private)
        logger.info("Pushed chat dataset to HuggingFace", repo_id=output_repo_id)

    return chat_dataset


def create_hf_grpo_dataset(
    input_repo_id: str,
    output_repo_id: Optional[str] = None,
    system_prompt_path: Optional[PathLike] = None,
    private: bool = False,
) -> DatasetDict:
    """Convert a dataset to GRPO format with prompts and completions as message lists.

    Args:
        input_repo_id: HuggingFace dataset ID to convert
        output_repo_id: HuggingFace repository ID to push the converted dataset to
        system_prompt_path: Path to the system prompt file
        private: Whether to make the repository private

    Returns:
        DatasetDict containing the converted dataset
    """
    logger.info("Loading dataset", input_repo_id=input_repo_id)
    dataset = load_dataset(input_repo_id)

    # Load system prompt
    if system_prompt_path is None:
        system_prompt = read_prompt(const.PROMPT_PATH.REASON_SYSTEM)
    else:
        system_prompt = read_prompt(system_prompt_path)

    # Function to convert a row to GRPO format
    def convert_to_grpo_format(example):
        think_block = f"<think>\n\n{example['reasoning']}\n</think>"
        solution_block = f"<solution>\n\n{example['solution']}\n</solution>"
        assistant_response = f"{think_block}\n\n{solution_block}"
        prompts = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": example["instruction"]},
        ]
        completions = [{"role": "assistant", "content": assistant_response}]
        return {
            "term": example["term"],
            "prompts": prompts,
            "completions": completions,
            "reasoning": example["reasoning"],  # Keep reasoning for reference
        }

    # Convert each split in the dataset
    grpo_dataset = DatasetDict()
    for split_name, split_data in dataset.items():
        grpo_dataset[split_name] = split_data.map(
            convert_to_grpo_format, remove_columns=["instruction", "solution"]
        )
        logger.info(
            f"Converted {split_name} split to GRPO format",
            num_rows=len(grpo_dataset[split_name]),
        )

    # Push to HuggingFace if output_repo_id is provided
    if output_repo_id:
        push_data_to_hf(grpo_dataset, output_repo_id, private=private)
        logger.info("Pushed GRPO dataset to HuggingFace", repo_id=output_repo_id)

    return grpo_dataset


# TODO: remove the function after refactoring
def create_class_dataset(
    input_path: PathLike,
    select_col_names: Optional[list[str]] = None,
    val_ratio: float = 0.2,
    seed: int = 42,
) -> DatasetDict:
    """Create a HuggingFace dataset where there are clearer class labels for the mnemonics.

    Args:
        input_path (PathLike): Path to the input data (filepath or SQLite)
        select_col_names (Optional[list[str]]): Column names to select from the dataset
        val_ratio (float): Proportion of data to use for validation
        seed (int): Random seed for reproducibility

    Returns:
        DatasetDict containing train and validation splits
    """
    # Create features specification
    mnemonic_type_labels: list[str] = LinguisticFeature.get_types()
    mnemonic_type_labels.append(None)  # Add None for missing values
    num_mnemonic_types = len(mnemonic_type_labels)

    # Column names to select
    select_col_names = select_col_names or list(Mnemonic.model_fields.keys())

    dataset: Dataset = load_local_dataset(input_path)

    dataset = dataset.select_columns(select_col_names)
    logger.debug("Loaded dataset columns", columns=dataset.column_names)

    features = Features(
        {
            "term": Value("string"),
            "mnemonic": Value("string"),
            "reasoning": Value("string"),
            "linguistic_feature": ClassLabel(
                names=mnemonic_type_labels, num_classes=num_mnemonic_types
            ),
        }
    )
    dataset = dataset.cast(features)

    # Split into train and validation
    splits: DatasetDict = dataset.train_test_split(
        test_size=val_ratio, stratify_by_column="linguistic_feature", seed=seed
    )

    return DatasetDict({"train": splits["train"], "val": splits["test"]})


if __name__ == "__main__":
    chat_dataset = create_hf_chat_dataset(
        input_repo_id=const.HF_CONST.DATASET_NAME,
        output_repo_id=const.HF_CONST.CHAT_DATASET_NAME,
    )

    grpo_dataset = create_hf_grpo_dataset(
        input_repo_id=const.HF_CONST.DATASET_NAME,
        output_repo_id=const.HF_CONST.RL_DATASET_NAME,
    )
