# src/data/format_and_upload_dataset.py
"""Module to format mnemonic dataset for HuggingFace and upload to the hub.

Main functions:
- create_hf_mnemonic_dataset: Create a HuggingFace dataset from mnemonic data.
- create_hf_chat_dataset: Create a HuggingFace dataset in chat format.
"""

from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING

from datasets import ClassLabel, Dataset, DatasetDict, Features, Value
from src.data_prep.data_hf import (
    load_from_database,
    load_local_dataset,
    push_data_to_hf,
)
from src.reason.mnemonic_models import MnemonicType
from src.utils import constants as const
from src.utils.common import read_prompt
from structlog import getLogger

if TYPE_CHECKING:
    from typing import Any, Optional

    from src.utils.types import PathLike
    from structlog.stdlib import BoundLogger

# Set up logging
logger: BoundLogger = getLogger(__name__)


def create_hf_mnemonic_dataset(
    input_path: PathLike,
    select_col_names: str | list[str] = None,
    val_ratio: float = 0.2,
    seed: int = 42,
) -> DatasetDict:
    """Create a HuggingFace dataset from mnemonic data.

    Args:
        input_path (PathLike): Path to the input data (filepath or SQLite)
        select_col_names (str | list[str]): Column names to select from the dataset
        val_ratio (float): Proportion of data to use for validation
        seed (int): Random seed for reproducibility

    Returns:
        DatasetDict containing train and validation splits
    """
    # Create features specification
    mnemonic_type_labels: list[str] = MnemonicType.get_types()
    mnemonic_type_labels.append(None)  # Add None for missing values
    num_mnemonic_types = len(mnemonic_type_labels)

    # Column names to select
    select_col_names = select_col_names or [
        "term",
        "mnemonic",
        "linguistic_reasoning",
        "main_type",
        "sub_type",
    ]

    # Read data
    if input_path.endswith(("db", "sqlite", "sqlite3")):
        dataset: Dataset = load_from_database("mnemonic", con=f"sqlite:///{input_path}")

    else:
        dataset: Dataset = load_local_dataset(input_path)
        # Change "mnemonic" column to the content of the "improved_mnemonic" column if it exists
        if "improved_mnemonic" in dataset.column_names:
            dataset = dataset.map(
                lambda x: {"mnemonic": x["improved_mnemonic"]},
                remove_columns=["improved_mnemonic"],
            )
            logger.debug("Replaced 'mnemonic' with 'improved_mnemonic'")

    dataset = dataset.select_columns(select_col_names)
    logger.debug("Loaded dataset columns", columns=dataset.column_names)

    features = Features(
        {
            "term": Value("string"),
            "mnemonic": Value("string"),
            "linguistic_reasoning": Value("string"),
            "main_type": ClassLabel(
                names=mnemonic_type_labels, num_classes=num_mnemonic_types
            ),
            "sub_type": ClassLabel(
                names=mnemonic_type_labels, num_classes=num_mnemonic_types
            ),
        }
    )
    dataset = dataset.cast(features)

    # Split into train and validation
    splits: DatasetDict = dataset.train_test_split(
        test_size=val_ratio, stratify_by_column="main_type", seed=seed
    )

    return DatasetDict({"train": splits["train"], "val": splits["test"]})


def create_assistant_response(
    term: str,
    mnemonic: str,
    linguistic_reasoning: str,
    main_type: Optional[str] = None,
    sub_type: Optional[str] = None,
    use_json: bool = False,
) -> str:
    """Format the assistant response to include linguistic reasoning and mnemonic details.

    Args:
        term: The vocabulary term
        mnemonic: The mnemonic to include
        linguistic_reasoning: The linguistic reasoning bridging the term and mnemonic
        main_type: The main linguistic type if available
        sub_type: The sub type if available
        use_json: If True, return the response as a JSON object instead of a string

    Returns:
        Formatted assistant response
    """
    # Format the response
    if use_json:
        response_dict = {
            "mnemonic": mnemonic,
            "linguistic_reasoning": linguistic_reasoning,
            "main_type": main_type,
            "sub_type": sub_type,
        }

        response = json.dumps(response_dict, indent=2)
    else:
        response = f"{linguistic_reasoning}\n\nMnemonic: {mnemonic}"
        if main_type:
            response = f"{linguistic_reasoning}. It draws from {main_type} of the term '{term})'.\n\nMnemonic: {mnemonic}"
            if sub_type and sub_type != main_type:
                if sub_type == "context":
                    response += f"\n\nThis mnemonic also provides a contextual clue to remember '{term}'."
                else:
                    response += f"\n\nThis mnemonic also relates to the {sub_type} of the term '{term}'."

    return response


def create_chat_format(
    system_prompt_path: PathLike,
    user_prompt_path: PathLike,
    term: str,
    mnemonic: str,
    linguistic_reasoning: str = "",
    main_type: str = None,
    sub_type: Optional[str] = None,
    is_multimodal: bool = False,
) -> list[dict[str, Any]]:
    """Convert dataset entries to OpenAI's chat template format.

    [
        {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
        {"role": "user", "content": [
            {"type": "text", "text": user_prompt},
            {"type": "image", "url": "https://example.com/image.png"}  # Optional image content
        ]},
        {"role": "assistant", "content": [{"type": "text", "text": assistant_response}]}
    ]

    Args:
        system_prompt_path (PathLike): Path to the system prompt file
        user_prompt_path (PathLike): Path to the user prompt file
        term (str): The vocabulary term
        mnemonic (str): The mnemonic to include
        improved_mnemonic (str, optional): An improved mnemonic if available
        linguistic_reasoning (str): The linguistic reasoning bridging the term and mnemonic
        main_type (str): The main linguistic type if available
        sub_type (str, optional): The sub type if available
        is_multimodal (bool): If True, format for multimodal input (e.g., images). If False, format for text-only input. Default is False.

    Returns:
        List of message dictionaries in OpenAI chat format
    """
    try:
        # Read system prompt that emphasizes linguistic knowledge
        system_prompt = read_prompt(system_prompt_path)

        # TODO: Create more diverse user instructions and sample them here.
        user_prompt = read_prompt(user_prompt_path, vars={"term": term})

        # Format assistant response
        use_json = re.search(r"json|dict|structured", user_prompt) is not None
        assistant_response = create_assistant_response(
            term=term,
            mnemonic=mnemonic,
            linguistic_reasoning=linguistic_reasoning,
            main_type=main_type,
            sub_type=sub_type,
            use_json=use_json,
        )

        # Format in chat template
        if is_multimodal:
            messages = [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": system_prompt}],
                },
                {"role": "user", "content": [{"type": "text", "text": user_prompt}]},
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": assistant_response}],
                },
            ]
        else:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
                {"role": "assistant", "content": assistant_response},
            ]

        return messages
    except Exception as e:
        logger.error(f"Error creating chat format: {e}")
        # Return a placeholder message in case of error
        raise e


def create_hf_chat_dataset(dataset_dict: DatasetDict) -> DatasetDict:
    """Transform the Mnemonic dataset to include a messages field for chat format.

    Args:
        dataset_dict: The dataset dictionary containing train and validation splits

    Returns:
        Reformatted dataset with messages field
    """
    # Create a new DatasetDict to store the transformed datasets
    chat_dataset_dict = DatasetDict()

    # Process each split in the dataset
    for split_name, split_dataset in dataset_dict.items():
        logger.info(f"Processing {split_name} split with {len(split_dataset)} examples")

        # Define the map function to add the messages column
        def add_messages_column(example):
            messages = create_chat_format(
                system_prompt_path=const.FILE_PROMPT_FINETUNE_SYSTEM,
                user_prompt_path=const.FILE_PROMPT_FINETUNE_USER,
                term=example["term"],
                mnemonic=example["mnemonic"],
                linguistic_reasoning=example["linguistic_reasoning"],
                main_type=example["main_type"],
                sub_type=example["sub_type"],
            )
            return {"messages": messages}

        # Apply the transformation to create a new dataset with only messages column
        transformed_dataset = split_dataset.map(
            add_messages_column,
            remove_columns=["sub_type"],
            desc=f"Transforming {split_name} split",
        )

        # Add the transformed dataset to the new DatasetDict
        chat_dataset_dict[split_name] = transformed_dataset

        # Log one example "messages" to verify the transformation
        logger.info(
            "An example row from the transformed dataset:",
            split=split_name,
            example=transformed_dataset[0],
        )

    return chat_dataset_dict


if __name__ == "__main__":
    mnemonic_dataset: DatasetDict = create_hf_mnemonic_dataset(
        # input_path=const.MNEMONIC_DB_URI,
        input_path=const.SEED_IMPROVED_CSV,
        val_ratio=0.2,
    )
    chat_dataset: DatasetDict = create_hf_chat_dataset(mnemonic_dataset)
    push_data_to_hf(dataset_dict=mnemonic_dataset, repo_id=const.HF_MNEMONIC_DATASET)
    push_data_to_hf(dataset_dict=chat_dataset, repo_id=const.HF_CHAT_DATASET)
