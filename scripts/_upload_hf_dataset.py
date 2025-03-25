# src/data/format_and_upload_dataset.py
"""Module to format mnemonic dataset for HuggingFace and upload to the hub.

Main functions:
- create_hf_mnemonic_dataset: Create a HuggingFace dataset from mnemonic data.
- create_hf_chat_dataset: Create a HuggingFace dataset in chat format.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from datasets import ClassLabel, Dataset, DatasetDict, Features, Value
from src.data.data_hf import load_from_database, load_local_dataset
from src.data.mnemonic_models import MnemonicType
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

    dataset = dataset.select_columns(select_col_names)

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

    return DatasetDict({"train": splits["train"], "validation": splits["test"]})


def create_assistant_response(
    term: str,
    mnemonic: str,
    linguistic_reasoning: str,
    main_type: str = None,
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
        response = {
            "term": term,
            "mnemonic": mnemonic,
            "linguistic_reasoning": linguistic_reasoning,
            "main_type": main_type,
            "sub_type": sub_type,
        }
    else:
        response = f"{linguistic_reasoning} (drawing from {main_type} of the term '{term})'.\n\nMnemonic: {mnemonic}"
        if sub_type and sub_type != main_type:
            if sub_type == "context":
                response += f"\n\nThis mnenonic also provides a contextual clue to remember '{term}'."
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
) -> list[dict[str, Any]]:
    """Convert dataset entries to OpenAI's chat template format.

    [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
        {"role": "assistant", "content": assistant_response}
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
    Returns:
        List of message dictionaries in OpenAI chat format
    """
    # Read system prompt that emphasizes linguistic knowledge
    system_prompt = read_prompt(
        system_prompt_path, vars_json_path=const.FILE_PROMPT_PLACEHOLDER_DICT
    )
    # TODO: Create more diverse user instructions and sample them here.
    user_prompt = read_prompt(
        user_prompt_path, vars={"term": term, "mnemonic": mnemonic}
    )

    # Format assistant response
    use_json = (
        True if "json" or "dict" or "structured" in user_prompt.lower() else False
    )
    assistant_response = create_assistant_response(
        term=term,
        mnemonic=mnemonic,
        linguistic_reasoning=linguistic_reasoning,
        main_type=main_type,
        sub_type=sub_type,
        use_json=use_json,
    )

    # Format in OpenAI's chat template
    messages = [
        {"role": "system", "content": {"type": "text", "text": system_prompt}},
        {"role": "user", "content": [{"type": "text", "text": user_prompt}]},
        {
            "role": "assistant",
            "content": {"type": "text", "text": assistant_response},
        },
    ]

    return messages


# FIXME: fix the transformation function so that it can be applied to each row
def create_hf_chat_dataset(dataset_dict: DatasetDict) -> DatasetDict:
    """Reformat the Mnemonic dataset to include a messages field for chat format.

    Args:
        dataset_dict (DatasetDict): The dataset dictionary containing train and validation splits

    Returns:
        Reformatted dataset with messages field
    """
    # Create a new column 'messages' for each split in the dataset
    # use dataset.map() to apply the transformation to each row
    for split_name, split_dataset in dataset_dict.items():
        # Apply the transformation to each row
        split_dataset = split_dataset.map(
            lambda row: {
                "messages": create_chat_format(
                    system_prompt_path=const.FILE_PROMPT_GENERATE_SYSTEM,
                    user_prompt_path=const.FILE_PROMPT_GENERATE_USER,
                    term=row["term"],
                    mnemonic=row["mnemonic"],
                    linguistic_reasoning=row["linguistic_reasoning"],
                    main_type=row["main_type"],
                    sub_type=row["sub_type"],
                )
            },
            remove_columns=[
                "term",
                "mnemonic",
                "linguistic_reasoning",
                "main_type",
                "sub_type",
            ],
        )
        dataset_dict[split_name] = split_dataset

    return dataset_dict


mnemonic_dataset = create_hf_mnemonic_dataset(
    # input_path=const.MNEMONIC_DB_URI,
    input_path=const.SEED_IMPROVED_CSV,
    val_ratio=0.2,
)
chat_dataset = create_hf_chat_dataset(mnemonic_dataset)
# Upload the chat dataset to Hugging Face Hub
# push_data_to_hf(dataset_dict=mnemonic_dataset, repo_id=const.HF_MNEMONIC_DATASET)
