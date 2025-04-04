# src/data/format_and_upload_dataset.py
"""Module to format mnemonic dataset for HuggingFace and upload to the hub.

Main functions:
- create_hf_mnemonic_dataset: Create a HuggingFace dataset from mnemonic data.
- create_hf_chat_dataset: Create a HuggingFace dataset in chat format.
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


# TODO: remove the function after refactoring
def create_class_dataset(
    input_path: PathLike,
    select_col_names: str | list[str] = None,
    val_ratio: float = 0.2,
    seed: int = 42,
) -> DatasetDict:
    """Create a HuggingFace dataset where there are clearer class labels for the mnemonics.

    Args:
        input_path (PathLike): Path to the input data (filepath or SQLite)
        select_col_names (str | list[str]): Column names to select from the dataset
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
    select_col_names = select_col_names or Mnemonic.model_fields.keys()

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


# TODO: convert to chat format {"role": "system", "content": "...", "role": "user", "content": "..."}

# TODO: convert to grpo, conversational format: columns prompts and completions will be lists of message dictionaries.
ds = load_dataset(const.HF_CONST.DATASET_NAME)
print(ds)
# DatasetDict(
#     {
#         train: Dataset(
#             {features: ["term", "instruction", "reasoning", "solution"], num_rows: 1870}
#         )
#     }
# )
