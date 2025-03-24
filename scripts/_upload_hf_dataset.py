# src/data/format_and_upload_dataset.py
"""Module to format mnemonic dataset for HuggingFace and upload to the hub."""

from __future__ import annotations

from typing import TYPE_CHECKING

from datasets import ClassLabel, Dataset, DatasetDict, Features, Value
from src._data_prep.data_io import push_data_to_hf
from src.data.data_loaders import load_from_database, load_local_dataset
from src.data.mnemonic_models import MnemonicType
from src.utils import constants as const
from structlog import getLogger

if TYPE_CHECKING:
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


mnemonic_dataset = create_hf_mnemonic_dataset(
    # input_path=const.MNEMONIC_DB_URI,
    input_path=const.SEED_IMPROVED_CSV,
    val_ratio=0.2,
)
push_data_to_hf(dataset_dict=mnemonic_dataset, repo_id=const.HF_MNEMONIC_DATASET)
