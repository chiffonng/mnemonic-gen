"""Module for loading data using pandas and/or Hugging Face datasets / HuggingFace hub."""

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from datasets import ClassLabel, DatasetDict, load_dataset

if TYPE_CHECKING:
    from datasets import Dataset

import utils.constants as c
from utils.aliases import ExtensionsType, PathLike
from utils.common import login_hf_hub
from utils.error_handling import check_dir_path, check_file_path

# Set up logging to console
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())
logger.handlers[0].setFormatter(
    logging.Formatter("%(levelname)s - %(funcName)s - %(message)s")
)


def load_local_dataset(file_path: PathLike, **kwargs) -> "Dataset":
    """Load a dataset from a file (parquet or csv).

    Args:
        file_path (Path or str): Path to the file.
        kwargs: Additional keyword arguments for the Hugging Face load_dataset() function, such as 'data_files' or 'data_dir' or 'split'. See documentation: https://huggingface.co/docs/datasets/en/loading for more details.

    Returns:
        Dataset: The loaded dataset.

    Raises:
        See src/utils/error_handling.py, check_file_path() for more details.
    """
    file_path = check_file_path(file_path, extensions=[c.PARQUET_EXT, c.CSV_EXT])

    if file_path.suffix == c.PARQUET_EXT:
        dataset = load_dataset("parquet", data_files=str(file_path), **kwargs)
    elif file_path.suffix == c.CSV_EXT:
        dataset = load_dataset("csv", data_files=str(file_path), **kwargs)

    logger.info(f"Loaded dataset from {file_path}.")
    if isinstance(dataset, DatasetDict):
        dataset = dataset["train"]

    # Map numeric value to category name
    dataset = dataset.map(
        lambda x: {c.CATEGORY_COL: c.CATEGORY_NAMES[x[c.CATEGORY_COL] + 1]}
    )
    dataset = dataset.cast_column(c.CATEGORY_COL, ClassLabel(names=c.CATEGORY_NAMES))

    logger.debug(f"Type of dataset: {type(dataset)}.")
    logger.info(f"Data shape: {dataset.shape}.")
    logger.info(f"Features: {dataset.features}.")
    return dataset


def load_hf_dataset(
    repo_id: str = None,
    to_csv: bool = False,
    file_path: PathLike = None,
    **kwargs,
) -> "DatasetDict":
    """Load a dataset from the Hugging Face hub.

    Args:
        repo_id (str): The Hugging Face repository ID. Defaults to None.
        to_csv (bool): Whether to save the dataset to a csv file. Defaults to False.
        file_path (Path or str): The path to save the dataset to. Defaults to None.
        kwargs: Additional keyword arguments for the Hugging Face load_dataset() function, such as 'split'. See documentation: https://huggingface.co/docs/datasets/en/loading for more details.

    Returns:
        DatasetDict: The loaded dataset.
    """
    login_hf_hub()

    if repo_id is None:
        repo_id = c.HF_DATASET_REPO

    logger.info(f"Loading dataset from {repo_id}.")
    dataset = load_dataset(repo_id, **kwargs)

    if to_csv:
        file_path = check_file_path(file_path, new_ok=True, extensions=c.CSV_EXT)
        dataset.to_csv(file_path)
        logger.info(f"Saved dataset to {file_path}.")
    else:
        logger.info(f"Cached file(s): {dataset.cache_files}")

    return dataset


# Example usage
# smart_dataset: "Dataset" = load_hf_dataset(
#     "nbalepur/Mnemonic_SFT",
#     split="train+test",
#     to_csv=True,
#     file_path=c.SMART_DATASET_CSV,
# )

if __name__ == "__main__":
    # Load a dataset from the Hugging Face hub
    mnemonic_dataset: "Dataset" = load_hf_dataset()
    logger.info(f"\n\n{mnemonic_dataset}")
