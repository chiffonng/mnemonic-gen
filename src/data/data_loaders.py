"""Module for loading data using pandas and/or Hugging Face datasets / HuggingFace hub."""

import logging
from typing import TYPE_CHECKING

from datasets import DatasetDict, load_dataset

if TYPE_CHECKING:
    from typing import Optional

    from datasets import Dataset

    from src.utils.aliases import PathLike

from src.huggingface import login_hf_hub
from src.utils import check_file_path

# Hugging Face collection
HF_DATASET_NAME = "chiffonng/en-vocab-mnemonics"  # <user>/<dataset_name>
HF_MODEL_NAME = "chiffonng/gemma2-9b-it-mnemonics"  # <user>/<model_name>

# Set up logging to console
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())
logger.handlers[0].setFormatter(
    logging.Formatter("%(levelname)s - %(funcName)s - %(message)s")
)


def load_local_dataset(file_path: "PathLike", **kwargs) -> "Dataset":
    """Load a dataset from a file (parquet or csv).

    Args:
        file_path (PathLike): Path to the file.
        kwargs: Additional keyword arguments for the Hugging Face load_dataset() function, such as 'data_files' or 'data_dir' or 'split'. See documentation: https://huggingface.co/docs/datasets/en/loading for more details.

    Returns:
        Dataset: The loaded dataset.

    Raises:
        See src/utils/error_handling.py, check_file_path() for more details.
    """
    file_path = check_file_path(file_path, extensions=[".parquet", ".csv"])

    if file_path.suffix == ".parquet":
        dataset = load_dataset("parquet", data_files=str(file_path), **kwargs)
    elif file_path.suffix == ".csv":
        dataset = load_dataset("csv", data_files=str(file_path), **kwargs)

    logger.info(f"Loaded dataset from {file_path}.")
    if isinstance(dataset, DatasetDict):
        dataset = dataset["train"]

    logger.debug(f"Type of dataset: {type(dataset)}.")
    logger.info(f"Data shape: {dataset.shape}.")
    logger.info(f"Features: {dataset.features}.")
    return dataset


def load_hf_dataset(
    repo_id: "Optional[str]" = None,
    to_csv: bool = False,
    file_path: "Optional[PathLike]" = None,
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
        repo_id = HF_DATASET_NAME

    logger.info(f"Loading dataset from {repo_id}.")
    dataset = load_dataset(repo_id, **kwargs)

    if to_csv:
        file_path = check_file_path(file_path, new_ok=True, extensions=[".csv"])
        if not file_path:
            raise ValueError(
                "Invalid file path. Must be a valid path of csv to save the dataset to."
            )
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
#     file_path=SMART_DATASET_CSV,
# )

if __name__ == "__main__":
    # Load a dataset from the Hugging Face hub
    mnemonic_dataset: "Dataset" = load_hf_dataset()
    logger.info(f"\n\n{mnemonic_dataset}")
