"""Module for loading data using pandas and/or Hugging Face datasets / HuggingFace hub."""

import logging
from typing import TYPE_CHECKING

import pandas as pd
from datasets import Dataset, DatasetDict, load_dataset

from src.huggingface import login_hf_hub
from src.utils import check_file_path

if TYPE_CHECKING:
    from typing import Optional

    from src.utils.aliases import PathLike
    from src.utils.constants import HF_DATASET_NAME, HF_MODEL_NAME, HF_TESTSET_NAME

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
        HuggingFace.Dataset: The loaded dataset.

    Raises:
        See src/utils/error_handling.py, check_file_path() for more details.
    """
    file_path = check_file_path(
        file_path, extensions=[".parquet", ".csv", ".json", ".txt"]
    )

    if file_path.suffix == ".parquet":
        dataset = load_dataset("parquet", data_files=str(file_path), **kwargs)
    elif file_path.suffix == ".csv":
        dataset = load_dataset("csv", data_files=str(file_path), **kwargs)
    elif file_path.suffix == ".json":
        dataset = load_dataset("json", data_files=str(file_path), **kwargs)
    elif file_path.suffix == ".txt":
        df = load_txt_file(file_path=file_path)
        dataset = Dataset.from_pandas(df)

    else:
        raise ValueError(
            f"Invalid file extension: {file_path.suffix}. Must be one of: '.parquet', '.csv', '.json', '.txt'."
        )

    logger.info(f"Loaded dataset from {file_path}.")
    if isinstance(dataset, DatasetDict):
        dataset = dataset["train"]

    logger.debug(f"Type of dataset: {type(dataset)}.")
    logger.info(f"Data shape: {dataset.shape}.")
    logger.info(f"Features: {dataset.features}.")
    return dataset


def load_txt_file(
    file_path: "PathLike", split_name: str = "test", col_name: str = "term"
) -> "DatasetDict":
    """Load a txt file as a pandas DataFrame.

    Args:
        file_path (PathLike): Path to the txt file.
        split_name (str): The name of the split. Defaults to 'test'.
        col_name (str): The name of the column. Defaults to 'term'.

    Returns:
        DatasetDict: The loaded dataset dictionary {split_name: dataset}
    """
    file_path = check_file_path(file_path, extensions=[".txt"])

    with file_path.open("r") as f:
        data = f.readlines()

    df = pd.DataFrame(data, columns=[col_name])
    dataset = Dataset.from_pandas(df)

    return DatasetDict({split_name: dataset})


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

    if to_csv and file_path is not None:
        file_path = check_file_path(file_path, new_ok=True, extensions=[".csv"])
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
