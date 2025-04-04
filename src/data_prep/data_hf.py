"""Module for loading data into Hugging Face datasets formats, and uploading data to HuggingFace Hub."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd
from datasets import Dataset, DatasetDict, load_dataset
from structlog import getLogger

from src.constants import HF_CONST, Column, Extension
from src.utils import check_file_path
from src.utils.hf_utils import login_hf_hub

if TYPE_CHECKING:
    from typing import Optional

    from structlog.stdlib import BoundLogger

    from src.utils import PathLike

# Set up logging to console
logger: BoundLogger = getLogger(__name__)


def load_local_dataset(file_path: PathLike, **kwargs) -> Dataset:
    """Load a dataset from a file into a Hugging Face Dataset.

    Args:
        file_path (PathLike): Path to the file, which can be in `.parquet`, `.csv`, `.json`, or `.jsonl` format.
        kwargs: Additional keyword arguments for the Hugging Face load_dataset() function, such as 'data_files' or 'data_dir' or 'split'. See documentation: https://huggingface.co/docs/datasets/en/loading for more details.

    Returns:
        HuggingFace.Dataset: The loaded dataset.

    Raises:
        See src/utils/error_handling.py, check_file_path() for more details.
    """
    file_path = check_file_path(
        file_path,
        extensions=[Extension.PARQUET, Extension.CSV, Extension.JSON, Extension.JSONL],
    )

    if file_path.suffix == ".parquet":
        dataset = load_dataset("parquet", data_files=str(file_path), **kwargs)
    elif file_path.suffix == ".csv":
        dataset = load_dataset(
            "csv",
            data_files=str(file_path),
            keep_default_na=False,
            na_values=[""],  # nan becomes empty string
            **kwargs,
        )
    elif file_path.suffix == Extension.JSON or file_path.suffix == Extension.JSONL:
        dataset = load_dataset("json", data_files=str(file_path), **kwargs)

    logger.info("Loaded dataset", source=file_path)
    if isinstance(dataset, DatasetDict):
        dataset = dataset["train"]

    logger.info(
        "Dataset information",
        dataset=dataset,
        features=dataset.features,
        nrows=dataset.num_rows,
        ncols=dataset.num_columns,
    )
    return dataset


def load_txt_file(
    file_path: PathLike, split_name: str = "test", col_name: str = Column.TERM
) -> DatasetDict:
    """Load a txt file as a pandas DataFrame.

    Args:
        file_path (PathLike): Path to the txt file.
        split_name (str): The name of the split. Defaults to 'test'.
        col_name (str): The name of the column. Defaults to 'term'.

    Returns:
        DatasetDict: The loaded dataset dictionary {split_name: dataset}
    """
    file_path = check_file_path(file_path, extensions=[Extension.TXT])

    with file_path.open("r") as f:
        data = f.readlines()

    df = pd.DataFrame(data, columns=[col_name])
    dataset = Dataset.from_pandas(df)

    return DatasetDict({split_name: dataset})


def load_txt_by_lines(
    source_path: PathLike,
    sample_size: Optional[int] = None,
    seed: int = 42,
    split_name: str = "train",
) -> Dataset:
    """Load vocabulary from .txt into HuggingFace datasets.Dataset.

    Args:
        source_path (PathLike): Path to the source file containing vocabulary terms.
        sample_size (int, optional): Number of samples to take from the dataset. If None, use the full dataset.
        seed (int, optional): Random seed for shuffling the dataset when sampling. Defaults to 42.
        split_name (str, optional): The name of the dataset split to create. Defaults to 'train'.

    Returns:
        Dataset containing vocabulary terms
    """
    source_path = check_file_path(source_path, extensions=[Extension.TXT])
    ds = Dataset.from_dict(
        {Column.TERM: source_path.open("r").read().splitlines()}, split=split_name
    )
    logger.info("Loaded full .txt dataset", source=source_path, size=ds.num_rows)

    # Sample if requested
    if sample_size and sample_size < len(ds):
        ds = ds.shuffle(seed=seed).select(range(sample_size))

    if ds.num_rows == 0:
        logger.warning("Empty dataset. check the source file.", source=source_path)
        raise ValueError("The loaded dataset is empty.")
    else:
        logger.info(
            "View one example of .txt dataset", source=source_path, example=ds[0]
        )
    return ds


def push_data_to_hf(
    dataset_dict: DatasetDict, repo_id: str, private: bool = False, **kwargs: dict
) -> str:
    """Upload dataset to HuggingFace Hub.

    Args:
        dataset_dict: DatasetDict to upload
        repo_id: Repository ID on HuggingFace (username/dataset-name)
        private: Whether the repository should be private
        kwargs: Additional keyword arguments for the push_to_hub() method

    Returns:
        URL of the uploaded dataset
    """
    logger.info("Uploading dataset to HuggingFace", dataset=dataset_dict, repo=repo_id)

    login_hf_hub()

    # Push to HuggingFace Hub
    dataset_dict.push_to_hub(repo_id=repo_id, private=private, **kwargs)

    logger.info(
        "Successfully uploaded dataset to HuggingFace",
        url=f"https://huggingface.co/datasets/{repo_id}",
        private=private,
    )

    return repo_id


if __name__ == "__main__":
    # Load a dataset from the Hugging Face hub
    mnemonic_dataset: Dataset = load_dataset()
    logger.info(f"\n\n{mnemonic_dataset}")
