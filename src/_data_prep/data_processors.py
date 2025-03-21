"""Local module for processing data, including.

- Loading, cleaning, and combining data
- Splitting datasets into training, validation, and testing sets
- Pushing datasets to the Hugging Face hub
"""

# mypy: disable-error-code="arg-type"
from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd
import structlog
from datasets import Dataset, DatasetDict

from src.data import load_txt_file
from src.llms.huggingface import login_hf_hub
from src.utils import check_dir_path, check_file_path
from src.utils import constants as c

if TYPE_CHECKING:
    from structlog.stdlib import BoundLogger

    from src.utils import PathLike


# Set up logging to console
logger: BoundLogger = structlog.getLogger(__name__)


def load_clean_txt_csv_data(dir_path: PathLike) -> pd.DataFrame:  # type: ignore
    """Load txt or csv data into a dataframe and clean it.

    Args:
        dir_path (PathLike): The path to the txt or csv data.

    Returns:
        df (pd.DataFrame): The txt or csv data as a dataframe.

    Raises:
        FileNotFoundError: If no txt or csv files are found in the specified path.
    """
    df = pd.DataFrame()
    file_paths = check_dir_path(dir_path, extensions=[c.TXT_EXT, c.CSV_EXT])

    if not file_paths or isinstance(file_paths, Path):
        raise FileNotFoundError(f"No txt or csv files found in '{dir_path}'.")

    logger.info(f"Loading txt/csv files from {[str(p) for p in file_paths]}.")

    for path in file_paths:
        if path.suffix == c.TXT_EXT:
            temp_data = pd.read_csv(
                path,
                sep="\t",
                header=None,
                usecols=[0, 1],
                skip_blank_lines=True,
                names=[c.TERM_COL, c.MNEMONIC_COL],
            )
        else:
            temp_data = pd.read_csv(
                path, names=[c.TERM_COL, c.MNEMONIC_COL], usecols=[0, 1]
            )
        df = pd.concat([df, temp_data])

    logger.info(f"Read {df.shape[0]} rows from txt/csv files.")

    # Drop empty mnemonics
    df.dropna(subset=[c.MNEMONIC_COL], inplace=True)

    # Drop mnemonics with only 2 words or less
    df = df[df[c.MNEMONIC_COL].str.split().str.len() > 2]
    logger.info(
        f"From txt/csv files, kept {df.shape[0]} rows with meaningful mnemonics."
    )

    # Format mnemonics
    df[c.MNEMONIC_COL] = df[c.MNEMONIC_COL].apply(format_mnemonics)

    assert df[c.TERM_COL].str.islower().all(), "All terms should be lower case."
    return df


def format_mnemonics(text: str) -> str:
    """Use a consistent format for mnemonics.

    Args:
        text (str): The mnemonic text to format.

    Returns:
        str: The formatted mnemonic.
    """
    # Remove leading and trailing spaces
    text = text.strip()

    # Replace all double quotes that are not at the beginning or end of the text with single quotes
    text = re.sub(r'(?<!^)"(?!$)', "'", text)

    # Replace multiple quotes (e.g., "" or '') inside the text with a single quote
    text = re.sub(r'["\']{2,}', '"', text)

    # Add a period at the end of the mnemonic if it doesn't already have one
    if text and text[-1] not in [".", "!", "?"]:
        text += "."

    return text


def combine_datasets(
    input_dir: PathLike,
    output_path: PathLike,
) -> pd.DataFrame:
    """Combines an external dataset with a local dataset, cleans the data by removing duplicates, and saves the result to a csv or parquet file.

    Args:
        input_dir (PathLike):
            The directory containing the local dataset and the external dataset.
        output_path (PathLike):
            The output directory where the combined data will be saved. Valid file formats are 'csv' and 'parquet'.

    Returns:
        pd.DataFrame: The cleaned, combined dataset.

    Raises:
        ValueError: If the provided output format is not 'csv' or 'parquet'.
    """
    checked_input_dir = check_dir_path(input_dir)

    # Load and combine the datasets
    combined_df = load_clean_txt_csv_data(checked_input_dir)

    # Clean the data
    combined_df.drop_duplicates(subset=[c.TERM_COL], inplace=True, keep="first")
    combined_df[c.MNEMONIC_COL] = combined_df[c.MNEMONIC_COL].apply(format_mnemonics)

    # Set up output directories and file
    output_path_obj: Path = check_file_path(
        output_path, new_ok=True, to_create=True, extensions=[c.CSV_EXT, c.PARQUET_EXT]
    )

    # Write to output file
    if output_path_obj.suffix == c.CSV_EXT:
        combined_df.to_csv(output_path_obj, index=False)
    elif output_path_obj.suffix == c.PARQUET_EXT:
        combined_df.to_parquet(output_path_obj, index=False)
    else:
        raise ValueError("Invalid output format. Must be either 'csv' or 'parquet'.")

    logger.info(f"Saved {combined_df.shape[0]} unique terms to '{str(output_path)}'.")

    return combined_df


def train_val_split(
    combined_data_path: PathLike,
    val_size: float = 0.2,
    seed: int = 42,
) -> DatasetDict:
    """Split the combined dataset into training and validation sets, and prepare test set.

    Args:
        combined_data_path (PathLike): Path to the combined dataset (.csv or .parquet).
        val_size (float): The proportion of the dataset to include in the validation set.
        seed (int): Random seed for reproducibility.

    Returns:
        DatasetDict: Dictionary containing 'train' and 'validation' datasets.
    """
    # Load the combined dataset
    df = (
        pd.read_csv(combined_data_path)
        if str(combined_data_path).endswith(".csv")
        else pd.read_parquet(combined_data_path)
    )
    logger.info(f"Loaded combined dataset with {len(df)} examples")

    # Convert to Dataset
    combined_dataset = Dataset.from_pandas(df)

    # Split into train and validation sets
    train_val_split = combined_dataset.train_test_split(test_size=val_size, seed=seed)
    return DatasetDict(
        {"train": train_val_split["train"], "val": train_val_split["test"]}
    )


def save_splits_locally(
    splits: DatasetDict, output_dir_path: str, format: str = "csv"
) -> dict[str, Path]:
    """Save the dataset splits to local files.

    Args:
        splits (dict[str, Dataset]): Dictionary of dataset splits.
        output_dir_path (str): Directory to save the splits.
        format (str): Format to save the splits in ('csv' or 'parquet'). Defaults to 'csv'.

    Returns:
        dict[str, Path]: Dictionary mapping split names to file paths.
    """
    output_dir = Path(output_dir_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    file_paths = {}

    for split_name, dataset in splits.items():
        if format == "csv":
            file_path = output_dir / f"{split_name}.csv"
            dataset.to_csv(str(file_path), index=False)
        else:
            file_path = output_dir / f"{split_name}.parquet"
            dataset.to_parquet(str(file_path), index=False)

        file_paths[split_name] = file_path
        logger.info(
            f"Saved {split_name} split with {len(dataset)} examples to {file_path}"
        )

    return file_paths


def push_data_to_hf_hub(
    dataset_dict: DatasetDict,
    repo_id: str,
    private: bool = False,
    **kwargs,
):
    """Push dataset splits to the Hugging Face hub.

    Args:
        dataset_dict (DatasetDict): Dictionary containing splits and their according datasets.
        repo_id (str): The Hugging Face repository ID. Defaults to the value in 'utils/constants.py'.
        private (bool): Whether to make the dataset private. Defaults to False.
        **kwargs: Additional keyword arguments for the push_to_hub() function.
    """
    # Login to Hugging Face with a write token
    login_hf_hub(write_permission=True)

    # Push to HF Hub
    dataset_dict.push_to_hub(
        repo_id=repo_id,
        private=private,
        **kwargs,
    )

    logger.info(
        f"Pushed dataset to Hugging Face hub. Go to https://huggingface.co/datasets/{repo_id} to view the dataset."
    )


# Get train and validation datasetdict
# And
if __name__ == "__main__":
    # Load and clean the data

    # Split the data into training and validation sets
    splits = train_val_split(
        combined_data_path=c.COMBINED_DATASET_CSV,
        val_size=0.2,
        seed=42,
    )
    test_split = load_txt_file(c.FINAL_TEST_DATASET_TXT)

    # Save the splits locally
    split_file_paths = save_splits_locally(
        splits=splits,
        output_dir_path=c.FINAL_DATA_DIR,
        format="csv",
    )

    # Push the splits to the Hugging Face hub
    push_data_to_hf_hub(dataset_dict=splits, repo_id=c.HF_DATASET_NAME)
    push_data_to_hf_hub(dataset_dict=test_split, repo_id=c.HF_TESTSET_NAME)
    logger.info("Finished processing data.")
