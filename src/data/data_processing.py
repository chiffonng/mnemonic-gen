"""A module for processing data, combining them from various sources and load into usable format(s)."""

import logging
import re
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd
from datasets import load_dataset

from constants import (
    PARQUET_EXT,
    CSV_EXT,
    TXT_EXT,
    RAW_DATA_DIR,
    COMBINED_DATASET_CSV,
    COMBINED_DATASET_PARQUET,
)
from utils.error_handling import check_file_path, check_dir_path

if TYPE_CHECKING:
    from datasets import Dataset, DatasetDict

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())
logger.handlers[0].setFormatter(
    logging.Formatter("%(levelname)s - %(funcName)s - %(message)s")
)


def load_parquet_data(path: Path | str) -> pd.DataFrame:
    """Load parquet data into a dataframe.

    Args:
        path (Path | str): The path to the parquet data.

    Returns:
        df (pd.DataFrame): The parquet data as a dataframe.
    """
    df = pd.DataFrame()
    paths = check_dir_path(path, extensions=[PARQUET_EXT])

    if not paths:
        logger.error("No parquet files found in the specified path.")
        raise FileNotFoundError("No parquet files found in the specified path.")

    for path in paths:
        temp_data = pd.read_parquet(path)
        df = pd.concat([df, temp_data])

    # Lowercase terms
    df["term"] = df["term"].str.lower()

    logger.info(f"Read {df.shape[0]} rows from parquet files.")

    assert df.shape[1] == 2, "Data must have 2 columns."
    assert df.columns[0] == "term", "First column must be 'term'."
    assert df.columns[1] == "mnemonic", "Second column must be 'mnemonic'."

    return df


def load_clean_txt_csv_data(path: Path | str) -> pd.DataFrame:
    """Load txt or csv data into a dataframe and clean it.

    Args:
        path (Path | str): The path to the txt or csv data.

    Returns:
        df (pd.DataFrame): The txt or csv data as a dataframe.

    Raises:
        FileNotFoundError: If no txt or csv files are found in the specified path.
    """
    df = pd.DataFrame()
    paths = check_dir_path(path, extensions=[TXT_EXT, CSV_EXT])
    logger.info(f"Loading txt/csv files from {[str(p) for p in paths]}.")

    # Read only the first two columns, skipping the first two rows
    for path in paths:
        if path.suffix == TXT_EXT:
            temp_data = pd.read_csv(
                path,
                sep="\t",
                header=None,
                skiprows=2,
                usecols=[0, 1],
                skip_blank_lines=True,
                names=["term", "mnemonic"],
            )
        else:
            temp_data = pd.read_csv(path, names=["term", "mnemonic"], usecols=[0, 1])
        df = pd.concat([df, temp_data])

    logger.info(f"Read {df.shape[0]} rows from txt/csv files.")

    # Drop empty mnemonics
    df.dropna(subset=["mnemonic"], inplace=True)

    # Drop mnemonics with only 2 words or less
    df = df[df["mnemonic"].str.split().str.len() > 2]
    logger.info(
        f"From txt/csv files, kept {df.shape[0]} rows with meaningful mnemonics."
    )

    # Format mnemonics
    df["mnemonic"] = df["mnemonic"].apply(format_mnemonics)

    assert df["term"].str.islower().all(), "All terms should be lower case."
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
    input_dir: Path | str,
    output_path: Path | str,
) -> pd.DataFrame:
    """Combines an external dataset with a local dataset, cleans the data by removing duplicates, and saves the result to a csv or parquet file.

    Args:
        input_dir (Path | str):
            The directory containing the local dataset and the external dataset.
        output_path (Path | str):
            The output directory where the combined data will be saved
    Returns:
        pd.DataFrame: The cleaned, combined dataset.

    Raises:
        ValueError: If the provided output format is not 'csv' or 'parquet'.
    """
    input_dir = check_dir_path(input_dir)

    # Load and combine the datasets
    external_df = load_parquet_data(input_dir)
    local_df = load_clean_txt_csv_data(input_dir)
    combined_df = pd.concat([local_df, external_df], ignore_index=True)

    # Clean the data
    combined_df.drop_duplicates(subset=["term"], inplace=True, keep="first")
    combined_df["mnemonic"] = combined_df["mnemonic"].apply(format_mnemonics)

    # Set up output directories and file
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Write to output file
    if output_path.suffix == CSV_EXT:
        combined_df.to_csv(output_path, index=False)
    elif output_path.suffix == PARQUET_EXT:
        combined_df.to_parquet(output_path, index=False)
    else:
        raise ValueError("Invalid output format. Must be either 'csv' or 'parquet'.")

    logger.info(f"Saved {combined_df.shape[0]} unique terms to '{output_path}'.")

    return combined_df


def make_hf_dataset(path: Path) -> Dataset:
    """Create a Hugging Face dataset from a local dataset."""
    raise NotImplementedError


def train_test_split(dataset: Dataset | pd.DataFrame) -> DatasetDict:
    """Split a dataset into training and testing sets."""
    raise NotImplementedError


def push_to_hf_hub(dataset: DatasetDict, dataset_name: str) -> None:
    """Push a dataset to the Hugging Face hub."""
    raise NotImplementedError


combine_datasets(RAW_DATA_DIR, COMBINED_DATASET_CSV)
