"""Local module for processing data, including.

- Loading, cleaning, and combining data
- Splitting datasets into training and testing sets
- Pushing datasets to the Hugging Face hub
"""

import logging
import re
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd
from datasets import ClassLabel, load_dataset

if TYPE_CHECKING:
    from datasets import Dataset, DatasetDict

from src.data.data_loaders import load_local_dataset
from src.utils import constants as c
from src.utils.aliases import ExtensionsType, PathLike
from src.utils.common import login_hf_hub
from src.utils.error_handling import check_dir_path, check_file_path

# Set up logging to console
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())
logger.handlers[0].setFormatter(
    logging.Formatter("%(levelname)s - %(funcName)s - %(message)s")
)


def load_clean_txt_csv_data(dir_path: PathLike) -> pd.DataFrame:
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

    # Read only the first two columns, skipping the first two rows
    for path in file_paths:
        if path.suffix == c.TXT_EXT:
            temp_data = pd.read_csv(
                path,
                sep="\t",
                header=None,
                skiprows=2,
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
        output_path, new_ok=True, extensions=[c.CSV_EXT, c.PARQUET_EXT]
    )
    output_path_obj.parent.mkdir(parents=True, exist_ok=True)

    # Write to output file
    if output_path_obj.suffix == c.CSV_EXT:
        combined_df.to_csv(output_path_obj, index=False)
    elif output_path_obj.suffix == c.PARQUET_EXT:
        combined_df.to_parquet(output_path_obj, index=False)
    else:
        raise ValueError("Invalid output format. Must be either 'csv' or 'parquet'.")

    logger.info(f"Saved {combined_df.shape[0]} unique terms to '{str(output_path)}'.")

    return combined_df


def train_test_split(dataset: "Dataset", test_size: float = 0.2) -> "DatasetDict":
    """Split a dataset into training and testing sets.

    Args:
        dataset (Dataset): The dataset to split.
        test_size (float): The proportion of the dataset to include in the test split. Defaults to 0.2.

    Returns:
        DatasetDict: A dictionary containing the training and testing sets.
    """
    # Split the dataset into training and testing sets
    dataset_dict = dataset.train_test_split(
        test_size=test_size,
        shuffle=True,
        seed=42,  # stratify_by_column=c.CATEGORY_COL
    )
    logger.info(f"\n{dataset_dict}")

    # Return the training and testing sets
    return dataset_dict


def push_to_hf_hub(
    dataset: "Dataset",
    repo_id: str = c.HF_DATASET_NAME,
    private: bool = False,
    **kwargs,
):
    """Push a dataset to the Hugging Face hub.

    Args:
        dataset (Dataset or DatasetDict): The dataset to push to the Hugging Face hub.
        repo_id (str): The Hugging Face repository ID. Defaults to the value in 'utils/constants.py'.
        private (bool): Whether to make the dataset private. Defaults to False.
        **kwargs: Additional keyword arguments for the push_to_hub() function. See documentation: https://huggingface.co/docs/datasets/main/en/package_reference/main_classes#datasets.Dataset.push_to_hub for more details.
    """
    # Login to Hugging Face with a write token
    login_hf_hub(write_permission=True)
    dataset.push_to_hub(
        repo_id=repo_id,
        private=private,
        **kwargs,
    )
    logger.info(
        f"Pushed dataset to Hugging Face hub. Go to https://huggingface.co/datasets/{repo_id} to view the dataset."
    )


# combine_datasets(RAW_DATA_DIR, COMBINED_DATASET_CSV)

if __name__ == "__main__":
    local_classified_dataset: "Dataset" = load_local_dataset(
        file_path=c.COMBINED_DATASET_CSV
    )

    splits = train_test_split(local_classified_dataset)
    push_to_hf_hub(splits, private=False)
