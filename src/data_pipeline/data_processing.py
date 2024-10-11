"""A module for processing data, combining them from various sources and load into usable format(s)."""

import logging
import re
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def load_parquet_data(path: Path | str) -> pd.DataFrame:
    """Load parquet data into a dataframe.

    Args:
        path (Path | str): The path to the parquet data.

    Returns:
        df (pd.DataFrame): The parquet data as a dataframe.
    """
    df = pd.DataFrame()
    paths = Path(path).rglob("*.parquet")

    if not paths:
        logger.error("No parquet files found in the specified path.")
        raise FileNotFoundError("No parquet files found in the specified path.")

    for path in paths:
        temp_data = pd.read_parquet(path, engine="pyarrow")
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
    paths = [p for p in Path(path).rglob("*") if p.suffix in [".txt", ".csv"]]
    logger.info(f"Loading txt/csv files from {paths}.")

    if not paths:
        logger.error("No txt or csv files found in the specified path.")
        raise FileNotFoundError("No txt or csv files found in the specified path.")

    # Read only the first two columns
    for path in paths:
        if path.suffix == ".txt":
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

    # Remove ALL double quotes and single quotes at the beginning and end of the mnemonic, including multiple occurrences
    text = re.sub(r'^["\']+', "", text)

    # Add a period at the end of the mnemonic if it doesn't already have one
    if text[-1] not in [".", "!", "?"]:
        text += "."

    return text


def combine_datasets(
    input_dir: Path | str = "data/raw",
    output_path: Path | str = "data/final",
    output_format: str = "csv",
) -> pd.DataFrame:
    """Combines an external dataset with a local dataset, cleans the data by removing duplicates, and saves the result to a specified format.

    Args:
        input_dir (Path | str):
            The directory containing the local dataset. Defaults to "data/raw".
        output_path (Path | str):
            The output directory where the combined data will be saved. Defaults to "data/final".
        output_format (str):
            The format in which to save the combined dataset ('csv' or 'parquet'). Defaults to 'csv'.

    Returns:
        pd.DataFrame: The cleaned, combined dataset.

    Raises:
        ValueError: If the provided output format is not 'csv' or 'parquet'.
    """
    if not Path(input_dir).exists():
        raise FileNotFoundError(f"Directory not found at {input_dir}.")

    # Set up output directories and file
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    output_file = f"{output_path}.{output_format}"

    # Load and combine the datasets
    external_df = load_parquet_data(input_dir)
    local_df = load_clean_txt_csv_data(input_dir)
    combined_df = pd.concat([local_df, external_df])

    # Clean the data
    combined_df.drop_duplicates(subset=["term"], inplace=True, keep="first")

    # Write to output file
    if output_format == "csv":
        combined_df.to_csv(output_file, index=False)
    elif output_format == "parquet":
        combined_df.to_parquet(output_file, index=False)
    else:
        raise ValueError("Invalid output format. Must be either 'csv' or 'parquet'.")

    logger.info(
        f"Saved combined data to '{output_file}' with {combined_df.shape[0]} unique terms."
    )

    return combined_df


combine_datasets()
