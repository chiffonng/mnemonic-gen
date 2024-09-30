"""A module for reading, cleaning, processing data, including code to standardize or diversify mnemonics."""

from pathlib import Path

import pandas as pd


def read_parquet_data():
    """Read parquet data into a dataframe."""
    data = pd.DataFrame()
    paths = Path("data").rglob("*.parquet")
    for path in paths:
        temp_data = pd.read_parquet(path, engine="pyarrow")
        data = pd.concat([data, temp_data])

    return data


def combine_datasets(
    output_path: Path | str = "data/final", output_format: str = "csv"
) -> pd.DataFrame:
    """Combines an external dataset with a local dataset, cleans the data by removing duplicates, and saves the result to a specified format.

    Args:
        output_path (Path | str): The output directory where the combined data will be saved. Defaults to "data/final".
        output_format (str): The format in which to save the combined dataset ('csv' or 'parquet'). Defaults to 'csv'.

    Returns:
        pd.DataFrame: The cleaned, combined dataset.

    Raises:
        ValueError: If the provided output format is not 'csv' or 'parquet'.
    """
    # Set up output directories and
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    output_file = f"{output_path}.{output_format}"

    # Load the external dataset
    external_data_path = Path("data/processed/combined_data.csv")
    if external_data_path.exists():
        external_df = pd.read_csv(external_data_path)
    else:
        external_df = read_parquet_data()

    # Load the local dataset
    local_csv_files = [
        f
        for f in Path("data").rglob("*.csv")
        if str(f.resolve()) != str(Path(output_file).resolve())
    ]
    if not local_csv_files:
        raise FileNotFoundError("No CSV files found in the 'data' directory.")
    local_df = pd.concat([pd.read_csv(f) for f in local_csv_files])

    # Standardize column names
    local_df.rename(columns={"Word": "term", "Mnemonic": "mnemonic"}, inplace=True)

    # Combine the local and external datasets, keeping only the relevant columns (assumed to be the first 2 in local_df)
    combined_df = pd.concat([local_df.iloc[:, :2], external_df])

    # Clean the data
    combined_df["term"] = combined_df["term"].str.lower()
    combined_df.drop_duplicates(subset=["term"], inplace=True)

    # Write to output file
    if output_format == "csv":
        combined_df.to_csv(output_file, index=False)
    elif output_format == "parquet":
        combined_df.to_parquet(output_file, index=False)
    else:
        raise ValueError("Invalid output format. Must be either 'csv' or 'parquet'.")

    return combined_df
