"""Convert data from one format to another."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

from src.utils.error_handlers import check_file_path

if TYPE_CHECKING:
    from src.utils.types import PathLike


def convert_csv_to_json(
    input_path: PathLike, output_path: PathLike, to_jsonl: bool = False
) -> str:
    """Convert CSV file to JSON file.

    Args:
        input_path (PathLike): Path to the input CSV file.
        output_path (PathLike): Path to the output JSON file.
        to_jsonl (bool): If True, convert to JSON Lines format. Defaults to False.

    Return:
        str: JSON content.
    """
    csv_path = check_file_path(input_path, extensions=[".csv"])
    output_path = check_file_path(
        output_path, extensions=[".json", ".jsonl"], new_ok=True
    )
    df = pd.read_csv(csv_path)

    if to_jsonl:
        content = df.to_json(output_path, orient="records", lines=True)
    else:
        content = df.to_json(output_path, orient="records")

    with output_path.open("w") as f:
        f.write(content)

    return content


def convert_csv_to_dict(input_path: PathLike) -> dict:
    """Convert CSV file to dictionary.

    Args:
        input_path (PathLike): Path to the input CSV file.

    Returns:
        dict: Dictionary representation of the CSV file.
    """
    csv_path = check_file_path(input_path, extensions=[".csv"])
    df = pd.read_csv(csv_path)
    return df.to_dict(orient="records")  # Convert DataFrame to dictionary
