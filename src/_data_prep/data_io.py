"""Module for reading data from various sources, into various formats (json, csv)."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pandas as pd

from src.utils import constants as const
from src.utils.error_handlers import check_file_path

if TYPE_CHECKING:
    from typing import Any

    from src.utils.types import PathLike


def read_csv_file(file_path: PathLike, **kwargs) -> Any:
    """Read a CSV file and return its contents as a list of dictionaries.

    Args:
        file_path: Path to the CSV file
        **kwargs: Additional arguments to process dataframe further

    Returns:
        List of dictionaries representing CSV rows
    """
    validated_path = check_file_path(file_path, extensions=[const.CSV_EXT])

    df = pd.read_csv(validated_path, encoding="utf-8")

    # Process the dataframe further if needed
    to_dict = kwargs.get("to_dict", False)
    to_lst_dict = kwargs.get("to_lst_dict", False)
    to_json = kwargs.get("to_json", False)
    to_jsonl = kwargs.get("to_jsonl", False)

    if to_dict:
        content: dict[dict[str, Any]] = df.to_dict()
    elif to_lst_dict:
        content: list[dict[str, Any]] = df.to_dict(orient="records")
    elif to_json:
        content: str = df.to_json(orient="records")
    elif to_jsonl:
        content: str = df.to_json(orient="records", lines=True)
    else:
        return df

    return content


def read_json_file(file_path: PathLike) -> list[dict[str, Any]]:
    """Read a JSON file and return its contents as a list of dictionaries.

    Args:
        file_path: Path to the JSON file
    Returns:
        List of dictionaries representing JSON objects
    """
    # Validate path using existing utility
    validated_path = check_file_path(file_path, extensions=[".json"])

    with validated_path.open("r", encoding="utf-8") as jsonfile:
        data = json.load(jsonfile)

    return data


def read_jsonl_file(file_path: PathLike) -> list[dict[str, Any]]:
    """Read a JSON Lines file and return its contents as a list of dictionaries.

    Args:
        file_path: Path to the JSON Lines file
    Returns:
        List of dictionaries representing JSON Lines objects
    """
    validated_path = check_file_path(file_path, extensions=[const.JSONL_EXT])

    data = []
    with validated_path.open("r", encoding="utf-8") as jsonlfile:
        for line in jsonlfile:
            if line.strip():  # Skip empty lines
                data.append(json.loads(line))

    return data


def read_txt_file(file_path: PathLike) -> list[str]:
    """Read a text file and return its contents as a list of strings.

    Args:
        file_path: Path to the text file
    Returns:
        List of strings representing lines in the text file
    """
    validated_path = check_file_path(file_path, extensions=[const.TXT_EXT])

    with validated_path.open("r", encoding="utf-8") as txtfile:
        data = [line.strip() for line in txtfile if line.strip()]

    return data


def write_jsonl_file(data: list[dict[str, Any]], file_path: PathLike) -> None:
    """Write a list of dictionaries to a JSONL file.

    Args:
        data: List of dictionaries to write
        file_path: Path to the output JSONL file
    """
    file_path = check_file_path(
        file_path, new_ok=True, to_create=True, extensions=[".jsonl"]
    )

    with file_path.open("w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
