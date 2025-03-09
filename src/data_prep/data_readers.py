"""Module for reading data from various sources, into various formats (json, csv)."""

import csv
import json
from typing import Any

from src.utils import PathLike, check_file_path


def read_csv_file(file_path: "PathLike") -> list[dict[str, Any]]:
    """Read a CSV file and return its contents as a list of dictionaries.

    Args:
        file_path: Path to the CSV file

    Returns:
        List of dictionaries representing CSV rows
    """
    # Validate path using existing utility
    validated_path = check_file_path(file_path, extensions=[".csv"])

    rows = []
    with validated_path.open("r", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            # Clean up whitespace in values
            rows.append(
                {k: v.strip() if isinstance(v, str) else v for k, v in row.items()}
            )

    return rows


def read_json_file(file_path: "PathLike") -> list[dict[str, Any]]:
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


def read_jsonl_file(file_path: "PathLike") -> list[dict[str, Any]]:
    """Read a JSON Lines file and return its contents as a list of dictionaries.

    Args:
        file_path: Path to the JSON Lines file
    Returns:
        List of dictionaries representing JSON Lines objects
    """
    validated_path = check_file_path(file_path, extensions=[".jsonl"])

    data = []
    with validated_path.open("r", encoding="utf-8") as jsonlfile:
        for line in jsonlfile:
            if line.strip():  # Skip empty lines
                data.append(json.loads(line))

    return data


def read_txt_file(file_path: "PathLike") -> list[str]:
    """Read a text file and return its contents as a list of strings.

    Args:
        file_path: Path to the text file
    Returns:
        List of strings representing lines in the text file
    """
    validated_path = check_file_path(file_path, extensions=[".txt"])

    with validated_path.open("r", encoding="utf-8") as txtfile:
        data = [line.strip() for line in txtfile if line.strip()]

    return data
