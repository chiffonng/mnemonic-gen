"""Module for reading data from various sources, into various formats (json, csv)."""

import csv
from typing import Any

from src.utils import check_file_path
from src.utils.aliases import PathLike


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
