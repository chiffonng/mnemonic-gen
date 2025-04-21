"""Module for reading data from various sources, into various formats (json, csv)."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pandas as pd
from structlog import getLogger

from src import constants as const
from src.utils.error_handlers import check_file_path

if TYPE_CHECKING:
    from typing import Any, Union

    from structlog.stdlib import BoundLogger

    from src.utils.types import PathLike

# Set up logging
logger: BoundLogger = getLogger(__name__)


def read_csv_file(
    file_path: PathLike, **kwargs
) -> Union[pd.DataFrame, dict[str, dict[str, Any]], list[dict[str, Any]], str]:
    """Read a CSV file and return its contents as a DataFrame or other formats.

    Args:
        file_path: Path to the CSV file
        **kwargs: Additional arguments to process dataframe further

    Returns:
        DataFrame or
        list of dictionaries representing CSV rows or
        dict (column -> {index -> value}) or
        str (json formatted string) representing the CSV file path
    """
    validated_path = check_file_path(file_path, extensions=[const.Extension.CSV])

    df = pd.read_csv(validated_path, na_values=[None], keep_default_na=False)

    if kwargs.get("to_dict", False):
        content: dict[str, dict[str, Any]] = df.to_dict()
    elif kwargs.get("to_list", False) or kwargs.get("to_records", False):
        content: list[dict[str, Any]] = df.to_dict(orient="records")
    elif kwargs.get("to_json", False):
        json_str: str = df.to_json(orient="records")
        content: str = json.dumps(json.loads(json_str), indent=4)
    elif kwargs.get("to_jsonl", False):
        json_str: str = df.to_json(orient="records", lines=True)
        content: str = json.dumps(json.loads(json_str), indent=4)
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
    validated_path = check_file_path(
        file_path, extensions=[const.Extension.JSON, const.Extension.JSONL]
    )

    with validated_path.open("r", encoding="utf-8") as jsonfile:
        if validated_path.suffix == const.Extension.JSON:
            data = json.load(jsonfile)
        elif validated_path.suffix == const.Extension.JSONL:
            data = [json.loads(line) for line in jsonfile if line.strip()]
        data = json.load(jsonfile)

    return data


def read_txt_file(
    file_path: PathLike,
    remove_empty_lines: bool = True,
    by_lines: bool = False,
) -> list[str]:
    """Read a text file and return its contents as a list of strings.

    Args:
        file_path: Path to the text file
        remove_empty_lines: Whether to remove empty lines
        by_lines: Whether to read the file line by line

    Raises:
        See `src.utils.error_handlers.check_file_path` for possible exceptions

    Returns:
        List of strings representing lines in the text file
    """
    validated_path = check_file_path(file_path, extensions=[const.Extension.TXT])

    with validated_path.open("r", encoding="utf-8") as txtfile:
        if by_lines and remove_empty_lines:
            return [line.strip() for line in txtfile]
        elif by_lines and not remove_empty_lines:
            return [line for line in txtfile]

        elif remove_empty_lines:
            data = [line.strip() for line in txtfile if line.strip()]
        else:
            data = [line.strip() for line in txtfile]

    return data


def write_jsonl_file(data: list[dict[str, Any]], file_path: PathLike) -> None:
    """Write a list of dictionaries to a JSONL file.

    Args:
        data: List of dictionaries to write
        file_path: Path to the output JSONL file
    """
    file_path = check_file_path(
        file_path, new_ok=True, to_create=True, extensions=[const.Extension.JSONL]
    )

    with file_path.open("w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
