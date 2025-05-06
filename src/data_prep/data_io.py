"""Module for reading data from various sources, into various formats (json, csv)."""

from __future__ import annotations

import json
from typing import (
    TYPE_CHECKING,
    Any,
    Literal,
    TypedDict,
    cast,
    overload,
)

import pandas as pd
from structlog import getLogger

from src import constants as const
from src.utils.error_handlers import check_file_path

if TYPE_CHECKING:
    from typing import Unpack

    from structlog.stdlib import BoundLogger

    from src.utils.types import DataType, PathLike

# Set up logging
logger: BoundLogger = getLogger(__name__)


class CSVReadOptions(TypedDict, total=False):
    """Options for reading a CSV file."""

    to_dict: bool
    to_list: bool
    to_records: bool
    to_json: bool
    to_jsonl: bool


@overload
def read_csv_file(file_path: PathLike) -> pd.DataFrame: ...


@overload
def read_csv_file(
    file_path: PathLike, *, to_dict: Literal[True]
) -> dict[str, dict[str, Any]]: ...


@overload
def read_csv_file(
    file_path: PathLike, *, to_list: Literal[True]
) -> list[dict[str, Any]]: ...


@overload
def read_csv_file(
    file_path: PathLike, *, to_records: Literal[True]
) -> list[dict[str, Any]]: ...


@overload
def read_csv_file(file_path: PathLike, *, to_json: Literal[True]) -> str: ...


@overload
def read_csv_file(file_path: PathLike, *, to_jsonl: Literal[True]) -> str: ...


def read_csv_file(file_path: PathLike, **kwargs: Unpack[CSVReadOptions]) -> DataType:
    """Read a CSV file and return its contents as a DataFrame or other formats.

    Args:
        file_path: Path to the CSV file
        **kwargs: Additional arguments to process dataframe further:
            - to_dict: Convert to dict format (column -> {index -> value})
            - to_list/to_records: Convert to list of dictionaries
            - to_json: Convert to JSON string with indentation
            - to_jsonl: Convert to JSONL format string

    Returns:
        Union[pd.DataFrame, dict[str, dict[str, Any]], list[dict[str, Any]], str]:
            - DataFrame if no kwargs provided
            - dict if to_dict=True
            - list of dicts if to_list=True or to_records=True
            - str (JSON) if to_json=True
            - str (JSONL) if to_jsonl=True
    """
    validated_path = check_file_path(file_path, extensions=[const.Extension.CSV])

    df = pd.read_csv(validated_path, na_values=[None], keep_default_na=False)

    if kwargs.get("to_dict", False):
        return cast(DataType, df.to_dict())
    elif kwargs.get("to_list", False) or kwargs.get("to_records", False):
        return cast(DataType, df.to_dict(orient="records"))
    elif kwargs.get("to_json", False):
        json_str = df.to_json(orient="records")
        return cast(DataType, json.dumps(json.loads(json_str), indent=4))
    elif kwargs.get("to_jsonl", False):
        json_str = df.to_json(orient="records", lines=True)
        return cast(DataType, json.dumps(json.loads(json_str), indent=4))
    else:
        return cast(DataType, df)


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
            # Ensure data is a list of dictionaries
            if isinstance(data, dict):
                data = [data]
            elif not isinstance(data, list):
                raise ValueError(
                    f"Expected JSON file to contain a list or dict, got {type(data)}"
                )
        elif validated_path.suffix == const.Extension.JSONL:
            data = [json.loads(line) for line in jsonfile if line.strip()]

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
