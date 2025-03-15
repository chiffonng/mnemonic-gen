"""Converts data from one format to another."""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING

from src.data_prep import read_csv_file
from src.utils import PathLike, check_file_path

if TYPE_CHECKING:
    from typing import Any, Optional

    from pydantic import BaseModel

    from src.utils import PathLike

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    logger.addHandler(logging.StreamHandler())


def convert_csv_to_json(
    input_path: PathLike,
    model_class: type[BaseModel],
    output_path: Optional[PathLike] = None,
    field_mappings: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    """Convert a CSV file to JSON using the specified Pydantic model.

    Args:
        input_path (PathLike): Path to the input CSV file
        output_path (PathLike, optional): Optional path to save the JSON output. If None, JSON is not saved to disk.
        model_class (pydantic.BaseModel): Pydantic model class to use for validation and conversion
        field_mappings (dict[str, str]): Optional dictionary mapping CSV column names to model field names

    Returns:
        List of validated dictionaries conforming to the model schema
    """
    # Validate input path
    input_path = check_file_path(input_path, extensions=[".csv"])

    # Validate output path if provided
    if output_path:
        output_path = check_file_path(
            output_path, new_ok=True, to_create=True, extensions=[".json"]
        )

    # Set default field mappings if not provided
    if field_mappings is None:
        field_mappings = {}

    # Read CSV data
    csv_data = read_csv_file(input_path)

    # Process and validate data
    validated_data = []

    for i, row in enumerate(csv_data, start=1):
        try:
            # Map fields according to provided mappings
            mapped_row = {}
            for csv_field, value in row.items():
                model_field = field_mappings.get(csv_field, csv_field)
                mapped_row[model_field] = value

            # Convert to model instance
            model_instance = model_class(**mapped_row)

            # Add to validated data
            validated_data.append(model_instance.model_dump(by_alias=True))

        except Exception as e:
            logger.error(f"Error processing row {i}: {e}")

    # Write to output file if specified
    if output_path:
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(validated_data, f, ensure_ascii=False, indent=2)

    return validated_data


def convert_csv_to_jsonl(
    input_path: PathLike,
    model_class: type[BaseModel],
    output_path: Optional[PathLike] = None,
    field_mappings: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    """Convert a CSV file to JSON Lines using the specified Pydantic model.

    Args:
        input_path (PathLike): Path to the input CSV file
        model_class (pydantic.BaseModel): Pydantic model class to use for validation and conversion
        output_path (PathLike, optional): Optional path to save the JSON Lines output. If None, JSON Lines is not saved to disk.
        field_mappings (dict[str, str]): Optional dictionary mapping CSV column names to model field names

    Returns:
        List of validated dictionaries conforming to the model schema
    """
    # Validate input path
    input_path = check_file_path(input_path, extensions=[".csv"])

    # Validate output path if provided
    if output_path:
        output_path = check_file_path(
            output_path, new_ok=True, to_create=True, extensions=[".jsonl"]
        )

    # Set default field mappings if not provided
    if field_mappings is None:
        field_mappings = {}

    # Read CSV data
    csv_data = read_csv_file(input_path)

    # Process and validate data
    validated_data = []

    for i, row in enumerate(csv_data, start=1):
        try:
            # Map fields according to provided mappings
            mapped_row = {}
            for csv_field, value in row.items():
                model_field = field_mappings.get(csv_field, csv_field)
                mapped_row[model_field] = value

            # Convert to model instance
            model_instance = model_class(**mapped_row)

            # Add to validated data
            validated_data.append(model_instance.model_dump(by_alias=True))

        except Exception as e:
            logger.error(f"Error processing row {i}: {e}")

    # Write to output file if specified
    if output_path:
        with output_path.open("w", encoding="utf-8") as f:
            for item in validated_data:
                json.dump(item, f, ensure_ascii=False)
                f.write("\n")

    return validated_data


def convert_json_to_csv(
    input_path: PathLike,
    output_path: Optional[PathLike] = None,
) -> list[dict[str, Any]]:
    """Convert a JSON file to CSV.

    Args:
        input_path (PathLike): Path to the input JSON file
        output_path (PathLike, optional): Optional path to save the CSV output. If None, CSV is not saved to disk.

    Returns:
        List of dictionaries representing the JSON data
    """
    # Validate input path
    input_path = check_file_path(input_path, extensions=[".json"])

    # Validate output path if provided
    if output_path:
        output_path = check_file_path(
            output_path, new_ok=True, to_create=True, extensions=[".csv"]
        )

    # Read JSON data
    with input_path.open("r", encoding="utf-8") as f:
        json_data = json.load(f)

    # Write to output file if specified
    if output_path:
        with output_path.open("w", encoding="utf-8") as f:
            if isinstance(json_data, list):
                # Write CSV header
                header = json_data[0].keys()
                f.write(",".join(header) + "\n")
                # Write CSV rows
                for item in json_data:
                    f.write(",".join(str(item.get(key, "")) for key in header) + "\n")
            else:
                logger.error("JSON data is not a list. Cannot convert to CSV.")
                raise ValueError("JSON data is not a list. Cannot convert to CSV.")

    return json_data
