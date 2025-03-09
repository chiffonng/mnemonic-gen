"""Module for pydantic models for converting data to JSON schema."""

import json
import logging
from typing import Annotated, Any, Optional

from pydantic import BaseModel, BeforeValidator, ConfigDict, Field
from pydantic.alias_generators import AliasGenerator, to_camel, to_snake

from src.data_prep import (
    ExplicitEnum,
    read_csv_file,
    validate_enum_field,
    validate_mnemonic,
    validate_term,
)
from src.utils import check_file_path
from src.utils.aliases import PathLike

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    logger.addHandler(logging.StreamHandler())


class BasicMnemonic(BaseModel):
    """Basic mnemonic model."""

    model_config = ConfigDict(
        populate_by_name=True,
        str_strip_whitespace=True,
        alias_generator=AliasGenerator(
            validation_alias=lambda field_name: to_snake(field_name),
            serialization_alias=lambda field_name: to_camel(field_name),
        ),
    )

    term: Annotated[str, BeforeValidator(validate_term)] = Field(
        ..., description="The vocabulary term.", max_length=100, min_length=1
    )
    mnemonic: Annotated[str, BeforeValidator(validate_mnemonic)] = Field(
        ..., description="The mnemonic aid for the term.", max_length=300, min_length=5
    )
    # TODO: Add source_id field


class MnemonicType(ExplicitEnum):
    """Enum for mnemonic types."""

    phonetics = "phonetics"
    orthography = "orthography"  # writing system
    etymology = "etymology"
    morphology = "morphology"
    semantic_field = "semantic-field"
    context = "context"
    other = "other"
    unknown = "unknown"  # fallback for when the type is not recognized.

    @classmethod
    def get_types(cls) -> list[str]:
        """Return a list of all available types."""
        return [member.value for member in cls]


class Mnemonic(BasicMnemonic):
    """Full mnemonic model."""

    linguistic_reasoning: str = Field(
        ...,
        description="The linguistic reasoning for the mnemonic.",
        max_length=100,
        min_length=5,
    )
    main_type: Annotated[
        MnemonicType, BeforeValidator(validate_enum_field(MnemonicType))
    ] = Field(
        ...,
        description="The main type of the mnemonic.",
    )
    sub_type: Annotated[
        Optional[MnemonicType], BeforeValidator(validate_enum_field(MnemonicType))
    ] = Field(
        default=None,
        description="The sub type of the mnemonic.",
    )

    def __str__(self) -> str:
        """Return the string representation of the mnemonic."""
        return f"{self.term}: {self.mnemonic} (types: {self.main_type} {self.sub_type})"


class FullMnemonic(Mnemonic):
    """Full mnemonic model with additional fields."""

    id: str = Field(..., description="The unique identifier for the mnemonic.")


class ImprovedMnemonic(Mnemonic):
    """Include both old and improved mnemonics."""

    improved_mnemonic: Annotated[str, BeforeValidator(validate_mnemonic)] = Field(
        ...,
        description="The improved mnemonic aid for the term.",
        max_length=300,
        min_length=5,
    )


def convert_csv_to_json(
    input_path: PathLike,
    output_path: Optional[PathLike] = None,
    model_class: type[BaseModel] = Mnemonic,
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
