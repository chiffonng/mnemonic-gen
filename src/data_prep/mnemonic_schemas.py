"""Module for pydantic models for converting data to JSON schema."""

import logging
from typing import Annotated, Optional

from pydantic import BaseModel, BeforeValidator, ConfigDict, Field
from pydantic.alias_generators import AliasGenerator, to_camel, to_snake

from src.data_prep import read_csv_file
from src.data_prep.data_validators import (
    ExplicitEnum,
    validate_enum_field,
    validate_mnemonic,
    validate_term,
)

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
