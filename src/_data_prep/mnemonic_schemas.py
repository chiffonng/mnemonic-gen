"""Module for pydantic models for converting data to JSON schema."""

from __future__ import annotations

import logging
import re
from typing import Annotated, Optional

from pydantic import (
    AliasGenerator,
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    model_validator,
)
from pydantic.alias_generators import to_camel, to_snake
from sqlmodel import Field as SQLField
from sqlmodel import SQLModel

from src._data_prep.data_validators import (
    ExplicitEnum,
    validate_enum_field,
    validate_mnemonic,
    validate_term,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    logger.addHandler(logging.StreamHandler())

default_config_dict = ConfigDict(
    populate_by_name=True,
    str_strip_whitespace=True,
    alias_generator=AliasGenerator(
        validation_alias=lambda field_name: to_snake(field_name),
        serialization_alias=lambda field_name: to_camel(field_name),
    ),
)


class BasicMnemonic(BaseModel):
    """Basic mnemonic model."""

    model_config = default_config_dict

    term: Annotated[str, BeforeValidator(validate_term)] = Field(
        ..., description="The vocabulary term.", max_length=100, min_length=1
    )
    mnemonic: Annotated[str, BeforeValidator(validate_mnemonic)] = Field(
        ..., description="The mnemonic aid for the term.", max_length=300, min_length=5
    )
    # TODO: Add source_id field


class ImprovedMnemonic(BaseModel):
    """Include both old and improved mnemonic, together with linguistic reasoning."""

    improved_mnemonic: Annotated[str, BeforeValidator(validate_mnemonic)] = Field(
        ...,
        description="The improved mnemonic aid for the term. The first sentence should say linguistic reasoning for the mnemonic.",
        max_length=400,
        min_length=5,
    )
    linguistic_reasoning: str = Field(
        ...,
        description="The linguistic reasoning for the mnemonic.",
        max_length=100,
        min_length=5,
    )

    @model_validator(mode="before")
    @classmethod
    def set_linguistic_reasoning(cls, values: dict) -> dict:
        """Extract the linguistic reasoning from the mnemonic if not provided."""
        mnemonic = values.get("mnemonic", "")
        linguistic = values.get("linguistic_reasoning")
        if not linguistic and mnemonic:
            # This regex captures the first sentence ending with a period.
            match = re.match(r"^(.*?\.)\s*", mnemonic)
            if match:
                values["linguistic_reasoning"] = match.group(1).strip()
            else:
                values["linguistic_reasoning"] = mnemonic.strip()
        return values


# TODO: Move this to a public module
class MnemonicType(ExplicitEnum):
    """Enum for mnemonic types."""

    phonetics = "phonetics"
    orthography = "orthography"  # writing system
    etymology = "etymology"
    morphology = "morphology"
    semantic_field = "semantic-field"
    context = "context"
    unknown = "unknown"  # fallback for when the type is not recognized.

    @classmethod
    def get_types(cls) -> list[str]:
        """Return a list of all available types."""
        return [member.value for member in cls]


class MnemonicClassification(BaseModel):
    """Classification of the mnemonic."""

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
        description="The sub type of the mnemonic, if applicable.",
    )


# TODO: Move this to a public module
class Mnemonic(SQLModel, table=True):
    """Ideal mnemonic model. Fields: term, mnemonic, main_type, sub_type, linguistic_reasoning."""

    id: Optional[int] = SQLField(default=None, primary_key=True)
    term: Annotated[str, BeforeValidator(validate_term)] = SQLField(
        ...,
        description="The vocabulary term.",
        max_length=100,
        min_length=1,
        unique=True,
        index=True,
    )
    mnemonic: Annotated[str, BeforeValidator(validate_mnemonic)] = SQLField(
        ..., description="The mnemonic aid for the term.", max_length=400, min_length=5
    )
    linguitic_reasoning: str = SQLField(
        ...,
        description="The linguistic reasoning for the mnemonic.",
        max_length=100,
        min_length=5,
    )
    main_type: Annotated[
        MnemonicType, BeforeValidator(validate_enum_field(MnemonicType))
    ] = SQLField(
        ...,
        description="The main type of the mnemonic.",
    )
    sub_type: Annotated[
        Optional[MnemonicType], BeforeValidator(validate_enum_field(MnemonicType))
    ] = SQLField(
        default=None,
        description="The sub type of the mnemonic, if applicable.",
    )
