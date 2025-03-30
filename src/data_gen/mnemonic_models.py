"""Public API for the mnemonic schemas."""

from __future__ import annotations

from typing import Annotated, Optional
from uuid import UUID, uuid4

from instructor import OpenAISchema
from pydantic import BeforeValidator
from pydantic.json_schema import SkipJsonSchema
from sqlmodel import Field, SQLModel

from src.data_prep.data_validators import (
    ExplicitEnum,
    validate_enum_field,
    validate_mnemonic,
    validate_term,
)


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


class Mnemonic(SQLModel, OpenAISchema, table=True):
    """Ideal mnemonic model. Fields: term, mnemonic, main_type, sub_type, linguistic_reasoning."""

    # Don't send the id field to OpenAI for schema generation
    id: SkipJsonSchema[UUID] = Field(default_factory=lambda: uuid4(), primary_key=True)
    term: Annotated[str, BeforeValidator(validate_term)] = Field(
        ...,
        description="The vocabulary term.",
        unique=True,
        index=True,
    )
    reasoning: str = Field(
        ...,
        description="The linguistic reasoning for the mnemonic.",
    )
    mnemonic: Annotated[str, BeforeValidator(validate_mnemonic)] = Field(
        ...,
        description="The mnemonic aid for the term.",
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
        description="The sub type of the mnemonic, if applicable.",
    )
