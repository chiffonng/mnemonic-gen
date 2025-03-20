"""Public API for the mnemonic schemas."""

from __future__ import annotations

from typing import Annotated, Optional

from pydantic import BeforeValidator
from sqlmodel import Field, SQLModel

from src.data.data_validators import (
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


class Mnemonic(SQLModel, table=True):
    """Ideal mnemonic model. Fields: term, mnemonic, main_type, sub_type, linguistic_reasoning."""

    id: Optional[int] = Field(default=None, primary_key=True)
    term: Annotated[str, BeforeValidator(validate_term)] = Field(
        ...,
        description="The vocabulary term.",
        max_length=100,
        min_length=1,
        unique=True,
        index=True,
    )
    mnemonic: Annotated[str, BeforeValidator(validate_mnemonic)] = Field(
        ..., description="The mnemonic aid for the term.", max_length=400, min_length=5
    )
    linguitic_reasoning: str = Field(
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
        description="The sub type of the mnemonic, if applicable.",
    )
