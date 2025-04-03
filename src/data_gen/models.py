"""Public API for the mnemonic schemas."""

from __future__ import annotations

from typing import Annotated, Any, Optional
from uuid import UUID, uuid4

from instructor import OpenAISchema
from instructor.utils import disable_pydantic_error_url
from pydantic import BaseModel, BeforeValidator, Field
from pydantic.json_schema import SkipJsonSchema

from src.data_prep.data_validators import (
    ExplicitEnum,
    validate_enum_field,
    validate_mnemonic,
    validate_term,
)

disable_pydantic_error_url()


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


class Mnemonic(BaseModel):
    """Mnemonic model. Fields: id (auto), term, reasoning, mnemonic, main_type, sub_type."""

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
        description="The mnemonic device for the term.",
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


class MnemonicResult(BaseModel):
    """Class representing the result of a mnemonic generation process."""

    reasoning: str
    solution: str

    class Config:
        """Pydantic model configuration."""

        arbitrary_types_allowed = True
        allow_population_by_field_name = True
