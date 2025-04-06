"""Public API for the mnemonic schemas."""

from __future__ import annotations

from typing import Annotated
from uuid import UUID, uuid4

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


class LinguisticFeature(ExplicitEnum):
    """Enum for mnemonic types."""

    phonetics = "phonetics"
    orthography = "orthography"  # writing system
    etymology = "etymology"
    morphology = "morphology"
    semantics = "semantics"
    custom = "custom"  # custom linguistic feature created by llm/user
    unknown = "unknown"  # fallback for when the type is not recognized.

    @classmethod
    def get_types(cls) -> list[str]:
        """Return a list of all available types."""
        return [member.value for member in cls]


class Mnemonic(BaseModel):
    """Mnemonic model. Fields: id (auto), term, reasoning, answer, linguistic_feature."""

    # Don't send the id field to OpenAI for schema generation
    id: SkipJsonSchema[UUID] = Field(default_factory=lambda: uuid4())
    term: Annotated[str, BeforeValidator(validate_term)] = Field(
        ...,
        description="The vocabulary term.",
    )
    reasoning: str = Field(
        ...,
        description="The linguistic reasoning for the mnemonic.",
    )
    answer: Annotated[str, BeforeValidator(validate_mnemonic)] = Field(
        ...,
        description="The mnemonic device for the term.",
    )
    linguistic_feature: Annotated[
        LinguisticFeature, BeforeValidator(validate_enum_field(LinguisticFeature))
    ] = Field(
        ...,
        description="The main type of the mnemonic.",
    )


class MnemonicResult(BaseModel):
    """Class representing the result of a mnemonic generation process."""

    reasoning: str
    answer: str

    class Config:
        """Pydantic model configuration."""

        validate_by_name = True
