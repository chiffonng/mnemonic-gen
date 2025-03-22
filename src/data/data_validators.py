"""Module for data validation functions."""

from __future__ import annotations

import json
from enum import Enum
from typing import TYPE_CHECKING

from pydantic import ValidationError
from structlog import getLogger

if TYPE_CHECKING:
    from typing import Any, Callable

    from structlog.stdlib import BoundLogger

    from src.utils.types import ModelT, StrNoneType

# Set up logging
logger: BoundLogger = getLogger(__name__)


class ExplicitEnum(str, Enum):
    """Enum with more explicit error message for missing values."""

    def __str__(self):
        """Return the string representation of the enum value."""
        return self.value

    @classmethod
    def _missing_(cls, value):
        raise ValueError(
            f"{value} is not a valid {cls.__name__}, please select one of {list(cls._value2member_map_.keys())}"
        )


def validate_enum_field(
    enum_class: type[Enum],
) -> Callable[[StrNoneType], StrNoneType]:
    """Create a validator for enum fields."""

    def validator(value: StrNoneType) -> StrNoneType:
        """Validate the enum field."""
        if value is None or value == "":
            return None
        elif isinstance(value, str):
            value = value.strip().lower()
            if not value:
                return None
            try:
                return enum_class(value)
            except ValueError:
                # If value doesn't match enum, use the _missing_ method
                if hasattr(enum_class, "_missing_"):
                    return enum_class._missing_(value)
                raise ValueError(
                    f"Invalid value '{value}' for {enum_class.__name__}"
                ) from None
        return value

    return validator


def validate_term(value: str) -> str:
    """Validate the term field."""
    cleaned = value.strip().lower()
    if not cleaned:
        raise ValueError("Term cannot be empty")
    return cleaned


def validate_mnemonic(value: str) -> str:
    """Validate the mnemonic field."""
    cleaned = value.strip()
    if not cleaned:
        raise ValueError("Mnemonic cannot be empty")

    # Ensure mnemonic ends with proper punctuation
    if cleaned and cleaned[-1] not in [".", "!", "?"]:
        cleaned += "."

    return cleaned


def validate_content_against_schema(content: Any, schema: type[ModelT]) -> ModelT:
    """Validate the content against the schema.

    Args:
        content: The content to validate (string, dict, or model instance)
        schema: The schema to validate against

    Returns:
        The content parsed and validated against the schema

    Raises:
        ValueError: If the content is not a string or dictionary
        json.JSONDecodeError: If the content is not valid JSON
        ValidationError: If the content does not match the schema
    """
    if content is None:
        logger.exception("Content is None")
        raise ValueError("Content is None")
    try:
        # if content is already schema, return it
        if isinstance(content, schema):
            return content
        elif isinstance(content, dict):
            logger.debug("Validating dictionary against schema")
            return schema.model_validate(content)

        elif isinstance(content, (str, bytes, bytearray)):
            try:
                logger.debug("Validating JSON string against schema")
                return schema.model_validate_json(content)
            except ValidationError as e:
                # Try parsing JSON first then validate
                logger.warning(
                    "Direct validation failed, parsing JSON first", error=str(e)
                )
                parsed_json = json.loads(content)
                return schema.model_validate(parsed_json)

        # Fallback for other types
        else:
            logger.warning(f"Unexpected content type: {type(content)}")
            content_str = str(content)
            return schema.model_validate_json(content_str)

    except json.JSONDecodeError as json_error:
        content_str = str(content)
        snippet = content_str[:100] + "..." if len(content_str) > 100 else content_str
        logger.exception("JSON decode error", error=str(json_error), content=snippet)

        # Try to fix incomplete JSON
        fixed_content = _attempt_fix_incomplete_json(str(content))
        fixed_snippet = (
            fixed_content[:100] + "..." if len(fixed_content) > 100 else fixed_content
        )
        logger.debug("Attempting with fixed JSON", fixed_content=fixed_snippet)

        try:
            return schema.model_validate_json(fixed_content)
        except Exception as fix_error:
            logger.exception("Error validating fixed JSON", error=str(fix_error))
            raise fix_error

    except ValidationError as validation_error:
        logger.exception("Error validating content with schema")
        raise validation_error

    except Exception as e:
        logger.exception("Unexpected error validating content")
        raise e


def _attempt_fix_incomplete_json(content: str) -> str:
    """Attempt to fix incomplete JSON by closing open brackets and quotes.

    Args:
        content: The incomplete JSON string

    Returns:
        The fixed JSON string
    """
    # Count open brackets and quotes
    open_braces = content.count("{") - content.count("}")
    open_brackets = content.count("[") - content.count("]")
    open_quotes = content.count('"') % 2

    # Fix open braces and brackets
    fixed_content = content

    # Handle trailing commas in objects or arrays
    if fixed_content.rstrip().endswith(","):
        fixed_content = fixed_content.rstrip()[:-1]

    # First fix quotes if needed
    if open_quotes > 0:
        # Try to find the last property name or value
        last_quote = content.rfind('"')
        if last_quote > 0:
            # Check if we're in a property name or value
            prev_colon = content.rfind(":", 0, last_quote)
            prev_comma = content.rfind(",", 0, last_quote)

            if prev_colon > prev_comma:  # We're in a value
                fixed_content = content + '"'
            else:  # We're in a property name
                fixed_content = content + '":""'

    # Then close brackets and braces
    if open_brackets > 0:
        fixed_content += "]" * open_brackets

    if open_braces > 0:
        fixed_content += "}" * open_braces

    return fixed_content
