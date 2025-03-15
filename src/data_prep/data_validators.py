"""Module for data validation functions."""

from enum import Enum
from typing import Callable, TypeVar, cast


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


T = TypeVar("T", bound=str | None)


def validate_enum_field(enum_class: type[Enum]) -> Callable[[T], T]:
    """Create a validator for enum fields."""

    def validator(value: T) -> T:
        if value is None:
            return value

        if isinstance(value, str) and value.strip():
            try:
                # Try to get enum value
                return cast(T, enum_class(value.strip().lower()))
            except ValueError:
                # If value doesn't match enum, use the _missing_ method
                if hasattr(enum_class, "_missing_"):
                    return cast(T, enum_class._missing_(value))
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


# TODO: validate a dictionary with schemas
