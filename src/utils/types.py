"""Module of type aliases for the project."""

from pathlib import Path
from typing import Any, TypeAlias, TypeVar

from pydantic import BaseModel

PathLike: TypeAlias = str | Path
ExtensionsType: TypeAlias = list[str] | str | None
StrNoneType = TypeVar("StrNoneType", str, None)  # str or None

ModelT = TypeVar("ModelT", bound=BaseModel)  # subclass of BaseModel
ResponseType: TypeAlias = str | ModelT | dict[str, Any]
BatchResponseType: TypeAlias = list[ResponseType]
