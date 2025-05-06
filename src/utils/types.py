"""Module of type aliases for the project."""

from pathlib import Path
from typing import Any, TypeAlias, TypeVar, Union

import pandas as pd
from pydantic import BaseModel

PathLike: TypeAlias = str | Path
ExtensionsType: TypeAlias = list[str] | str
ExampleList: TypeAlias = list[dict[str, Any]]

ModelType = TypeVar("ModelType", bound=BaseModel)  # subclass of BaseModel
DataType: TypeAlias = Union[
    pd.DataFrame, dict[str, dict[str, Any]], list[dict[str, Any]], str
]
