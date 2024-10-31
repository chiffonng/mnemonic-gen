"""Module of type aliases for the project."""

from pathlib import Path
from typing import TypeAlias

# Type aliases
PathLike: TypeAlias = str | Path
ExtensionsType: TypeAlias = list[str] | str | None
