"""Module for handling common errors and exceptions."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Optional

    from src.utils.aliases import ExtensionsType, PathLike


def validate_path(path: PathLike) -> Path:
    """Validate path and convert it to a Path object if it is a string.

    Args:
        path (PathLike): The path to validate.

    Returns:
        path (Path): The validated path as a Path object.

    Raises:
        TypeError: If 'path' is not a string or a Path object.
    """
    if not isinstance(path, (Path, str)):
        raise TypeError(
            f"{path} must be a pathlib.Path object or a string. Got {type(path)} instead."
        )
    return Path(path) if isinstance(path, str) else path


def validate_and_normalize_extensions(
    extensions: ExtensionsType,
) -> list[str]:
    """Normalize extensions to a list format, ensuring each starts with a dot.

    Args:
        extensions (str | list[str] | None): A string or a list of strings representing file extensions.

    Returns:
        extensions (list[str]): A list of file extensions, each starting with a dot; or an empty list if 'extensions' is None.

    Raises:
        TypeError: If 'extensions' is not a string or a list of strings.
    """
    if extensions is None:
        return []

    if isinstance(extensions, str):
        extensions = [extensions]
    elif not all(isinstance(ext, str) for ext in extensions):
        raise TypeError(
            f"Extensions must be a string or a list of strings. Got {type(extensions)} instead."
        )

    # Ensure extensions start with a dot
    extensions = [ext if ext.startswith(".") else f".{ext}" for ext in extensions]
    return extensions


def check_extension(path: Path, extensions: ExtensionsType) -> None:
    """Check if the path has one of the allowed extensions."""
    if extensions and path.suffix not in extensions:
        raise ValueError(
            f"File '{path.name}' must have one of the following extensions: {extensions}. Current extension is '{path.suffix}'."
        )


# TODO: Refactor into two functions, one for file, and one for checking extensions
def check_file_path(
    path: PathLike,
    new_ok: bool = False,
    to_create: bool = False,
    extensions: Optional[ExtensionsType] = None,
) -> Path:
    """Convert path to a Path object if it is a string, and return it. Optionally, check if the file has one of the specified extensions or if it exists.

    Args:
        path (PathLike): The path to the file.
        new_ok (bool, optional): If True, the file does not have to exist. Defaults to False.
        to_create (bool, optional): If True, the file will be created if it does not exist. Defaults to False. Ignored if 'new_ok' is False.
        extensions (list[str], optional): A list of allowed file extensions. Defaults to []. If provided, the file must have one of the specified extensions.

    Returns:
        path (Path): The path to the file.

    Raises:
        TypeError: If 'path' is not a string or a Path object OR if 'extensions' is not a string or a list of strings.
        FileNotFoundError: If the file does not exist.
        ValueError: If the extensions do not start with a dot OR if the file does not have the specified extension
    """
    path = validate_path(path)
    if not new_ok and not path.exists():
        raise FileNotFoundError(f"{path.resolve()}")
    elif new_ok and to_create:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()

    extensions = validate_and_normalize_extensions(extensions)
    check_extension(path, extensions)

    return path


# TODO: Refactor into two functions, one for file, and one for checking extensions
def check_dir_path(
    dir_path: PathLike,
    new_ok: bool = False,
    extensions: Optional[ExtensionsType] = None,
) -> Path | list[Path]:
    """Check if the directory path exists, convert it to a Path object if it is a string, and return it. Optionally, check if the directory contains files with the specified extensions.

    Args:
        dir_path (PathLike): The path to the directory.
        new_ok (bool, optional): If True, the directory does not have to exist. Defaults to False.
        extensions (list[str], optional): A list of allowed file extensions. Defaults to [].

    Returns:
        dir_path (Path or list[Path]): The path to the directory. If extensions are provided, returns a list of file paths with the specified extensions.

    Raises:
        TypeError: If 'dir_path' is not a string or a Path object OR if 'extensions' is not a string or a list of strings.
        FileNotFoundError: If the directory does not exist, OR if "extensions" are provided but and no files with the specified extensions are found.
        ValueError: If the extensions do not start with a dot.
    """
    dir_path = validate_path(dir_path)
    if not new_ok and (not dir_path.exists() or not dir_path.is_dir()):
        raise FileNotFoundError(f"Directory not found: {dir_path.resolve()}")

    extensions = validate_and_normalize_extensions(extensions)

    # Collect files with the specified extensions if provided
    if extensions:
        paths = [p for p in dir_path.rglob("*") if p.suffix in extensions]
        if not paths:
            raise FileNotFoundError(
                f"No files with one of the extensions {extensions} found in directory: {dir_path.resolve()}"
            )
        return paths

    return dir_path


def which_file_exists(
    *files: PathLike, extensions: Optional[ExtensionsType] = None
) -> Path:
    """Return the first file found in the list of files. Optionally, return the first file with the specified extensions.

    Args:
        files (PathLike): The list of file paths to check.
        extensions (list[str], optional): A list of allowed file extensions. Defaults to [].

    Returns:
        file_path (Path): The first file found in the list.
    """
    for file in files:
        file_path = check_file_path(file, new_ok=True, extensions=extensions)
        if file_path.exists():
            return file_path

    raise FileNotFoundError(
        f"None of the specified files were found: {[str(p) for p in files]}."
    )
