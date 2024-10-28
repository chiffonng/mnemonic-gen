"""Module for handling common errors and exceptions."""

from pathlib import Path


def check_file_path(
    path: Path | str, new_ok: bool = False, extensions: list[str] = None
) -> Path:
    """Convert path to a Path object if it is a string, and return it. Optionally, check if the file has one of the specified extensions or if it exists.

    Args:
        path (Path | str): The path to the file.
        new_ok (bool, optional): If True, the file does not have to exist. Defaults to False.
        extensions (list[str], optional): A list of allowed file extensions. Defaults to [].

    Returns:
        path (Path): The path to the file.
    """
    if not isinstance(path, Path) and not isinstance(path, str):
        raise TypeError("Parameter 'path' must be a pathlib.Path object or a string.")

    if isinstance(path, str):
        path = Path(path)

    if not new_ok and not path.exists():
        raise FileNotFoundError(f"{path.resolve()}")

    if extensions and path.suffix not in extensions:
        raise ValueError(
            f"File must have one of the following extensions: {extensions}"
        )

    return path


def check_dir_path(
    dir_path: Path | str, extensions: list[str] = None
) -> Path | list[Path]:
    """Check if the directory path exists, convert it to a Path object it, and return it. Optionally, check if the directory contains files with the specified extensions.

    Args:
        dir_path (Path | str): The path to the directory.
        extensions (list[str], optional): A list of allowed file extensions. Defaults to [].

    Returns:
        dir_path (Path): The path to the directory. If extensions are provided, returns a list of file paths with the specified extensions.
    """
    if not isinstance(dir_path, Path) and not isinstance(dir_path, str):
        raise TypeError("Directory path must be a pathlib.Path object or a string.")

    if isinstance(dir_path, str):
        dir_path = Path(dir_path)

    if not dir_path.exists():
        raise FileNotFoundError(f"Directory not found: {dir_path.resolve()}")

    if extensions:
        paths = [p for p in dir_path.rglob("*") if p.suffix in extensions]
        if not paths:
            raise FileNotFoundError(
                f"No files with one of the extensions {extensions} found in directory: {dir_path.resolve()}"
            )
        return paths

    return dir_path


def which_file_exists(
    *files: list[Path] | list[str], extensions: list[str] = None
) -> Path:
    """Return the first file found in the list of files. Optionally, return the first file with the specified extensions.

    Args:
        files (list[Path] | list[str]): The list of files to check.
        extensions (list[str], optional): A list of allowed file extensions. Defaults to [].

    Returns:
        file (Path): The first file found in the list.
    """
    for file in files:
        file = check_file_path(file, new_ok=True, extensions=extensions)
        if file.exists():
            return file

    raise FileNotFoundError(
        f"None of the specified files were found: {[str(p) for p in files]}."
    )
