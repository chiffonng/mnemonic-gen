"""Module for exporting data to different formats and pushing to Hugging Face hub."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from src.utils import check_file_path

if TYPE_CHECKING:
    from typing import Any

    from src.utils import PathLike


def write_jsonl_file(data: list[dict[str, Any]], file_path: PathLike) -> None:
    """Write a list of dictionaries to a JSONL file.

    Args:
        data: List of dictionaries to write
        file_path: Path to the output JSONL file
    """
    file_path = check_file_path(
        file_path, new_ok=True, to_create=True, extensions=[".jsonl"]
    )

    with file_path.open("w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
