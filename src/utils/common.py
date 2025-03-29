"""Module for common utility functions."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from src.utils.constants import FILE_PROMPT_PLACEHOLDER_DICT
from src.utils.error_handlers import check_file_path

if TYPE_CHECKING:
    from typing import Any, Optional

    from src.utils import PathLike


def read_prompt(
    prompt_path: PathLike,
    vars: Optional[dict[str, Any]] = None,
    vars_json_path: Optional[PathLike] = None,
) -> str:
    """Read the system prompt from a .txt file.

    Args:
        prompt_path (Path): The path to the prompt file.
        vars (dict, optional): A dictionary of variables to replace in the prompt.
        vars_json_path (PathLike, optional): The path to a JSON file containing variables.

    Returns:
        str: The prompt.
    """
    prompt_path = check_file_path(prompt_path, extensions=["txt"])

    with prompt_path.open("r") as file:
        prompt = file.read().strip()

    if vars_json_path:
        vars_from_json = read_config(vars_json_path)

        # If vars is also provided, extend it with the values from the JSON file
        # else use the values from the JSON file as the vars
        if vars:
            vars.update(vars_from_json)
        else:
            vars = vars_from_json

    elif vars_json_path is None and "system" in prompt_path.name:
        vars = read_config(FILE_PROMPT_PLACEHOLDER_DICT)

    if vars:
        return prompt.format(**vars)
    return prompt


def read_config(conf_path: PathLike) -> dict:
    """Read a configuration file.

    Args:
        conf_path (PathLike): The path to the configuration file. Must be a JSON file.

    Returns:
        dict: The configuration.
    """
    # Convert to Path object, ensure the file path exists and has the correct extension
    conf_path_obj = check_file_path(conf_path, extensions=["json"])

    with conf_path_obj.open("r") as file:
        return json.load(file)


def update_config(config_filepath: PathLike, key: str, new_value: str):
    """Update the config file with the new_value for the key.

    Args:
        config_filepath (PathLike): The path to the config file. The file should be in JSON format.
        key (str): The key to update.
        new_value (str): The new value to set for the key.
    """
    config_path = check_file_path(config_filepath, extensions=["json"])
    try:
        config_data: dict = read_config(config_path)
        config_data[key] = new_value
        with config_path.open("w", encoding="utf-8") as f:
            json.dump(config_data, f)
    except Exception as e:
        raise e
