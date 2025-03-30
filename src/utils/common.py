"""Module for common utility functions."""

from __future__ import annotations

import json
import random
from typing import TYPE_CHECKING

import yaml
from structlog import getLogger

from src.utils.constants import PROMPT_FILES
from src.utils.error_handlers import check_file_path

if TYPE_CHECKING:
    from typing import Any, Optional

    from src.utils import PathLike

logger = getLogger(__name__)


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
        vars = read_config(PROMPT_FILES["PLACEHOLDER_DICT"])

    if vars:
        return prompt.format(**vars)
    return prompt


def sample_prompt(prompt_path: PathLike, num_samples: int = 1) -> str | list[str]:
    """Read a random instruction from a .txt file.

    Args:
        prompt_path (PathLike): The path to the file (.txt) with prompts
        num_samples (int): The number of random prompts to return. Default is 1.

    Returns:
        str: A random instruction from the file.
    """
    prompt_path = check_file_path(prompt_path, extensions=[".txt"])

    with prompt_path.open("r") as file:
        prompts = file.readlines()

    # Strip whitespace and remove empty lines
    prompts = [line.strip() for line in prompts if line.strip()]

    if not prompts:
        raise ValueError(f"No valid prompts found in {prompt_path}")

    if num_samples == 1:
        chosen_prompt = random.choice(prompts)
        logger.debug(
            "Sampling a single prompt", source=prompt_path, prompt=chosen_prompt
        )
        return chosen_prompt
    else:
        if num_samples > len(prompts):
            raise ValueError(
                f"Requested {num_samples} samples, but only {len(prompts)} available."
            )

        chosen_prompts = random.sample(prompts, num_samples)
        logger.debug(
            "Sampling multiple prompts",
            source=prompt_path,
            prompts=chosen_prompts,
        )
        return chosen_prompts


def read_config(conf_path: PathLike) -> dict:
    """Read a configuration file.

    Args:
        conf_path (PathLike): The path to the configuration file. Must be a JSON file.

    Returns:
        dict: The configuration.
    """
    # Convert to Path object, ensure the file path exists and has the correct extension
    conf_path_obj = check_file_path(conf_path, extensions=[".json", ".yaml", ".yml"])

    if conf_path_obj.suffix == ".json":
        with conf_path_obj.open("r") as file:
            return json.load(file)
    elif conf_path_obj.suffix in [".yaml", ".yml"]:
        with conf_path_obj.open("r") as file:
            return yaml.safe_load(file) or {}


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
