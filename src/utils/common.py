"""Module for common utility functions."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.utils.aliases import PathLike


def login_hf_hub(write_permission: bool = False):
    """Login to the Hugging Face hub. See documentation: https://huggingface.co/docs/huggingface_hub/en/package_reference/authentication.

    Args:
        write_permission (bool, optional): Whether to add write permission. Defaults to False.
    """
    from huggingface_hub import login

    HF_ACCESS_TOKEN = get_hf_token()
    login(
        token=HF_ACCESS_TOKEN,
        add_to_git_credential=True,
        write_permission=write_permission,
    )


def get_hf_token():
    """Get the Hugging Face token from the environment."""
    import os

    from dotenv import load_dotenv

    load_dotenv()
    return os.getenv("HF_TOKEN")


def read_conf(conf_path: "PathLike") -> dict:
    """Read a configuration file.

    Args:
        conf_path (PathLike): The path to the configuration file. Must be a JSON file.

    Returns:
        dict: The configuration.
    """
    import json

    from src.utils.error_handling import check_file_path

    # Convert to Path object, ensure the file path exists and has the correct extension
    conf_path_obj = check_file_path(conf_path, extensions=["json"])

    with conf_path_obj.open("r") as file:
        return json.load(file)
