"""Module for common utility functions."""

import os

from dotenv import load_dotenv
from huggingface_hub import login


def login_hf_hub(write_permission: bool = False):
    """Login to the Hugging Face hub. See documentation: https://huggingface.co/docs/huggingface_hub/en/package_reference/authentication."""
    load_dotenv()
    HF_ACCESS_TOKEN = os.getenv("HUGGINGFACE_ACCESS_TOKEN")
    login(
        token=HF_ACCESS_TOKEN,
        add_to_git_credential=True,
        write_permission=write_permission,
    )
