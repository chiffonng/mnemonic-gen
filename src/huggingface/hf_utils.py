"""Module for common utility functions for interacting with HuggingFace."""


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
