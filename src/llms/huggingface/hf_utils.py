"""Module for common utility functions for interacting with HuggingFace."""


def login_hf_hub():
    """Login to the Hugging Face hub. See documentation: https://huggingface.co/docs/huggingface_hub/en/package_reference/authentication."""
    from huggingface_hub import login

    hf_access_token = get_hf_token()
    login(
        token=hf_access_token,
        add_to_git_credential=True,
    )


def get_hf_token() -> str:
    """Get the Hugging Face token from the environment."""
    import os

    from dotenv import load_dotenv

    load_dotenv()
    return os.getenv("HF_TOKEN")
