"""Module for common utility functions for interacting with HuggingFace."""


def login_hf_hub(**kwargs):
    """Login to the Hugging Face hub. See documentation: https://huggingface.co/docs/huggingface_hub/en/package_reference/authentication.

    Args:
        **kwargs: Additional keyword arguments to pass to the `login` function from the `huggingface_hub` package.
    """
    from huggingface_hub import login

    login(token=get_hf_token(), add_to_git_credential=True, **kwargs)


def get_hf_token() -> str:
    """Get the Hugging Face token from the environment."""
    import os

    from dotenv import load_dotenv

    load_dotenv()
    return os.getenv("HF_TOKEN")
