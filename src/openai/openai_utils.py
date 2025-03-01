"""Utility functions for OpenAI API interactions."""

import json
import logging
from collections import defaultdict
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path
    from typing import Optional

    from openai import OpenAI

from src.utils import check_file_path, read_config

logger = logging.getLogger(__name__)


def validate_openai_config(input_path: "Path"):
    """Validate the configuration file to be used for fine-tuning or generating completions using OpenAI.

    Args:
        input_path (Path): The path to the input configuration file. The configuration should be in JSON format.

    Raises:
        ValueError: If the input configuration is empty or not a dictionary.
        TypeError: If the input configuration is not a dictionary.
        IndexError: If the input configuration is missing required keys or has unrecognized keys.
    """
    input_path = check_file_path(input_path, extensions=["json"])

    config = read_config(input_path)

    # Check data
    if not config:
        logger.error(f"{input_path} is empty.")
        raise ValueError(f"{input_path} is empty.")
    elif not isinstance(config, dict):
        logger.error("Data cannot be read to a dictionary.")
        raise TypeError(
            f"Data cannot be read to a dictionary. Please review the data from {input_path}"
        )

    # Validate configuration
    required_keys = ["model"]
    optional_keys = [
        "temperature",
        "max_completion_tokens",
        "frequency_penalty",
        "presence_penalty",
        "top_p",
        "stream",
    ]

    missing_keys = [key for key in required_keys if key not in config]
    if missing_keys:
        logger.error(f"Missing required keys: {missing_keys}")
        raise IndexError(f"Missing required keys: {missing_keys}")

    unrecognized_keys = [
        key for key in config if key not in required_keys + optional_keys
    ]
    if unrecognized_keys:
        logger.error(f"Unrecognized keys: {unrecognized_keys}")
        raise IndexError(f"Unrecognized keys: {unrecognized_keys}")

    # Add defaults for missing optional keys
    optional_defaults = {
        "temperature": 0.4,
        "max_completion_tokens": 2048,
        "frequency_penalty": 0,
        "presence_penalty": 0,
        "top_p": 1.0,
        "stream": False,
    }

    for key, default in optional_defaults.items():
        if key not in config:
            config[key] = default

    return config


def validate_openai_file(input_path: "Path"):
    """Validate the data to be uploaded to OpenAI's API. Source code from OpenAI Cookbook: https://cookbook.openai.com/examples/chat_finetuning_data_prep.

    Args:
        input_path (Path): The path to the input data. The data should be in JSONL format.

    Raises:
        ValueError: If the input data is empty or not a list of dictionaries.
        TypeError: If the input data is not a list of dictionaries.
    """
    input_path = check_file_path(input_path, extensions=["jsonl"])

    with input_path.open("r", encoding="utf-8") as f:
        dataset = [json.loads(line) for line in f]

    # Check data
    if not dataset:
        logger.error(f"{input_path} is empty.")
        raise ValueError(f"{input_path} is empty.")
    elif not isinstance(dataset, list):
        logger.error("Data cannot be read to a list.")
        raise TypeError(
            f"Data cannot be read to a list. Please review the data from {input_path}"
        )
    elif not all(isinstance(ex, dict) for ex in dataset):
        logger.error("Data cannot be read as a list of dictionaries.")
        return ValueError(
            f"Data cannot be read as a list of dictionaries. Please review the data from {input_path}"
        )

    # Format error checks
    format_errors: defaultdict = defaultdict(int)

    for ex in dataset:
        if not isinstance(ex, dict):
            format_errors["data_type"] += 1
            continue

        # TODO: Refactor validating messages as a separate function
        messages = ex.get("messages", None)
        if not messages:
            format_errors["missing_messages_list"] += 1
            continue

        for message in messages:
            # Check that 'role' is present and at least one of 'content' or 'tool' is provided.
            if "role" not in message or (
                "content" not in message and "tool" not in message
            ):
                format_errors["message_missing_key"] += 1

            # Validate that all keys in the message are recognized.
            if any(
                k
                not in (
                    "role",
                    "content",
                    "name",
                    "weight",
                    "refusal",
                    "audio",
                    "tool_calls",
                    "tool_call_id",
                )
                for k in message
            ):
                format_errors["message_unrecognized_key"] += 1
            if message.get("role", None) not in (
                "developer",
                "system",
                "user",
                "assistant",
                "tool",
            ):
                format_errors["unrecognized_role"] += 1

            # Check that either 'content' or 'tool' exists and that 'content', if present, is a string.
            content = message.get("content")
            tool = message.get("tool")
            if (content is None and tool is None) or (
                content is not None and not isinstance(content, str)
            ):
                format_errors["missing_content"] += 1

        # Each example should have at least one assistant message.
        if not any(msg.get("role") == "assistant" for msg in messages):
            format_errors["example_missing_assistant_message"] += 1
        # END TODO

        tools = ex.get("tools", None)
        if tools:
            for tool in tools:
                if not isinstance(tool, dict):
                    format_errors["tool_data_type"] += 1
                    continue
                if any(k not in ("type", "function") for k in tool):
                    format_errors["tool_unrecognized_key"] += 1

    if format_errors:
        logger.error(f"Errors found in formatting data from {input_path}:")
        for k, v in format_errors.items():
            logger.error(f"{k}: {v}")
    else:
        logger.info(
            f"Data from {input_path} is formatted ROUGHLY correctly. Always check the data manually with the OpenAI API reference here: https://platform.openai.com/docs/api-reference/fine-tuning/chat-input."
        )
        logger.info(f"Number of examples: {len(dataset)}")


def upload_file_to_openai(client: "OpenAI", input_path: "Path") -> "Optional[str]":
    """Upload the input file to OpenAI's Files API.

    Args:
        client (OpenAI): The OpenAI client object.
        input_path (Path): The path to the input file.

    Returns:
        Optional[str]: The id of the uploaded file.

    Raises:
        e: Exception if there was an error uploading the file.
    """
    try:
        with input_path.open("rb") as file_bin:
            logger.info(f"Uploading file: {input_path}")
            logger.info(f"Type of file_bin: {type(file_bin)}")
            file_obj = client.files.create(file=file_bin, purpose="fine-tune")
        if file_obj is None:
            logger.error("Error uploading file: received None as file object.")
            raise Exception("Error uploading file: received None as file object.")
        if getattr(file_obj, "status", None) == "failed":
            logger.error(f"Upload failed: {file_obj.error}")
            raise Exception(f"Upload failed: {file_obj.error}")

        return file_obj.id

    except Exception as e:
        logger.error(f"Error during file upload: {e}")
        raise e
