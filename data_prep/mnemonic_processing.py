"""Module for processing mnemonic data using OpenAI's API.

Finetuning: https://platform.openai.com/docs/guides/fine-tuning
Files API: https://platform.openai.com/docs/api-reference/files
"""

# import asyncio
import csv
import json
import logging
from collections import defaultdict
from typing import TYPE_CHECKING

from dotenv import load_dotenv
from openai import OpenAI
from src.utils.common import read_conf
from src.utils.error_handling import check_file_path

if TYPE_CHECKING:
    from pathlib import Path

# Set up and validates the paths
prompt_path = check_file_path("data_prep/prompts/improve_sft_system.txt")
raw_input_path = check_file_path(
    "data_prep/raw/improved_mnemonics.csv", extensions=["csv"]
)
input_path = check_file_path(
    "data_prep/processed/improved_mnemonics.jsonl", extensions=["jsonl"], new_ok=True
)
config_file_path = check_file_path(
    "data_prep/config/improve_sft.json", extensions=["json"]
)

# Set up logging to console
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler())
logger.handlers[0].setFormatter(
    logging.Formatter("%(levelname)s - %(funcName)s - %(message)s")
)


def read_system_prompt(prompt_path: "Path") -> str:
    """Read the system prompt from a file.

    Args:
        prompt_path (Path): The path to the system prompt file.

    Returns:
        str: The system prompt.
    """
    with prompt_path.open("r") as file:
        system_prompt = file.read().strip()
    return system_prompt


def prepare_finetune_data(
    input_data: "Path" = raw_input_path,
    input_prompt: "Path" = prompt_path,
    output_jsonl: "Path" = input_path,
) -> None:
    """Prepare the data (JSONL) for fine-tuning with OpenAI's API.

    Args:
        input_data (Path): The path to the input data.
        input_prompt (Path): The path to the system prompt.
        output_jsonl (Path): The path to save the output data.

    Returns:
        None
    """
    system_prompt = read_system_prompt(input_prompt)

    num_examples = 0
    with (
        input_data.open("r", encoding="utf-8") as input_file,
        output_jsonl.open("w", encoding="utf-8") as output_file,
    ):
        reader = csv.DictReader(input_file)
        for row in reader:
            term = row["term"].strip()
            mnemonic = row["mnemonic"].strip()
            improved_mnemonic = row["improved_mnemonic"].strip()

            # Build the JSONL object
            data = {
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": f"Term: {term}\nCurrent mnemonic: {mnemonic}",
                    },
                    {"role": "assistant", "content": improved_mnemonic},
                ]
            }
            output_file.write(json.dumps(data) + "\n")
            num_examples += 1

    logger.info(f"{num_examples} examples have been saved to {output_jsonl}")


def validate_finetune_data(input_data: "Path" = input_path) -> None:
    """Validate the data for fine-tuning with OpenAI's API. Source code from OpenAI Cookbook: https://cookbook.openai.com/examples/chat_finetuning_data_prep.

    Args:
        input_data (Path): The path to the input data. The data should be in JSONL format.

    Returns:
        None
    """
    input_data = check_file_path(input_data, extensions=["jsonl"])

    with input_data.open("r", encoding="utf-8") as f:
        dataset = [json.loads(line) for line in f]

    # Format error checks
    format_errors: defaultdict = defaultdict(int)

    for ex in dataset:
        if not isinstance(ex, dict):
            format_errors["data_type"] += 1
            continue

        messages = ex.get("messages", None)
        if not messages:
            format_errors["missing_messages_list"] += 1
            continue

        for message in messages:
            if "role" not in message or "content" not in message:
                format_errors["message_missing_key"] += 1

            if any(
                k not in ("role", "content", "name", "function_call", "weight")
                for k in message
            ):
                format_errors["message_unrecognized_key"] += 1

            if message.get("role", None) not in (
                "system",
                "user",
                "assistant",
                "function",
            ):
                format_errors["unrecognized_role"] += 1

            content = message.get("content", None)
            function_call = message.get("function_call", None)

            if (not content and not function_call) or not isinstance(content, str):
                format_errors["missing_content"] += 1

        if not any(message.get("role", None) == "assistant" for message in messages):
            format_errors["example_missing_assistant_message"] += 1

    if format_errors:
        logger.error("Errors found in formatting fineturning data:")
        for k, v in format_errors.items():
            logger.error(f"{k}: {v}")
    else:
        logger.info("Data is formatted correctly.")
        logger.info(f"Number of examples: {len(dataset)}")


def upload_finetune_data(
    client: "OpenAI",
    input_path: "Path" = input_path,
    config_file_path: "Path" = config_file_path,
    use_cache: bool = True,
    to_overwrite: bool = True,
) -> str | None:
    """Upload the fine-tuning data to OpenAI's Files API, reusing a cached file id if available.

    Args:
        client (OpenAI): The OpenAI client object.
        input_path (Path): Path to the JSONL input data.
        config_file_path (Path): Path to the JSON config file.
        use_cache (bool): If True, reuse the cached file id from config.
        to_overwrite (bool): If True, delete the cached file from OpenAI and reupload. Only relevant if use_cache is False.

    Returns:
        str: The file id of the uploaded file, or None if there was an error.
    """
    # Ensure input_path is valid (expects a .jsonl file)
    input_path = check_file_path(input_path, extensions=["jsonl"])

    # Log ALL the argument values
    logger.info(f"input_path: {input_path}, config_file_path: {config_file_path}")
    logger.info(f"use_cache: {use_cache}, to_overwrite: {to_overwrite}")

    # Attempt to use the cached file id if allowed.
    if use_cache:
        cached_file_id = _get_cached_file_id(
            client, config_file_path, to_overwrite=False
        )
        if cached_file_id:
            return cached_file_id
    else:
        # If overwriting, delete the cached file id.
        _get_cached_file_id(client, config_file_path, to_overwrite=to_overwrite)

    # Upload the file since no valid cache was found.
    new_file_id = _upload_file(client, input_path)
    if new_file_id:
        _update_config_file(config_file_path, new_file_id)
        logger.info(f"Uploaded file {input_path} with new file id: {new_file_id}")
        return new_file_id

    return None


def fine_tune(
    input_file: "Path" = input_path,
    config_file: "Path" = config_file_path,
):
    """Fine tune an OpenAI model with the given data.

    Args:
        input_file (PathLike): The path to the input data. Must be of JSONL format.
        config_file (PathLike): The path to the configuration file. Must be of JSON format.
    """
    config_data: dict = read_conf(config_file)
    config_data.get("model_id")
    config_data.get("training_file")
    config_data.get("validation_file")


def _get_cached_file_id(
    client: "OpenAI", config_file_path: "Path", to_overwrite: bool
) -> str | None:
    """Check for a cached file id in the config file. If found and not overwriting, verify it exists in OpenAI's Files API. If overwriting, attempt to delete it.

    Args:
        client (OpenAI): The OpenAI client object.
        config_file_path (Path): The path to the config file.
        to_overwrite (bool): If True, delete the cached file from OpenAI.

    Returns:
        str: The file id of the cached file, or None if there was an error.
    """
    try:
        config_data = read_conf(config_file_path)

        cached_file_id = config_data.get("training_file")

        cached_file_info = client.files.retrieve(cached_file_id)

        # 1. If we're using the cached file id and it exists, return it.
        # 2. If we're overwriting, delete the file
        # 3. If we're using the cached file id and it doesn't exist, return None.
        if cached_file_info and not to_overwrite:
            logger.info(f"Using cached file id: {cached_file_id}")
            return cached_file_id
        elif cached_file_info and to_overwrite:
            client.files.delete(cached_file_id)
            logger.info(f"Deleted cached file id: {cached_file_id}")
        else:
            logger.info("No cached file id found.")
    except Exception as e:
        logger.error(f"Error reading config file {config_file_path}: {e}")

    return None


def _upload_file(client: "OpenAI", input_path: "Path") -> str | None:
    """Upload the input file to OpenAI's Files API.

    Args:
        client (OpenAI): The OpenAI client object.
        input_path (Path): The path to the input file.

    Returns:
        str: The file id of the uploaded file, or None if there was an error.
    """
    try:
        with input_path.open("rb") as file_bin:
            file_obj = client.files.create(file=file_bin, purpose="fine-tune")
        if file_obj is None:
            logger.error("Error uploading file: received None as file object.")
            return None
        if getattr(file_obj, "status", None) == "failed":
            logger.error(f"Upload failed: {file_obj.error}")
            return None
        return file_obj.id
    except Exception as e:
        logger.error(f"Error during file upload: {e}")
        return None


def _update_config_file(config_file_path: "Path", file_id: str):
    """Update the config file with the new file id. Keep other config data intact.

    Args:
        config_file_path (Path): The path to the config file.
        file_id (str): The new file id to store in the config file.
    """
    try:
        with config_file_path.open("w", encoding="utf-8") as f:
            config_data: dict = read_conf(config_file_path)
            config_data["training_file"] = file_id
            json.dump(config_data, f)
    except Exception as e:
        logger.error(f"Error updating config file {config_file_path}: {e}")


if __name__ == "__main__":
    # prepare_finetune_data()
    # validate_finetune_data()
    load_dotenv()
    client = OpenAI()
    upload_finetune_data(client, input_path)
    logger.debug(f"OpenAI Files API has the following files: {client.files.list()}")
    # fine_tune()
