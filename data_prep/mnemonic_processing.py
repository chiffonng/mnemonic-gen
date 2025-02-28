"""Module for processing mnemonic data using OpenAI's API.

Finetuning: https://platform.openai.com/docs/guides/fine-tuning
Files API: https://platform.openai.com/docs/api-reference/files
"""

import csv
import json
import logging
import random
from collections import defaultdict
from typing import TYPE_CHECKING

from dotenv import load_dotenv
from src.openai.openai_ft import finetune_from_config, upload_file_to_openai
from src.utils import check_file_path, read_config, read_prompt, update_config

from openai import OpenAI

if TYPE_CHECKING:
    from pathlib import Path
    from typing import Optional

# Set up and validates the paths
prompt_path = check_file_path("prompts/improve_ft_system.txt")
raw_input_path = check_file_path("data_prep/raw/improved_data.csv", extensions=["csv"])
train_input_path = check_file_path(
    "data_prep/processed/improve_sft_train.jsonl", extensions=["jsonl"], new_ok=True
)
val_input_path = check_file_path(
    "data_prep/processed/improve_sft_val.jsonl", extensions=["jsonl"], new_ok=True
)
config_file_path = check_file_path(
    "data_prep/config/improve_sft.json", extensions=["json"]
)
finetune_model_id_path = check_file_path(
    "data_prep/config/improve_sft_model_id.txt", extensions=["txt"], new_ok=True
)

# Set up logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.FileHandler("logs/mnemonic_processing.log"))
logger.handlers[0].setFormatter(
    logging.Formatter("%(asctime)s - %(levelname)s - %(funcName)s - %(message)s")
)


def prepare_and_split_finetune_data(
    input_data: "Path" = raw_input_path,
    input_prompt: "Path" = prompt_path,
    output_train_jsonl: "Path" = train_input_path,
    output_val_jsonl: "Optional[Path]" = None,
    split_ratio: float = 0.8,
) -> None:
    """Prepare the data (JSONL) for fine-tuning with OpenAI's API and split into training and validation files.

    Args:
        input_data (Path): The path to the input data.
        input_prompt (Path): The path to the system prompt.
        output_train_jsonl (Path): The path to save the training data.
        output_val_jsonl (Path): The path to save the validation data. If None, the validation data will not be saved.
        split_ratio (float): The ratio to split the data.

    Returns:
        None

    Raises:
        ValueError: If the split ratio is invalid.
    """
    if output_val_jsonl is None:
        split_ratio = 1

    if not isinstance(split_ratio, float):
        split_ratio = float(split_ratio)

    if split_ratio < 0 or split_ratio > 1:
        logging.error("Invalid split ratio. Must be between 0 and 1.")
        raise ValueError(f"Invalid split ratio: {split_ratio}. Must be between 0 and 1")

    system_prompt = read_prompt(input_prompt)

    # Read all rows from CSV into a list of dictionaries
    with input_data.open("r", encoding="utf-8") as infile:
        reader = csv.DictReader(infile)
        rows = list(reader)

    num_examples = len(rows)
    if num_examples == 0:
        logging.error("No data found in the CSV file.")
        return

    random.shuffle(rows)

    train_rows = rows[: int(num_examples * split_ratio)]
    val_rows = rows[int(num_examples * split_ratio) :] if split_ratio < 1 else None

    # Write the training and if not None, the validation data to JSONL files.
    with output_train_jsonl.open("w", encoding="utf-8") as train_file:
        for row in train_rows:
            data = {
                "messages": [
                    {"role": "developer", "content": system_prompt},
                    {
                        "role": "user",
                        "content": f"Term: {row['term'].strip()}\nCurrent mnemonic: {row['mnemonic'].strip()}",
                    },
                    {"role": "assistant", "content": row["improved_mnemonic"].strip()},
                ]
            }
            train_file.write(json.dumps(data) + "\n")

    if output_val_jsonl is not None and val_rows is not None:
        with output_val_jsonl.open("w", encoding="utf-8") as val_file:
            for row in val_rows:
                data = {
                    "messages": [
                        {"role": "developer", "content": system_prompt},
                        {
                            "role": "user",
                            "content": f"Term: {row['term'].strip()}\nCurrent mnemonic: {row['mnemonic'].strip()}",
                        },
                        {
                            "role": "assistant",
                            "content": row["improved_mnemonic"].strip(),
                        },
                    ]
                }
                val_file.write(json.dumps(data) + "\n")

    logger.info(
        f"Data prepared ({num_examples} examples) and split into training and validation files.\nTraining data path: {output_train_jsonl}, \nValidation data path: {output_val_jsonl}"
    )


# TODO: Move it to openai_ft.py, or another module.
def validate_finetune_data(input_data: "Path") -> None:
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

        tools = ex.get("tools", None)
        if tools:
            for tool in tools:
                if not isinstance(tool, dict):
                    format_errors["tool_data_type"] += 1
                    continue
                if any(k not in ("type", "function") for k in tool):
                    format_errors["tool_unrecognized_key"] += 1

    if format_errors:
        logger.error("Errors found in formatting fineturning data:")
        for k, v in format_errors.items():
            logger.error(f"{k}: {v}")
    else:
        logger.info(
            "Data is formatted ROUGHLY correctly. Always check the data manually with the OpenAI API reference here: https://platform.openai.com/docs/api-reference/fine-tuning/chat-input."
        )
        logger.info(f"Number of examples: {len(dataset)}")


def upload_finetune_data(
    client: "OpenAI",
    input_path: "Path",
    config_file_path: "Path" = config_file_path,
    file_type: str = "training",  # or "validation"
    use_cache: bool = True,
    to_overwrite: bool = True,
) -> str | None:
    """Upload the fine-tuning data to OpenAI's Files API, reusing a cached file id if available.

    Args:
        client (OpenAI): The OpenAI client object.
        input_path (Path): Path to the JSONL input data.
        config_file_path (Path): Path to the JSON config file.
        file_type (str): The type of file to upload (e.g., "training" or "validation").
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
    logger.debug(f"Before, OpenAI FILES: {client.files.list()}")

    # Attempt to use the cached file id if allowed.
    if use_cache:
        cached_file_id = _get_cached_file_id(
            client, config_file_path, file_type, to_overwrite=False
        )
        if cached_file_id:
            return cached_file_id
    else:
        # If overwriting, delete the cached file id.
        _get_cached_file_id(
            client, config_file_path, file_type, to_overwrite=to_overwrite
        )

    # Upload the file since no valid cache was found.
    new_file_id = upload_file_to_openai(client, input_path)
    if new_file_id:
        update_config(config_file_path, "training_file", new_file_id)
        logger.info(f"Uploaded file {input_path} with new file id: {new_file_id}")
        return new_file_id

    logger.debug(f"After, OpenAI FILES: {client.files.list()}")

    return None


# TODO: Add a function to validate the config file, or update src.utils.common.read_conf to validate the config file.


def _get_cached_file_id(
    client: "OpenAI", config_file_path: "Path", file_type: str, to_overwrite: bool
) -> str | None:
    """Retrieve the cached file ID for the specified file type from the config file.

    For the given file type (e.g. "training_file" or "validation_file"), verify that the cached
    file exists via OpenAI's Files API. If to_overwrite is True, delete the file from the API.

    Args:
        client (OpenAI): The OpenAI client object.
        config_file_path (Path): Path to the JSON config file.
        file_type (str): The key in the config (e.g., "training_file", "validation_file").
        to_overwrite (bool): If True, delete the cached file from the API.

    Returns:
        str | None: The valid cached file id, or None if not found or if deleted.
    """

    def _check_and_handle_file(file_id: str) -> str | None:
        """Helper function to check if the file exists in the API and handle it accordingly."""
        try:
            file_info = client.files.retrieve(file_id)
            if file_info:
                if to_overwrite:
                    client.files.delete(file_id)
                    logger.info(f"Deleted cached file id: {file_id}")
                    return None
                else:
                    logger.info(f"Using cached file id: {file_id}")
                    return file_id
            else:
                logger.info(f"File id {file_id} not found in API.")
                return None
        except Exception as e:
            logger.error(f"Error retrieving file {file_id}: {e}")
            return None

    # Handle file types
    if not isinstance(file_type, str):
        logger.error("Invalid file type. Must be a string.")
        raise TypeError(f"Invalid file_type {type(file_type)}. Must be a string.")
    elif file_type.lower().startswith("train"):
        file_type = "training_file"
    elif file_type.lower().startswith("val"):
        file_type = "validation_file"
    else:
        logger.error("Invalid file_type. Must be 'training' or 'validation'.")
        raise ValueError("Invalid file_type. Must be 'training' or 'validation'.")

    # Read the config file and get the file id
    try:
        config_data = read_config(config_file_path)
        candidate = config_data.get(file_type, "").strip()
        if candidate:
            return _check_and_handle_file(candidate)
        else:
            return None
    except Exception as e:
        logger.error(f"Error reading config file {config_file_path}: {e}")
        raise e


if __name__ == "__main__":
    prepare_and_split_finetune_data()
    validate_finetune_data(train_input_path)
    validate_finetune_data(val_input_path)

    load_dotenv()
    client = OpenAI()
    # upload_finetune_data(client, input_path=train_input_path)
    # update_finetune_data(client, input_path=val_input_path)

    # finetune_from_config(client, config_file_path, finetune_model_id_path)
