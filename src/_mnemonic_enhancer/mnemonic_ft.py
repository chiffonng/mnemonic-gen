"""Module for processing mnemonic data using OpenAI's API.

Finetuning: https://platform.openai.com/docs/guides/fine-tuning
Files API: https://platform.openai.com/docs/api-reference/files
"""

from __future__ import annotations

import csv
import json
import random
from typing import TYPE_CHECKING

import structlog
from dotenv import load_dotenv
from openai import OpenAI

from src import const
from src.llms.openai import (
    finetune_from_config,
    upload_file_to_openai,
    validate_openai_file,
)
from src.utils import check_file_path, read_config, read_prompt, update_config

if TYPE_CHECKING:
    from pathlib import Path
    from typing import Optional

    from structlog.stdlib import BoundLogger

# Set up and validates the paths
prompt_path = check_file_path(
    const.FILE_PROMPT_IMPROVE_SFT_SYSTEM, extensions=const.TXT_EXT
)
raw_input_path = check_file_path(const.SEED_IMPROVED_CSV, extensions=const.CSV_EXT)
train_input_path = check_file_path(
    const.SFT_IMPROVE_TRAIN, extensions=const.JSONL_EXT, new_ok=True
)
val_input_path = check_file_path(
    const.SFT_IMPROVE_VAL, extensions=const.JSONL_EXT, new_ok=True
)
config_file_path = check_file_path(const.CONF_OPENAI_SFT, extensions=const.JSON_EXT)
finetune_model_id_path = check_file_path(
    "out/improve_sft_model_id.txt", extensions=["txt"], new_ok=True
)

# Set up logging
logger: BoundLogger = structlog.getLogger(__name__)


load_dotenv()
client = OpenAI()


def prepare_and_split_finetune_data(
    input_path: Path = raw_input_path,
    input_prompt: Path = prompt_path,
    output_train_jsonl: Path = train_input_path,
    output_val_jsonl: Optional[Path] = None,
    split_ratio: float = 0.8,
) -> None:
    """Prepare the data (JSONL) for fine-tuning with OpenAI's API and split into training and validation files.

    Args:
        input_path (Path): The path to the input data.
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
        logger.info(
            "There is no validation data path provided. Validation data will not be saved."
        )
        split_ratio = 1

    if not isinstance(split_ratio, float):
        split_ratio = float(split_ratio)

    if split_ratio < 0 or split_ratio > 1:
        logger.error("Invalid split ratio. Must be between 0 and 1.")
        raise ValueError(f"Invalid split ratio: {split_ratio}. Must be between 0 and 1")

    system_prompt = read_prompt(input_prompt)

    # Read all rows from CSV into a list of dictionaries
    with input_path.open("r", encoding="utf-8") as infile:
        reader = csv.DictReader(infile)
        rows = list(reader)

    num_examples = len(rows)
    if num_examples == 0:
        logger.error("No data found in the CSV file.")
        return

    random.shuffle(rows)

    train_rows = rows[: int(num_examples * split_ratio)]
    val_rows = rows[int(num_examples * split_ratio) :] if split_ratio < 1 else None

    # Write the training and if not None, the validation data to JSONL files.
    logger.debug("Processing training data...")
    with output_train_jsonl.open("w", encoding="utf-8") as train_file:
        for row in train_rows:
            data = {
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": f"Term: {row['term'].strip()}\nCurrent mnemonic: {row['mnemonic'].strip()}",
                    },
                    {"role": "assistant", "content": row["improved_mnemonic"].strip()},
                ]
            }
            train_file.write(json.dumps(data) + "\n")

    if output_val_jsonl is not None and val_rows is not None:
        logger.debug("Processing validation data...")
        with output_val_jsonl.open("w", encoding="utf-8") as val_file:
            for row in val_rows:
                data = {
                    "messages": [
                        {"role": "system", "content": system_prompt},
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


def upload_finetune_data(
    client: OpenAI,
    input_path: Path,
    config_file_path: Path = config_file_path,
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


def _get_cached_file_id(
    client: OpenAI, config_file_path: Path, file_type: str, to_overwrite: bool
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


def run_finetune_pipeline():
    """Run the full pipeline for fine-tuning a model with improved mnemonics."""
    prepare_and_split_finetune_data(
        output_train_jsonl=train_input_path,
        output_val_jsonl=val_input_path,
    )
    validate_openai_file(train_input_path)
    validate_openai_file(val_input_path)

    upload_finetune_data(client, input_path=train_input_path, file_type="train")
    upload_finetune_data(client, input_path=val_input_path, file_type="validation")

    finetune_from_config(client, config_file_path, finetune_model_id_path)
