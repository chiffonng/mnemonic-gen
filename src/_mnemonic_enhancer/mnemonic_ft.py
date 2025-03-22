"""Module for processing mnemonic data using OpenAI's API.

Finetuning: https://platform.openai.com/docs/guides/fine-tuning
Files API: https://platform.openai.com/docs/api-reference/files
"""

from __future__ import annotations

import random
from typing import TYPE_CHECKING

import pandas as pd
from pydantic import ValidationError
from structlog import getLogger

from src._data_prep.data_exporters import write_jsonl_file
from src.data.data_validators import validate_content_against_schema
from src.llms.openai import (
    upload_file_to_openai,
    validate_openai_file,
)
from src.utils import constants as const
from src.utils.common import read_config, read_prompt, update_config
from src.utils.error_handlers import check_file_path, check_file_paths

if TYPE_CHECKING:
    from typing import Any, Literal, Optional

    from openai import OpenAI
    from structlog.stdlib import BoundLogger

    from src.utils.types import ModelT, PathLike

# Set up logging
logger: BoundLogger = getLogger(__name__)


def prepare_split_finetune_data(
    input_path: PathLike,
    system_prompt_path: PathLike,
    user_prompt_path: PathLike,
    output_train_jsonl: PathLike,
    output_val_jsonl: Optional[PathLike] = None,
    validation_schema: Optional[ModelT] = None,
    split_ratio: float = 0.8,
    seed: Optional[int] = 42,
) -> None:
    """Prepare the data (JSONL) for fine-tuning with OpenAI's API and split into training and validation files.

    The output file is in JSONL format, where each line is a JSON object with the following structure:
       {"messages": [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}, {"role": "assistant", "content": assistant_response}]}

    Args:
        input_path (Path): The path to the input data.
        system_prompt_path (Path): The path to the system prompt file.
        user_prompt_path (Path): The path to the user prompt file.
        output_train_jsonl (Path): The path to save the training data.
        output_val_jsonl (Path): The path to save the validation data. If None, the validation data will not be saved.
        validation_schema (ModelT, optional): The schema to validate the data against. Defaults to None.
        split_ratio (float): The ratio to split the data.
        seed (int, optional): The random seed for shuffling the data. Defaults to 42.

    Returns:
         None

    Raises:
         ValueError: If the split ratio is invalid.
    """
    # Validate paths
    input_path = check_file_path(input_path, extensions=const.CSV_EXT)
    system_prompt_path, user_prompt_path = check_file_paths(
        system_prompt_path, user_prompt_path, extensions=const.TXT_EXT
    )
    output_train_jsonl = check_file_path(
        output_train_jsonl, new_ok=True, to_create=True, extensions=const.JSONL_EXT
    )

    if output_val_jsonl is None:
        logger.info(
            "There is no validation data path provided. Validation data will not be saved."
        )
    else:
        output_val_jsonl = check_file_path(
            output_val_jsonl, extensions=const.JSONL_EXT, new_ok=True, to_create=True
        )

    # Read prompts
    try:
        system_prompt = read_prompt(system_prompt_path)
        logger.debug(f"Read system prompt ({len(system_prompt)} chars)")
    except Exception as e:
        logger.exception("Error reading system prompt")
        raise ValueError(f"Error reading system prompt: {e}") from e
    try:
        user_prompt_template = read_prompt(user_prompt_path)
        logger.debug(f"Read user prompt template ({len(user_prompt_template)} chars)")
    except Exception as e:
        logger.exception("Error reading user prompt template")
        raise ValueError(f"Error reading user prompt template: {e}") from e

    # Read CSV file, convert to JSONL, and validate schema
    try:
        df = pd.read_csv(input_path)
        rows = df.to_dict(orient="records")
        logger.debug(f"Read {len(rows)} rows from {input_path} using pandas")

        if not rows:
            raise ValueError(f"No data found in {input_path}")

        # Validate schema
        if validation_schema:
            for row in rows:
                row = validate_content_against_schema(row, schema=validation_schema)

        # Split data if needed
        if output_val_jsonl:
            train_rows, val_rows = _split_data(rows, split_ratio, seed=seed)
            num_train = len(train_rows)
            num_val = len(val_rows)
            logger.info(
                "Split data into training and validation sets",
                num_train=num_train,
                num_val=num_val,
            )
            write_jsonl_file(data=train_rows, file_path=output_train_jsonl)
            write_jsonl_file(data=val_rows, file_path=output_val_jsonl)
            validate_openai_file(output_train_jsonl)
            validate_openai_file(output_val_jsonl)
            logger.info(
                "Validation data saved to JSONL files",
                val_file=output_val_jsonl,
                num_val=num_val,
            )
        else:
            train_rows = rows
            logger.info(
                "No validation data path provided. Using all data for training."
            )
            write_jsonl_file(data=train_rows, file_path=output_train_jsonl)
            validate_openai_file(output_train_jsonl)
            logger.info(
                "Training data saved to JSONL file",
                train_file=output_train_jsonl,
                num_train=len(train_rows),
            )

        return num_train, num_val

    except ValidationError:
        logger.exception("Validation error while processing content file")

    except Exception as e:
        logger.exception("Error processing content file")
        raise ValueError(f"Error processing content file: {e}") from e


def _split_data(
    data: list[dict[str, Any]], split_ratio: float, seed: int = 42
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split data into training and validation sets.

    Args:
        data: List of dictionaries formatted for OpenAI finetuning.
        split_ratio: Ratio to split data (0.0-1.0).
        seed: Random seed for shuffling.

    Returns:
        Tuple containing training and validation data lists.
    """
    if not isinstance(split_ratio, float):
        split_ratio = float(split_ratio)

    if split_ratio < 0 or split_ratio > 1:
        logger.exception("Invalid split ratio. Must be between 0 and 1.")
        raise ValueError(f"Invalid split ratio: {split_ratio}. Must be between 0 and 1")

    # Shuffle data
    shuffled = data.copy()
    random.seed(seed)
    random.shuffle(shuffled)

    # Calculate split point
    split_idx = int(len(shuffled) * split_ratio)

    # Split data
    train_data = shuffled[:split_idx]
    val_data = shuffled[split_idx:]

    return train_data, val_data


def upload_finetune_data(
    client: OpenAI,
    input_path: PathLike,
    config_file_path: PathLike,
    file_type: Optional[Literal["training_file", "validation_file"]] = "training_file",
    to_reupload: bool = False,
) -> str | None:
    """Upload the fine-tuning data to OpenAI's Files API, reusing a cached file id if available.

    Args:
        client (OpenAI): The OpenAI client object.
        input_path (PathLike): Path to the JSONL input data.
        config_file_path (PathLike): Path to the JSON config file.
        file_type (str, optional): The type of file to upload (e.g., "train", "val", "training", "valid", "validation"). Defaults to "training_file".
        to_reupload (bool): If True, delete the cached file id from config and re-upload the file.

    Returns:
        str: The file id of the uploaded file, or None if there was an error.
    """
    # Ensure input_path is valid (expects a .jsonl file)
    input_path = check_file_path(input_path, extensions=["jsonl"])

    if (
        file_type is None
        or isinstance(file_type, str)
        or file_type != "training_file"
        or file_type != "validation_file"
    ):
        logger.warning(
            "File type is None or not specified. Attempting to infer from input path."
        )

    if "train" in str(input_path).lower():
        file_type = "training_file"
    elif "val" in str(input_path).lower():
        file_type = "validation_file"
    else:
        logger.warning(
            "File type could not be inferred from input path. Defaulting to 'training_file'."
        )
        file_type = "training_file"

    # Log ALL the argument values
    logger.debug(
        "Show all arguments",
        input_path=input_path,
        config_file_path=config_file_path,
        file_type=file_type,
        to_reupload=to_reupload,
    )
    logger.debug(
        "Current OpenAI finetune FILES", files=client.files.list(purpose="fine-tune")
    )

    # Attempt to use the cached file id if allowed.
    if not to_reupload:
        cached_file_id = _get_cached_file_id(
            client, config_file_path, file_type, to_reupload=to_reupload
        )
        if cached_file_id:
            return cached_file_id
    else:
        # If overwriting, delete the cached file id.
        _get_cached_file_id(
            client, config_file_path, file_type, to_reupload=to_reupload
        )

    # Upload the file since no valid cache was found.
    new_file_id = upload_file_to_openai(
        client, input_path=input_path, file_type=file_type
    )
    if new_file_id:
        update_config(config_file_path, key=file_type, new_value=new_file_id)
        logger.info(
            "Uploaded file for finetuning to OpenAI",
            file_id=new_file_id,
            source=input_path,
        )
        return new_file_id

    logger.debug(
        "Current OpenAI finetune FILES", files=client.files.list(purpose="fine-tune")
    )

    return None


def _get_cached_file_id(
    client: OpenAI, config_file_path: PathLike, file_type: str, to_reupload: bool
) -> str | None:
    """Retrieve the cached file ID for the specified file type from the config file.

    For the given file type (e.g. "training_file" or "validation_file"), verify that the cached
    file exists via OpenAI's Files API. If to_overwrite is True, delete the file from the API.

    Args:
        client (OpenAI): The OpenAI client object.
        config_file_path (Path): Path to the JSON config file.
        file_type (str): The key in the config (e.g., "training_file", "validation_file").
        to_reupload (bool): If True, delete the cached file from the API.

    Returns:
        str | None: The valid cached file id, or None if not found or if deleted.
    """

    def _check_and_handle_file(file_id: str) -> str | None:
        """Helper function to check if the file exists in the API and handle it accordingly."""
        try:
            file_info = client.files.retrieve(file_id)
            if file_info:
                if to_reupload:
                    client.files.delete(file_id)
                    logger.debug("Deleted cached file id", file_id=file_id)
                    return None
                else:
                    logger.debug("Using cached file id:", file_id=file_id)
                    return file_id
        except Exception:
            logger.exception("Error retrieving file", file_id=file_id)
            return None

    # Read the config file and get the file id
    try:
        config_file_path = check_file_path(config_file_path, extensions=const.JSON_EXT)
        config_data = read_config(config_file_path)
        candidate = config_data.get(file_type, "").strip()
        if candidate:
            return _check_and_handle_file(candidate)
        else:
            return None
    except Exception as e:
        logger.exception("Error reading config file:", config_file=config_file_path)
        raise e
