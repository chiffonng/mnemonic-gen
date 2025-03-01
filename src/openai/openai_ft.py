"""Module for fine-tuning OpenAI models."""

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
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())


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


def finetune_from_config(
    client: "OpenAI",
    config_file_path: "Path",
    finetuned_model_id_path: "Path",
    poll_seconds: int = 60,
) -> "Optional[str]":
    """Fine tune an OpenAI model using the configuration specified in the config file. This function creates a fine-tuning job via the OpenAI API and polls until the job reaches a terminal state.

    The config file should have the "fine_tuning.job" object format described here: https://platform.openai.com/docs/api-reference/fine-tuning/object

    Args:
        client (OpenAI): The OpenAI client object.
        config_file_path (Path): The path to the config file.
        finetuned_model_id_path (Path): The path to the file where the fine-tuned model id will be written.
        poll_seconds (int): The number of seconds to wait between querying the job status.

    Returns:
        finetuned_model_id (Optional[str]): The id of the fine-tuned model, or None if there was an error.

    Raises:
        e: Exception if there was an error creating the fine-tuning job.
    """
    config_kwargs: dict = read_config(config_file_path)

    logger.info(f"Creating fine-tuning job with kwargs: {config_kwargs}")

    # Create the fine-tuning job.
    try:
        # TODO: Add wandb integrations
        job_response = client.fine_tuning.jobs.create(**config_kwargs)
    except Exception as e:
        logger.error(f"Error creating fine-tuning job: {e}")
        raise e

    job_id = job_response.id
    logger.info(f"Started fine-tuning job with ID: {job_id}")
    logger.debug(f"OpenAI Fine-tuning JOBS: {client.fine_tuning.jobs.list()}")

    # Poll until the job reaches a terminal status.
    terminal_statuses = ("succeeded", "failed", "cancelled")
    while True:
        try:
            job_info = client.fine_tuning.jobs.retrieve(job_id)
        except Exception as e:
            logger.error(f"Error retrieving job {job_id}: {e}")
            break

        status = job_info.status
        logger.info(f"Fine-tuning job {job_id} status: {status}")

        if status in terminal_statuses:
            break

        import time

        time.sleep(poll_seconds)

    if status == "succeeded":
        finetuned_model_id = job_info.fine_tuned_model
        logger.info(f"Fine-tuning succeeded. Fine-tuned model: {finetuned_model_id}")

        # TODO: Write fine-tuned model object to an out file
        finetuned_model_id_path.parent.mkdir(parents=True, exist_ok=True)
        with finetuned_model_id_path.open("w", encoding="utf-8") as f:
            f.write(finetuned_model_id)

        return finetuned_model_id
    elif status == "failed":
        logger.error(f"Fine-tuning job failed: {job_info.error}")
        raise Exception(f"Fine-tuning job failed: {job_info.error}")
    else:
        logger.error(f"Fine-tuning job ended with status: {status}")
        return None
