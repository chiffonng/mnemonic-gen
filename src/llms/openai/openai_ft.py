"""Module for fine-tuning OpenAI models."""

from __future__ import annotations

from typing import TYPE_CHECKING

import structlog

if TYPE_CHECKING:
    from pathlib import Path
    from typing import Optional

    from openai import OpenAI
    from structlog.stdlib import BoundLogger

from src.utils import read_config

logger: BoundLogger = structlog.getLogger(__name__)


def finetune_from_config(
    client: OpenAI,
    config_file_path: Path,
    finetuned_model_id_path: Path,
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

        # Write the fine-tuned model id to a file.
        with finetuned_model_id_path.open("w", encoding="utf-8") as f:
            f.write(finetuned_model_id)

        return finetuned_model_id
    elif status == "failed":
        logger.error(f"Fine-tuning job failed: {job_info.error}")
        raise Exception(f"Fine-tuning job failed: {job_info.error}")
    else:
        logger.error(f"Fine-tuning job ended with status: {status}")
        return None
