"""Module for fine-tuning OpenAI models."""

from __future__ import annotations

from typing import TYPE_CHECKING

from structlog import getLogger

if TYPE_CHECKING:
    from typing import Optional

    from openai import OpenAI
    from structlog.stdlib import BoundLogger

    from src.utils.types import PathLike

from src.utils.common import read_config, update_config

logger: BoundLogger = getLogger(__name__)


def finetune_from_config(
    client: OpenAI,
    finetune_config_path: PathLike,
    completion_config_path: PathLike,
    poll_seconds: int = 90,
) -> Optional[str]:
    """Fine tune an OpenAI model using the configuration specified in the config file. This function creates a fine-tuning job via the OpenAI API and polls until the job reaches a terminal state.

    The config file should have the "fine_tuning.job" object format described here: https://platform.openai.com/docs/api-reference/fine-tuning/object

    Args:
        client (OpenAI): The OpenAI client object.
        finetune_config_path (PathLike): The path to the config file.
        completion_config_path (PathLike): The path to the completion config file.
        poll_seconds (int): The number of seconds to wait between querying the job status. Default is 90 seconds.

    Returns:
        finetuned_model_id (Optional[str]): The id of the fine-tuned model, or None if there was an error.

    Raises:
        e: Exception if there was an error creating the fine-tuning job.
    """
    config_kwargs: dict = read_config(finetune_config_path)

    logger.info("Creating fine-tuning job with arguments", config=config_kwargs)

    # Create the fine-tuning job.
    try:
        # TODO: Add wandb integrations
        job_response = client.fine_tuning.jobs.create(**config_kwargs)
    except Exception as e:
        logger.exception("Error creating fine-tuning job:")
        raise e

    job_id = job_response.id
    logger.info("Started fine-tuning job with ID", job_id=job_id)
    logger.debug(
        "OpenAI last fine-tuning JOBS", openai_ft_jobs=client.fine_tuning.jobs.list()
    )

    # Poll until the job reaches a terminal status.
    terminal_statuses = ("succeeded", "failed", "cancelled")
    while True:
        try:
            job_info = client.fine_tuning.jobs.retrieve(job_id)
        except Exception:
            logger.exception("Error retrieving fine-tuning job", job_id=job_id)
            break

        status = job_info.status
        logger.debug(
            f"Polling OpenAI fine-tuning job every {poll_seconds} seconds",
            job_id=job_id,
            status=status,
        )

        if status in terminal_statuses:
            break

        import time

        time.sleep(poll_seconds)

    if status == "succeeded":
        finetuned_model_id = job_info.fine_tuned_model
        logger.info(
            "Fine-tuning succeeded. Fine-tuned model", model_id=finetuned_model_id
        )

        # Save the fine-tuned model id to completion config
        update_config(completion_config_path, key="model", new_value=finetuned_model_id)

        return finetuned_model_id
    elif status == "failed":
        logger.exception("Fine-tuning job failed", job_id=job_id, error=job_info.error)
        raise Exception(f"Fine-tuning job failed: {job_info.error}")
    else:
        logger.warning("Fine-tuning job cancelled", job_id=job_id, status=status)
        return None
