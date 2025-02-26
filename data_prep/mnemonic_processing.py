"""Module for processing mnemonic data."""

import csv
import json
import logging
from collections import defaultdict
from typing import TYPE_CHECKING

from dotenv import load_dotenv
from openai import OpenAI
from src.utils.error_handling import check_file_path

if TYPE_CHECKING:
    from pathlib import Path

    from src.utils.aliases import PathLike

# Set up and validates the paths
prompt_path = check_file_path("data_prep/prompts/improve_ft_system.txt")
raw_input_path = check_file_path(
    "data_prep/raw/improved_mnemonics.csv", extensions=["csv"]
)
input_path = check_file_path(
    "data_prep/processed/improved_mnemonics.jsonl", extensions=["jsonl"], new_ok=True
)

# Set up logging to console
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())
logger.handlers[0].setFormatter(
    logging.Formatter("%(levelname)s - %(funcName)s - %(message)s")
)

# Initialize the OpenAI client
load_dotenv()
client = OpenAI()


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


if __name__ == "__main__":
    prepare_finetune_data()
    validate_finetune_data()
