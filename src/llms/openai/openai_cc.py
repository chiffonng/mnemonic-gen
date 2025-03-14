"""Module for generating completions using OpenAI's Chat Completions API."""

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Any

    from openai import OpenAI

    from src.utils import PathLike

from src.utils import check_file_path, read_config, read_prompt

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())


def openai_generate_completion(
    client: "OpenAI",
    config_path: "PathLike",
    prompt_path: "PathLike",
    user_messages: list[dict],
) -> "Any":
    """Generate a chat completion using the OpenAI API.

    Args:
        client (OpenAI): The OpenAI client instance.
        config_path (PathLike): Path to the JSON configuration file.
        prompt_path (PathLike): Path to the system prompt (.txt) file.
        user_messages (list[dict], optional): A list of messages.
            Each message should be a dict with at least "role" and "content" keys.
            If not provided, the function will prompt for a user input.

    Returns:
        Any: The response from the OpenAI API.

    Raises:
        ValueError: If config_path or prompt_path do not exist or have incorrect extensions.
        Exception: If there is an error during API call.
    """
    # Validate paths
    config_path = check_file_path(config_path, extensions=["json"])
    prompt_path = check_file_path(prompt_path, extensions=["txt"])

    # Read configuration and system prompt
    config = read_config(config_path)
    system_prompt = read_prompt(prompt_path)

    # Prepare messages
    messages = [{"role": "system", "content": system_prompt}, *user_messages]
    # TODO: Temporarily added, should have better validation
    assert all(
        isinstance(m, dict) and "role" in m and "content" in m for m in messages
    ), "Invalid message format"

    # Extract required parameters from config
    model = config.get("model", "gpt-4o-mini-2024-07-18")
    if not model:
        raise ValueError(f"Model not specified in config file: {config_path}")

    stream = config.get("stream", False)

    logger.info(f"Parameters: {config}")

    try:
        response = client.chat.completions.create(**config, messages=messages)

        # Handle streaming response
        if stream:
            full_response = ""
            for chunk in response:
                if chunk.choices[0].delta.content:
                    content = chunk.choices[0].delta.content
                    full_response += content
                    print(content, end="", flush=True)
            print()  # Add a newline at the end
            return full_response

        # Return the completion content
        return response.choices[0].message.content

    except Exception as e:
        logger.error(f"Error generating completion: {e}")
        raise e


def improve_mnemonic(
    client: "OpenAI",
    config_path: "PathLike",
    prompt_path: "PathLike",
    term: str,
    mnemonic: str,
) -> str:
    """Generate an improved mnemonic for a given term using the OpenAI API.

    Args:
        client (OpenAI): The OpenAI client instance.
        config_path (PathLike): Path to the JSON configuration file.
        prompt_path (PathLike): Path to the system prompt (.txt) file.
        term (str): The term for which to improve the mnemonic.
        mnemonic (str): The current mnemonic to improve.

    Returns:
        str: The improved mnemonic.
    """
    additional_messages = [
        {"role": "user", "content": f"Term: {term}\nCurrent mnemonic: {mnemonic}"}
    ]

    improved_mnemonic = openai_generate_completion(
        client, config_path, prompt_path, additional_messages
    )

    return improved_mnemonic


def batch_improve_mnemonics(
    client: "OpenAI",
    config_path: "PathLike",
    prompt_path: "PathLike",
    input_csv_path: "PathLike",
    output_csv_path: "PathLike",
    batch_size: int = 10,
) -> None:
    """Process a CSV file with terms and mnemonics, generating improved mnemonics for each.

    Args:
        client (OpenAI): The OpenAI client instance.
        config_path (PathLike): Path to the JSON configuration file.
        prompt_path (PathLike): Path to the system prompt (.txt) file.
        input_csv_path (PathLike): Path to the input CSV file with "term" and "mnemonic" columns.
        output_csv_path (PathLike): Path to save the output CSV with improved mnemonics.
        batch_size (int): The number of rows to process in each batch.
    """
    import csv

    # Validate paths
    input_csv_path = check_file_path(input_csv_path, extensions=["csv"])
    output_csv_path = check_file_path(
        output_csv_path, extensions=["csv"], new_ok=True, to_create=True
    )

    # Read input CSV
    rows = []
    with input_csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    # Process rows in batches
    for batch_start in range(0, len(rows), batch_size):
        batch_end = min(batch_start + batch_size, len(rows))
        batch = rows[batch_start:batch_end]

        # Filter out rows with missing terms or mnemonics
        valid_batch = []
        for i, row in enumerate(batch):
            term = row.get("term", "").strip()
            mnemonic = row.get("mnemonic", "").strip()

            if not term or not mnemonic:
                logger.warning(
                    f"Row {batch_start + i + 1}: Missing term or mnemonic. Skipping."
                )
                row["improved_mnemonic"] = ""
            else:
                valid_batch.append((i, row))

        if not valid_batch:
            logger.warning(
                f"Batch {batch_start // batch_size + 1}: No valid entries. Skipping."
            )
            continue

        logger.info(
            f"Processing batch {batch_start // batch_size + 1}/{(len(rows) - 1) // batch_size + 1} with {len(valid_batch)} terms"
        )

        try:
            # Prepare batch format for the API
            batch_content = "\n\n".join(
                [
                    f"Index: {idx}\nTerm: {row['term']}\nCurrent mnemonic: {row['mnemonic']}"
                    for idx, row in valid_batch
                ]
            )

            # Create messages for the API call
            user_messages = [
                {"role": "user", "content": batch_content},
            ]

            # Make the API call
            response = openai_generate_completion(
                client, config_path, prompt_path, user_messages
            )

            # Extract improved mnemonics from the response. Each improved mnemonic is separated by a line break.
            response.split("\n")

        except Exception as e:
            logger.error(f"Error improving mnemonic for term '{term}': {e}")
            row["improved_mnemonic"] = ""

    # Write output CSV
    with output_csv_path.open("w", encoding="utf-8", newline="") as f:
        fieldnames = ["term", "mnemonic", "improved_mnemonic"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "term": row.get("term", ""),
                    "mnemonic": row.get("mnemonic", ""),
                    "improved_mnemonic": row.get("improved_mnemonic", ""),
                }
            )

    logger.info(
        f"Completed processing {len(rows)} terms. Output saved to {output_csv_path}"
    )
