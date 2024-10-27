"""Module for processing mnemonics, including code to classify, standardize or diversify them using OpenAI."""

import logging
from pathlib import Path
from warnings import warn

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI, RateLimitError
from tenacity import (
    after_log,
    before_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_random_exponential,
)
from tqdm import tqdm
from yaml import safe_load

from constants import (
    PARQUET_EXT,
    CSV_EXT,
    COMBINED_DATASET_CSV,
    COMBINED_DATASET_PARQUET,
    CLASSIFIED_DATASET_CSV,
    CLASSIFIED_DATASET_PARQUET,
)
from utils.error_handling import check_file_path, which_file_exists

load_dotenv()  # Load environment variables

# Set up logging to file
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.FileHandler("logs/mnemonic_processing.log"))
formatter = logging.Formatter(
    "%(asctime)s - %(levelname)s - %(funcName)s - %(message)s"
)
logger.handlers[0].setFormatter(formatter)

logger.addHandler(logging.StreamHandler())  # Log to console

# Initialize OpenAI client
client = OpenAI()

# Load config and prompts
with open("prompts/classify_mnemonics.yaml", "r") as f:
    classification_conf = safe_load(f)
    CLASSIFY_SYSTEM_PROMPT = classification_conf["prompts"]["system"]
    CLASSIFY_USER_PROMPT = classification_conf["prompts"]["user"]


def combine_key_value(path: str) -> list[str]:
    """Load 2-column data from a file, to format: key: value.

    Args:
        path (str): The path to the file containing the 2-column data.

    Returns:
        combined_col (list[str]): The combined key and value columns.
    """
    path = check_file_path(path, extensions=[PARQUET_EXT, CSV_EXT])

    if path.suffix == PARQUET_EXT:
        df = pd.read_parquet(path, engine="pyarrow")
    elif path.suffix == CSV_EXT:
        df = pd.read_csv(path, header="infer", quotechar='"')

    logger.info(f"Read {df.shape[0]} rows from {str(path)}.")

    if df.shape[1] > 2:
        warn(
            "More than 2 columns detected. Only the first 2 columns will be used.",
            category=UserWarning,
        )
        logger.warning(
            "More than 2 columns detected. Only the first 2 columns will be used for processing."
        )
    elif df.shape[1] < 2:
        raise ValueError("File must have at least 2 columns.")

    combined_col = df.iloc[:, 0] + ": " + df.iloc[:, 1]

    return combined_col.to_list()


def create_batches(data: list[str], batch_size: int = 50) -> tuple[list[str], int]:
    """Build batches of text data to send to OpenAI's API.

    Args:
        data (list[str]): The list of data to process.
        batch_size (int): The size of each batch. Defaults to 50.

    Returns:
        flattened_batches (list[str]): The list of batches, each item is a batch of text data
        batch_size (int): The size of each batch

    Raises:
        ValueError: if no data is provided or if the batch size is invalid.
    """
    if not data:
        raise ValueError("No data to process.")
    if batch_size < 1 or batch_size > len(data):
        warning = f"Batch size must be between 1 and the number of mnemonics ({len(data)}). Adjusting batch size to {len(data)}."
        warn(warning, category=UserWarning)
        logger.warning(warning)
        batch_size = min(batch_size, len(data))

    logger.info(f"Creating batches of {batch_size} mnemonics.")
    batches = [data[i : i + batch_size] for i in range(0, len(data), batch_size)]
    flattened_batches = ["\n".join(batch) for batch in batches]
    logger.info(f"Created {len(batches)} batches of mnemonics.")

    return flattened_batches, batch_size


@retry(
    retry=retry_if_exception_type(RateLimitError, ValueError),
    retry_error_callback=lambda x: logger.error(f"Exception during retries: {x}"),
    stop=stop_after_attempt(3),
    wait=wait_random_exponential(multiplier=1, min=0, max=4),  # 2^0 to 2^4 seconds
    before=before_log(logger, logging.WARNING),
    after=after_log(logger, logging.WARNING),
)
def classify_mnemonics_api(
    batches: list[str],
    batch_size: int,
):
    """Classify mnemonics using OpenAI's API, GPT-4o mini, and write results to a file (to save costs).

    Args:
        batches (list[str]): The list of batches of mnemonics to categorize.
        batch_size (int): The size of each batch.

    Returns:
        responses (str): The string of responses from OpenAI's API, formatted as a string of numbers separated by commas.

    Raises:
        ValueError:
        - If the output file is not in parquet or csv
        - If the input (batches) is not a list or collections.abc.Iterable of strings.
    """
    if not isinstance(batches, (list, str)):
        raise ValueError("Batches must be a string or a list of strings.")
    batches = [batches] if isinstance(batches, str) else batches

    logger.info(f"Processing {len(batches)} batches...")
    responses = [
        client.chat.completions.create(
            model=classification_conf["model"],
            messages=[
                {"role": "system", "content": CLASSIFY_SYSTEM_PROMPT},
                {"role": "user", "content": f"{CLASSIFY_USER_PROMPT}{batch}"},
            ],
            max_completion_tokens=batch_size * 3 + 3,  # 3-4 tokens per mnemonic
            temperature=classification_conf["temperature"],
            n=classification_conf["num_outputs"],
        )
        .choices[0]
        .message.content
        for batch in tqdm(batches, desc="Processing batches", unit="batch")
    ]
    return ",".join(responses)


def parse_save_classification_results(
    res_str: str, output_path: str | Path
) -> list[int]:
    """Parse comma-separated categories and save them to a file.

    Args:
        res_str (str): The string of numbers (which are the categories) separated by commas.
        output_path (str | Path): The path to .csv or .parquet file to write the parsed.

    Returns:
        scores (list[int]): The list of parsed numbers.

    Raises:
        ValueError: If the output file is not in parquet or csv format.
    """
    logger.info(
        f"Received {len(res_str)} characters from OpenAI's API. Preview: {res_str[:100]}"
    )
    categories = [int(c) for c in res_str.split(",") if c.strip().isdigit()]
    if not all(c in {-1, 0, 1, 2} for c in categories):
        raise ValueError("Parsed categories must be -1, 0, 1, or 2.")

    # Set up output path
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Read initial dataset to get the number of rows
    f = which_file_exists(COMBINED_DATASET_CSV, COMBINED_DATASET_PARQUET)
    df = pd.read_csv(f) if f.suffix == CSV_EXT else pd.read_parquet(f)
    if len(df) != len(categories):
        logger.error(
            f"Number of rows in the file does not match the number of categories. Number of rows: {len(df)}, number of categories: {len(categories)}"
        )
        raise ValueError(
            f"Number of rows in the file does not match the number of categories. Number of rows: {len(df)}, number of categories: {len(categories)}"
        )

    # Add the categories column and save to the requested format
    df["category"] = categories
    save_func = df.to_parquet if output_path.suffix == PARQUET_EXT else df.to_csv
    save_func(output_path, index=False)
    logger.info(f"Saved classification results to {str(output_path)}.")
    return categories


def standardize_mnemonics_api(batches):
    """Standardize mnemonics using OpenAI's API, GPT-4o mini."""
    raise NotImplementedError


def diversify_mnemonics_api(batches):
    """Diversify mnemonics using OpenAI's API, GPT-4o mini."""
    raise NotImplementedError


def classify_mnemonics(
    input_path: str, output_path: str, batch_size: int = 50, n: int = 1
) -> list[int]:
    """End-to-end function for classifying mnemonics.

    Args:
        input_path (str): The path to the file containing the mnemonics.
        output_path (str): The path to the file to save the classification results.
        batch_size (int): The size of each batch. Defaults to 50.
        n (int): The number of completions to generate. Defaults to 1.

    Returns:
        (list[int]): The list of parsed categories.

    Raises:
        ValueError: If the output file is not in parquet or csv format.
    """
    data = combine_key_value(input_path)
    batches, batch_size = create_batches(data, batch_size)
    raw_response = classify_mnemonics_api(batches, batch_size)
    return parse_save_classification_results(raw_response, output_path)


classify_mnemonics(COMBINED_DATASET_CSV, CLASSIFIED_DATASET_CSV)
