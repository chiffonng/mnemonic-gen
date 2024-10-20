"""Module for processing mnemonics, including code to categorize, standardize or diversify them using OpenAI."""

import logging
from collections.abc import Iterable, Sequence
from pathlib import Path

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

load_dotenv()  # Load environment variables

# Set up logging to file
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
handler = logging.FileHandler("logs/mnemonic_processing.log")
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)

# Initialize OpenAI client
client = OpenAI()

# Load config and prompts
with open("prompts.categorize_mnemonics.yaml", "r") as f:
    categorization_config = safe_load(f)
    CATEGORIZE_SYSTEM_PROMPT = categorization_config["prompts"]["system"]
    CATEGORIZE_USER_PROMPT = categorization_config["prompts"]["user"]


def combine_key_value(path: str) -> list[str]:
    """Load 2-column data from a file, to format: key: value.

    Args:
        path (str): The path to the file containing the 2-column data.

    Returns:
        combined_col (Iterable[str]): The combined key and value columns.
    """
    if not Path(path).exists():
        raise FileNotFoundError(f"File not found at {path}.")

    elif Path(path).suffix == ".parquet":
        df = pd.read_parquet(path, engine="pyarrow")
        logger.info(f"Read {df.shape[0]} rows from parquet file.")

    elif Path(path).suffix == ".csv":
        df = pd.read_csv(path, header="infer")
        logger.info(f"Read {df.shape[0]} rows from csv file.")

    else:
        raise ValueError("File must be in parquet or csv format.")

    combined_col = df.iloc[:, 0] + ": " + df.iloc[:, 1]

    return combined_col.to_list()


def build_batches(data: list[str], batch_size: int = 50) -> Sequence[list[str], int]:
    """Build batches of text data to send to OpenAI's API.

    Args:
        data (list[str]): The list of data to process.
        batch_size (int): The size of each batch. Defaults to 200.

    Returns:
        flattened_batches (list[str]): The list of batches, each item is a batch of text data
        batch_size (int): The size of each batch

    Raises:
        ValueError: one of the following conditions is met:
        - no data is provided,
        - batch size is less than 1,
        - batch size is greater than the number of mnemonics
    """
    if not data:
        raise ValueError("No data to process.")
    elif batch_size < 1:
        raise ValueError("Batch size must be greater than 0.")
    elif batch_size > len(data):
        logger.warning(
            "Batch size is greater than the number of mnemonics. Reducing batch size..."
        )
        batch_size = min(batch_size, len(data))

    logger.info(f"Creating batches of {batch_size} mnemonics.")
    batches = [data[i : i + batch_size] for i in range(0, len(data), batch_size)]
    flattened_batches = ["\n".join(batch) for batch in batches]
    logger.info(f"Created {len(batches)} batches of mnemonics.")

    return flattened_batches, batch_size


@retry(
    retry=retry_if_exception_type(RateLimitError),
    retry_error_callback=lambda x: logger.error(f"Exception during retries: {x}"),
    stop=stop_after_attempt(3),
    wait=wait_random_exponential(multiplier=1, min=0, max=4),  # 2^0 to 2^4 seconds
    before=before_log(logger, logging.WARNING),
    after=after_log(logger, logging.WARNING),
)
def categorize_mnemonics(
    batches: list[str],
    batch_size: int,
    n: int = 1,  # !: Currently only supports n=1
    output_path: str | Path = "data/final.csv",
):
    """Categorize mnemonics using OpenAI's API, GPT-4o mini, and write results to a file (to save costs).

    Args:
        batches (list[str]): The list of batches of mnemonics to categorize.
        batch_size (int): The size of each batch.
        n (int): The number of completions to generate for each batch. Defaults to 1.
        output_path (str): The path to the .csv (or .parquet) file to write the results to. Defaults to FINAL_DATASET_CSV.

    Returns:
        categories (list[int]): The list of categories for each mnemonic.

    Raises:
        ValueError:
        - If the output file is not in parquet or csv
        - If the input (batches) is not a list or collections.abc.Iterable of strings.
    """
    if isinstance(batches, str):
        batches = [batches]
    elif isinstance(batches, Iterable):
        batches = list(batches)
    else:
        raise ValueError(
            "Batches must be a list or collections.abc.Iterable of strings."
        )

    logger.info(f"Processing {len(batches)} batches...")
    res_str = ""
    for i in tqdm(range(len(batches)), desc="Processing batches", unit="batch"):
        completion = client.chat.completions.create(
            model=categorization_config["model"],
            messages=[
                {"role": "system", "content": CATEGORIZE_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": CATEGORIZE_USER_PROMPT + batches[i],
                },
            ],
            max_completion_tokens=batch_size * 2 + 3,  # 2 tokens per mnemonic
            temperature=categorization_config["temperature"],
            n=n,  # Generate n completions for each batch
        )
        logger.info(f"Completed categorization for batch {i + 1}.")
        logger.info(
            f"Used total {completion.usage.total_tokens} tokens, including {completion.usage.prompt_tokens} prompt tokens and {completion.usage.completion_tokens} completion tokens."
        )
        res_str += completion.choices[0].message.content
    return parse_write_categorization_results(res_str, output_path)


def parse_write_categorization_results(
    str_of_nums: str, output_path: str | Path
) -> list[int]:
    """Parse a string of numbers into a list of numbers and write to a file.

    Args:
        str_of_nums (str): The string of numbers to parse.
        output_path (str | Path): The path to .csv or .parquet file to write the parsed.

    Returns:
        scores (list[int]): The list of parsed numbers.

    Raises:
        ValueError: If the output file is not in parquet or csv format.
    """
    categories = [int(c) for c in str_of_nums.split(",")]

    assert all(
        c in [-1, 0, 1, 2] for c in categories
    ), "Categories must be -1, 0, 1, or 2."

    # Add a column for categories to the final dataset, whether it's a csv or parquet file
    if Path(output_path).suffix == ".parquet":
        df = pd.read_parquet(output_path, engine="pyarrow")
        if df.shape[0] != len(categories):
            df.to_csv("temp/categories.csv", index=False)
            raise ValueError(
                "Number of rows in the existing file does not match the number of categories."
            )
        df["category"] = categories
        df.to_parquet(output_path, engine="pyarrow")

    elif Path(output_path).suffix == ".csv":
        df = pd.read_csv(output_path, header="infer")
        if df.shape[0] != len(categories):
            df.to_csv("temp/categories.csv", index=False)
            raise ValueError(
                "Number of rows in the existing file does not match the number of categories."
            )
        df["category"] = categories
        df.to_csv(output_path, index=False)

    else:
        raise ValueError("Output file must be in parquet or csv format.")

    return categories


def standardize_mnemonics(batches):
    """Standardize mnemonics using OpenAI's API, GPT-4o mini."""
    raise NotImplementedError


def diversify_mnemonics(batches):
    """Diversify mnemonics using OpenAI's API, GPT-4o mini."""
    raise NotImplementedError


# Prepare data to send to OpenAI
data = combine_key_value("data/final.csv")
batches, batch_size = build_batches(data, batch_size=50)

# Categorize mnemonics
categorize_mnemonics(batches, batch_size, output_path="data/final.csv")
