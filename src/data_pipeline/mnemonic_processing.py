"""Module for processing mnemonics, including code to categorize, standardize or diversify them using OpenAI."""

import logging
from collections.abc import Iterable
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

load_dotenv()  # Load environment variables

# Set up logging to console
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)

client = OpenAI()


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


def build_batches(data: list[str], batch_size: int = 75) -> list[str]:
    """Build batches of text data to send to OpenAI's API.

    Args:
        data (list[str]): The list of data to process.
        batch_size (int): The size of each batch.

    Returns:
        flattened_batches (list[str]): The list of batches, each item is a batch of text data

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

    return flattened_batches


@retry(
    retry=retry_if_exception_type(RateLimitError),
    stop=stop_after_attempt(3),
    wait=wait_random_exponential(multiplier=2, max=100),
    before=before_log(logger, logging.WARNING),
    after=after_log(logger, logging.WARNING),
)
def categorize_mnemonics(
    batches: list[str],
    n: int = 1,  # !: Currently only supports n=1
    output_path: str = "temp/categorization_results.txt",
):
    """Categorize mnemonics using OpenAI's API, GPT-4o mini, and write results to a file (to save costs).

    Args:
        batches (list[str]): The list of batches of mnemonics to categorize.
        n (int): The number of completions to generate for each batch. Defaults to 1.

    Returns:
    """
    CATEGORIZE_SYSTEM_PROMPT = """You are an expert in English mnemonics. You are given a list of terms and mnemonics to categorize as shallow-encoding (0), deep-encoding (1), or mixed (2). Think through the reasoning for categorization by yourself. Respond with a number (0, 1, or 2) for each mnemonic, seperated by commas. If unsure, skip return -1."""
    CATEGORIZE_USER_PROMPT = """Categorize the mnemonics below as:\n
    - Shallow (0): Focus on how the word sounds, looks, or rhymes.
    - Deep (1): Focus on semantics, morphology, etymology, context (inferred meaning, imagery), related words (synonyms, antonyms, words with same roots). Repeating the word or using a similar-sounding word is NOT deep encoding.
    - Mixed (2): Contains both shallow and deep features.

    Examples:
    - olfactory: Sounds like "old factory." The old factory had a strong smell, reminding workers of its olfactory history. The mnemonic is shallow since it's based on the sound of the word.
    - vacuous: Same Latin root "vacare" (empty) as "vacuum, vacant". His expression was as empty as a vacuum, showing no signs of thought. The mnemonic is deep since only uses etymology and related words.
    - malevolent: From male 'ill' + volent 'wishing' (as in "benevolent"). These male species are so violent that they always have evil plans. The mnemonic is mixed because it uses etymology and antonyms (deep), and the sounds of "male" and "violent" (shallow)\n"""

    if isinstance(batches, str):
        batches = [batches]
    elif isinstance(batches, Iterable):
        batches = list(batches)
    else:
        raise ValueError(
            "Batches must be a list or collections.abc.Iterable of strings."
        )

    logger.info(f"Processing {len(batches)} batches of mnemonics.")
    res_str = ""
    for i in tqdm(range(len(batches)), desc="Processing batches", unit="batch"):
        try:
            completion = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": CATEGORIZE_SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": CATEGORIZE_USER_PROMPT + batches[i],
                    },
                ],
                max_completion_tokens=len(batches[i]) * 2 + 3,
                temperature=0.4,
                n=n,  # Generate n completions for each batch
            )
            logger.info(f"Completed categorization for batch {i + 1}.")
            logger.info(
                f"Used total {completion.usage.total_tokens} tokens, including {completion.usage.prompt_tokens} prompt tokens and {completion.usage.completion_tokens} completion tokens."
            )
            res_str += completion.choices[0].message.content

        except Exception as e:
            logger.error(f"Error occurred during categorization: {e}")
            raise e
    return parse_write_categorization_results(res_str, output_path)


def parse_write_categorization_results(
    str_of_nums: str, output_path: str | Path
) -> list[int]:
    """Parse a string of numbers into a list of numbers and write to a file.

    Args:
        str_of_nums (str): The string of numbers to parse.
        output_path (str | Path): The path to .csv file to write the parsed

    Returns:
        scores (list[int]): The list of parsed numbers.
    """
    scores = [int(score) for score in str_of_nums.split(",")]

    assert all(
        score in [-1, 0, 1, 2] for score in scores
    ), "Scores must be -1, 0, 1, or 2."

    if isinstance(output_path, str):
        output_path = Path(output_path)
    elif not isinstance(output_path, Path):
        raise ValueError("Output path must be a string or Path.")

    if not output_path.parent.exists():
        output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.suffix != ".csv":
        logger.warning("Output file must be in .csv format. Changing file extension...")
        output_path = output_path.with_suffix(".csv")

    df = pd.DataFrame(scores, columns=["score"])
    df.to_csv(output_path, index=False)
    logger.info(f"Wrote categorization results to {output_path}.")

    return scores


def standardize_mnemonics(batches):
    """Standardize mnemonics using OpenAI's API, GPT-4o mini."""
    raise NotImplementedError


def diversify_mnemonics(batches):
    """Diversify mnemonics using OpenAI's API, GPT-4o mini."""
    raise NotImplementedError


# Prepare data to send to OpenAI
data = combine_key_value("data/final.csv")
batches = build_batches(data, batch_size=5)

# Categorize mnemonics
categorize_mnemonics(batches[:1])
