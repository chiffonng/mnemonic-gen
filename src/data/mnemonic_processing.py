"""Module for processing mnemonics, including code to classify, standardize or diversify them using OpenAI."""

import logging
from pathlib import Path
from typing import TYPE_CHECKING, no_type_check
from warnings import warn

import pandas as pd
from dotenv import load_dotenv
from openai import LengthFinishReasonError, OpenAI, OpenAIError, RateLimitError
from pydantic import BaseModel, ValidationError
from pydantic.functional_validators import AfterValidator
from tenacity import (
    after_log,
    before_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_random_exponential,
)
from tqdm import tqdm
from typing_extensions import Annotated
from yaml import safe_load

if TYPE_CHECKING:
    from openai import Response

from utils.aliases import PathLike
from utils.constants import (
    CLASSIFIED_DATASET_CSV,
    CLASSIFIED_DATASET_PARQUET,
    COMBINED_DATASET_CSV,
    COMBINED_DATASET_PARQUET,
    CSV_EXT,
    PARQUET_EXT,
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

# Initialize OpenAI client
client = OpenAI()

# Load config and prompts
with Path("config/classify_mnemonics.yaml").open("r") as f:
    classification_conf = safe_load(f)  # dict of config
    batch_size = classification_conf["batch_size"]


def validate_classification(value: int) -> int:
    """Validate classification value to be -1, 0, 1, or 2. Otherwise, return -1."""
    return value if value in {-1, 0, 1, 2} else -1


ValidClassification = Annotated[int, AfterValidator(validate_classification)]


# Mnemonic classification schema
class ClassificationSchema(BaseModel):
    """Pydantic schema for the classification of mnemonics."""

    classifications: list[ValidClassification]


def combine_key_value(path: PathLike) -> list[str]:
    """Load 2-column data from a file, to format: key: value.

    Args:
        path (PathLike): The path to the file containing the 2-column data.

    Returns:
        combined_col (list[str]): The combined key and value columns.
    """
    path_obj: Path = check_file_path(path, extensions=[PARQUET_EXT, CSV_EXT])

    if path_obj.suffix == PARQUET_EXT:
        df = pd.read_parquet(path, engine="pyarrow")
    elif path_obj.suffix == CSV_EXT:
        df = pd.read_csv(path, header="infer", quotechar='"')

    logger.info(f"Read {df.shape[0]} rows from {str(path)}.")

    if df.shape[1] > 2:
        warn(
            "More than 2 columns detected. Only the first 2 columns will be used.",
            category=UserWarning,
            stacklevel=2,
        )
        logger.warning(
            "More than 2 columns detected. Only the first 2 columns will be used for processing."
        )
    elif df.shape[1] < 2:
        raise ValueError("File must have at least 2 columns.")

    combined_col = df.iloc[:, 0] + ": " + df.iloc[:, 1]

    return combined_col.to_list()


def create_batches(data: list[str], batch_size=batch_size) -> list[str]:
    """Build batches of text data to send to OpenAI's API.

    Args:
        data (list[str]): The list of data to process.
        batch_size (int, optional): The number of mnemonics to include in each batch. Defaults to batch_size read from the config.

    Returns:
        flattened_batches (list[str]): The list of batches, each item is a batch of text data

    Raises:
        ValueError: if no data is provided or if the batch size is invalid.
    """
    if not data:
        raise ValueError("No data to process.")
    if batch_size < 1 or batch_size > len(data):
        warning = f"Batch size must be between 1 and the number of mnemonics ({len(data)}). Adjusting batch size to {len(data)}."
        warn(warning, category=UserWarning, stacklevel=2)
        logger.warning(warning)
        batch_size = min(batch_size, len(data))

    logger.info(f"Creating batches of {batch_size} mnemonics.")
    batches = [data[i : i + batch_size] for i in range(0, len(data), batch_size)]
    flattened_batches = ["\n".join(batch) for batch in batches]
    logger.info(f"Created {len(batches)} batches of mnemonics.")

    return flattened_batches


@retry(
    retry=retry_if_exception_type(RateLimitError),
    stop=stop_after_attempt(3),
    wait=wait_random_exponential(multiplier=1, min=0, max=4),  # 2^0 to 2^4 seconds
    before=before_log(logger, logging.WARNING),
    after=after_log(logger, logging.WARNING),
)
def classify_mnemonics_api(batches: str | list[str]):
    """Classify mnemonics using OpenAI's API, GPT-4o mini and return the responses as JSON array of numbers. Retry up to 3 times if rate limited.

    Args:
        batches (list[str]): The list of batches of mnemonics to categorize.

    Returns:
        classification_by_batch (list[ValidClassification]): The list of parsed categories.

    Raises:
        ValueError:
        - If the output file is not in parquet or csv
        - If the input (batches) is not a list of strings.
    """
    if not isinstance(batches, (list, str)):
        raise ValueError(
            f"Batches must be a string or a list of strings. Current type: {type(batches)}"
        )
    batches = [batches] if isinstance(batches, str) else batches

    logger.info(f"Processing {len(batches)} batches...")
    logger.info(
        f"Configurations: batch_size={batch_size}, model={classification_conf['model']}, temperature={classification_conf['temperature']}, num_outputs={classification_conf['num_outputs']}."
    )

    classification_by_batch = []
    for i, batch in tqdm(enumerate(batches), desc="Processing batches", unit="batch"):
        classification_msg = get_structured_response(
            i,
            batch,
            model_config=classification_conf,
            response_format=ClassificationSchema,
        )
        classification_batch_i = parse_structured_response(classification_msg, batch, i)
        classification_by_batch.extend(classification_batch_i)

    logger.info(f"Returned {len(classification_by_batch)} classifications.")
    return classification_by_batch


def get_structured_response(
    i: int,
    batch: str,
    model_config: dict,
    response_format: BaseModel = ClassificationSchema,
) -> dict:  # mypy: ignore
    """Get response from OpenAI API. Documentation: https://platform.openai.com/docs/guides/structured-outputs/how-to-use.

    Args:
        i (int): The index of the batch.
        batch (str): The batch of mnemonics to classify.
        model_config (dict): The model configuration.
        response_format (BaseModel, optional): The response format. Defaults to ClassificationSchema.

    Returns:
        structure_msg (dict = openai.Response...message): The structured message object.
    """
    try:
        structure_msg = (
            client.beta.chat.completions.parse(
                model=model_config["model"],
                messages=[
                    {"role": "system", "content": model_config["prompts"]["system"]},
                    {
                        "role": "user",
                        "content": model_config["prompts"]["user"] + batch,
                    },
                ],
                max_tokens=batch_size * 3 + 1,  # 3 tokens per mnemonic
                temperature=model_config["temperature"],
                n=model_config["num_outputs"],
                response_format=response_format,
            )
            .choices[0]
            .message
        )
        if structure_msg.refusal:
            logger.error(f"Batch {i+1}: OpenAI refused to process the request.")
            raise OpenAIError("OpenAI refused to process the request.")

        return structure_msg

    except Exception as e:
        if isinstance(e, LengthFinishReasonError):
            logger.error(f"LengthFinishReasonError: {e}")
            raise ValueError(
                "OpenAI run out of tokens. Please try: reducing the batch_size, or increasing the max_tokens parameter."
            ) from e
        else:
            logger.error(f"Exception: {e}")
            raise e


@no_type_check
def parse_structured_response(
    structure_msg: object,
    batch: str,
    batch_index: int,
) -> list[int]:
    """Parse the structured message from OpenAI's API.

    Args:
        structure_msg (openai.Response...message): The structured message object.
        batch (str): The batch of mnemonics.
        batch_index (int): The index of the batch.

    Returns:
        (list[int]): The list of parsed categories.
    """
    try:
        if structure_msg.parsed:
            classification_batch_i = structure_msg.parsed.classifications
            batch_i_size = len(batch.split("\n"))
            classification_i_size = len(classification_batch_i)

            # Log batch debug info
            logger.debug(
                f"Batch {batch_index+1} with {batch_i_size} mnemonics: {classification_i_size} classifications."
            )
            logger.debug(
                f"Batch {batch_index+1} classifications: {classification_batch_i}"
            )
            logger.debug(
                f"Batch {batch_index+1} types: {type(classification_batch_i[0])}"
            )

            # Handle when the number of classifications does not match the number of mnemonics
            if classification_i_size > batch_i_size:
                logger.warning(
                    f"Batch {batch_index+1}: Number of classifications {classification_i_size} exceeds the number of mnemonics {batch_i_size}. Truncating to match the number of mnemonics..."
                )
                return classification_batch_i[:batch_i_size]

            elif classification_i_size < batch_i_size:
                logger.warning(
                    f"Batch {batch_index+1}: Number of classifications {classification_i_size} is less than the number of mnemonics {batch_i_size}. Padding with -1..."
                )
                return classification_batch_i + [-1] * (
                    batch_i_size - classification_i_size
                )

            else:  # classification_i_size == batch_i_size
                return classification_batch_i

    except ValidationError as e:
        logger.error(f"ValidationError: {e}")
        raise ValueError(
            f"Batch {batch_index+1}: The response didn't match the expected format. Check the logs for more details."
        ) from e


def save_structured_outputs(
    outputs: list[ValidClassification], input_path: PathLike, output_path: PathLike
):
    """Save the classification results to an existing file of mnemonics.

    Args:
        outputs (list[ValidClassification]): The list of parsed categories.
        input_path (PathLike): The path to the file containing the mnemonics.
        output_path (PathLike): The path to .csv or .parquet file to write the parsed.

    Raises:
        ValueError: If the output file is not in parquet or csv format.
    """
    # Set up output path
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Read initial dataset to get the number of rows
    input_path = check_file_path(
        input_path, new_ok=True, extensions=[PARQUET_EXT, CSV_EXT]
    )
    df = (
        pd.read_csv(input_path)
        if input_path.suffix == CSV_EXT
        else pd.read_parquet(input_path)
    )
    if len(df) != len(outputs):
        error_msg = f"Number of rows in the file does not match the number of categories. Number of rows: {len(df)}, number of categories: {len(outputs)}"
        logger.error(error_msg)
        raise ValueError(error_msg)

    # Add the categories column and save to the requested format
    df["category"] = outputs
    save_func = df.to_parquet if output_path.suffix == PARQUET_EXT else df.to_csv
    save_func(output_path, index=False)
    logger.info(f"Saved classification results to {str(output_path)}.")


def standardize_mnemonics_api(batches):
    """Standardize mnemonics using OpenAI's API, GPT-4o mini."""
    raise NotImplementedError


def diversify_mnemonics_api(batches):
    """Diversify mnemonics using OpenAI's API, GPT-4o mini."""
    raise NotImplementedError


def classify_mnemonics(input_path: str, output_path: str):
    """End-to-end function for classifying mnemonics.

    Args:
        input_path (str): The path to the file containing the mnemonics.
        output_path (str): The path to the file to save the classification results.

    Raises:
        ValueError: If the output file is not in parquet or csv format.
    """
    data = combine_key_value(input_path)
    batches = create_batches(data)
    classifications = classify_mnemonics_api(batches)
    save_structured_outputs(classifications, input_path, output_path)


classify_mnemonics(COMBINED_DATASET_CSV, CLASSIFIED_DATASET_CSV)
