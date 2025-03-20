"""Module for improving and classifying mnemonics using LLM APIs."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import pandas as pd
from sqlmodel import Session, select
from tqdm import tqdm

from src._data_prep.init_db import engine
from src._data_prep.mnemonic_schemas import ImprovedMnemonic, MnemonicClassification
from src.data.mnemonic_models import Mnemonic, MnemonicType
from src.llms.client import batch_complete, complete
from src.utils import check_file_path, read_prompt
from src.utils import constants as const

if TYPE_CHECKING:
    from typing import Optional

    from src.utils import PathLike

# Set up logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(handler)


def improve_mnemonic(
    term: str,
    mnemonic: str,
    config_path: PathLike,
    use_mock: bool = False,
) -> dict:
    """Improve a mnemonic using LLM API with structured output.

    Args:
        term (str): The vocabulary term
        mnemonic (str): The mnemonic aid for the term
        config_path (PathLike): Path to the LLM configuration
        use_mock (bool): Whether to use a mock response

    Returns:
        dict: The improved mnemonic with linguistic reasoning and classification
    """
    # Check if the mnemonic already exists in the database; if so, return it
    with Session(engine) as session:
        existing_mnemonic = session.exec(
            select(Mnemonic).where(Mnemonic.term == term)
        ).first()
        if existing_mnemonic:
            logger.info(f"Mnemonic for term '{term}' already exists in the database.")
            return existing_mnemonic

    # Read prompts and configuration
    improve_system_prompt = read_prompt(
        const.FILE_PROMPT_IMPROVE_SYSTEM,
        vars_json_path=const.FILE_PROMPT_PLACEHOLDER_DICT,
    )
    classify_system_prompt = read_prompt(
        const.FILE_PROMPT_CLASSIFY_SYSTEM,
        vars_json_path=const.FILE_PROMPT_PLACEHOLDER_DICT,
    )
    improve_user_prompt = read_prompt(
        const.FILE_PROMPT_USER, vars={"term": term, "mnemonic": mnemonic}
    )

    # Call the LLM API
    if use_mock:
        improve_mock_response = "Mock response for improving mnemonic."
        classify_mock_response = "Mock response for classifying mnemonic."
    else:
        improve_mock_response = None
        classify_mock_response = None

    with Session(engine) as session:
        # Send the request to the LLM API and get the response
        # Improve the mnemonic
        logger.debug(f"Improving mnemonic for term '{term}'...")
        improve_messages = [
            {"role": "system", "content": improve_system_prompt},
            {"role": "user", "content": improve_user_prompt},
        ]
        improved_mnemonic_obj = complete(
            messages=improve_messages,
            config_path=config_path,
            output_schema=ImprovedMnemonic,
            mock_response=improve_mock_response,
        )
        logger.debug(f"Improved mnemonic: {improved_mnemonic_obj}")
        improved_mnemonic = improved_mnemonic_obj.improved_mnemonic

        # Classify the mnemonic
        logger.debug(f"Classifying mnemonic for term '{term}'...")
        classify_user_prompt = read_prompt(
            const.FILE_PROMPT_USER, vars={"term": term, "mnemonic": improved_mnemonic}
        )
        classify_messages = [
            {"role": "system", "content": classify_system_prompt},
            {"role": "user", "content": classify_user_prompt},
        ]
        classification_obj = complete(
            messages=classify_messages,
            config_path=config_path,
            output_schema=MnemonicClassification,
            mock_response=classify_mock_response,
        )
        logger.debug(f"Mnemonic classification: {classification_obj}")

        # Create the final response
        mnemonic_entry = Mnemonic(
            term=term,
            mnemonic=improved_mnemonic,
            linguistic_reasoning=improved_mnemonic_obj.linguistic_reasoning,
            main_type=classification_obj.main_type,
            sub_type=classification_obj.sub_type,
        )

        # Save the mnemonic to the database
        session.add(mnemonic_entry)
        session.commit()
        session.refresh(mnemonic_entry)  # get the latest state of the object

    return mnemonic_entry


def batch_improve_mnemonics(
    input_path: PathLike = const.COMBINED_DATASET_CSV,
    config_path: PathLike = const.CONF_OPENAI,
    batch_size: int = 5,
    max_items: Optional[int] = 10,
    use_mock: bool = False,
) -> int:
    """Batch improve mnemonics using LLM API with parallel processing and store in database.

    Args:
        input_path (PathLike): Path to the input CSV file
        config_path (PathLike): Path to the LLM configuration
        batch_size (int): Number of mnemonics to process in each batch
        max_items (int, optional): Maximum number of mnemonics to process
        use_mock (bool): Whether to use mock responses

    Returns:
        int: Number of mnemonics processed and stored in the database
    """
    # Read the dataset
    input_path = check_file_path(input_path, extensions=[".csv"])
    df = pd.read_csv(input_path)
    logger.info(f"Loaded {len(df)} mnemonics from {input_path}")

    # Limit the number of items if specified
    if max_items and max_items < len(df):
        df = df.sample(max_items, random_state=42)
        logger.info(f"Limited to {max_items} mnemonics")

    # Read prompts
    improve_system_prompt = read_prompt(
        const.FILE_PROMPT_IMPROVE_SYSTEM,
        vars_json_path=const.FILE_PROMPT_PLACEHOLDER_DICT,
    )
    classify_system_prompt = read_prompt(
        const.FILE_PROMPT_CLASSIFY_SYSTEM,
        vars_json_path=const.FILE_PROMPT_PLACEHOLDER_DICT,
    )

    # Prepare mock responses if needed
    if use_mock:
        improve_mock_response = '{"improved_mnemonic": "Mock improved mnemonic.", "linguistic_reasoning": "Mock linguistic reasoning."}'
        classify_mock_response = '{"main_type": "etymology", "sub_type": null}'
    else:
        improve_mock_response = None
        classify_mock_response = None

    processed_count = 0

    # Process data in batches
    for i in tqdm(range(0, len(df), batch_size), desc="Processing batches"):
        batch = df.iloc[i : i + batch_size]

        # Step 1: Prepare batched improve messages
        improve_messages_batch = []
        for _, row in batch.iterrows():
            user_prompt = read_prompt(
                const.FILE_PROMPT_USER,
                vars={"term": row["term"], "mnemonic": row["mnemonic"]},
            )
            improve_messages_batch.append(
                [
                    {"role": "system", "content": improve_system_prompt},
                    {"role": "user", "content": user_prompt},
                ]
            )

        # Step 2: Send batch request for improving mnemonics
        try:
            improved_batch = batch_complete(
                messages=improve_messages_batch,
                config_path=config_path,
                output_schema=ImprovedMnemonic,
                mock_response=improve_mock_response,
            )
            logger.debug(
                f"Batch improvement successful. Results: {len(improved_batch)}"
            )
        except Exception as e:
            logger.error(f"Error in batch improvement: {e}")
            improved_batch = [None] * len(batch)

        # Step 3: Prepare batched classify messages based on improvement results
        classify_messages_batch = []
        batch_data = []

        for idx, (_, row) in enumerate(batch.iterrows()):
            term = row["term"]
            original_mnemonic = row["mnemonic"]

            # Handle successful or failed improvement
            if idx < len(improved_batch) and improved_batch[idx] is not None:
                improved_mnemonic = improved_batch[idx].improved_mnemonic
                linguistic_reasoning = improved_batch[idx].linguistic_reasoning
            else:
                improved_mnemonic = original_mnemonic
                linguistic_reasoning = "Failed to generate improved mnemonic."

            # Store data for later use
            batch_data.append(
                {
                    "term": term,
                    "mnemonic": improved_mnemonic,
                    "linguistic_reasoning": linguistic_reasoning,
                }
            )

            # Prepare classification message
            user_prompt = read_prompt(
                const.FILE_PROMPT_USER,
                vars={"term": term, "mnemonic": improved_mnemonic},
            )
            classify_messages_batch.append(
                [
                    {"role": "system", "content": classify_system_prompt},
                    {"role": "user", "content": user_prompt},
                ]
            )

        # Step 4: Send batch request for classification
        try:
            classification_batch = batch_complete(
                messages=classify_messages_batch,
                config_path=config_path,
                output_schema=MnemonicClassification,
                mock_response=classify_mock_response,
            )
            logger.debug(
                f"Batch classification successful. Results: {len(classification_batch)}"
            )
        except Exception as e:
            logger.error(f"Error in batch classification: {e}")
            classification_batch = [None] * len(batch)

        # Step 5: Process results and store in database
        with Session(engine) as session:
            batch_processed = 0
            for idx, data in enumerate(batch_data):
                try:
                    # Get classification or use default
                    if (
                        idx < len(classification_batch)
                        and classification_batch[idx] is not None
                    ):
                        main_type = classification_batch[idx].main_type
                        sub_type = classification_batch[idx].sub_type
                    else:
                        main_type = MnemonicType.unknown
                        sub_type = None

                    # Check if entry already exists in database
                    existing = session.exec(
                        select(Mnemonic).where(Mnemonic.term == data["term"])
                    ).first()

                    if existing:
                        # Update existing entry
                        existing.mnemonic = data["improved_mnemonic"]
                        existing.linguistic_reasoning = data["linguistic_reasoning"]
                        existing.main_type = main_type
                        existing.sub_type = sub_type
                    else:
                        # Create new entry
                        mnemonic_entry = Mnemonic(
                            term=data["term"],
                            mnemonic=data["improved_mnemonic"],
                            linguistic_reasoning=data["linguistic_reasoning"],
                            main_type=main_type,
                            sub_type=sub_type,
                        )
                        session.add(mnemonic_entry)

                    batch_processed += 1

                except Exception as e:
                    logger.error(
                        f"Error processing result for term '{data['term']}': {e}"
                    )

            # Commit all changes at once
            session.commit()
            processed_count += batch_processed

            logger.info(
                f"Processed batch {i // batch_size + 1}: {batch_processed} mnemonics stored in database"
            )

    logger.info(f"Total mnemonics processed and stored in database: {processed_count}")
    return processed_count
