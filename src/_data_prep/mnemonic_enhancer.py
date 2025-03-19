"""Module for improving and classifying mnemonics using LLM APIs."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from sqlmodel import Session, select

from src._data_prep.init_db import engine
from src._data_prep.mnemonic_schemas import ImprovedMnemonic, MnemonicClassification
from src.data.mnemonic_models import Mnemonic
from src.llms.client import batch_complete, complete
from src.utils import read_prompt

if TYPE_CHECKING:
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
    improve_system_prompt_path: PathLike,
    classify_system_prompt_path: PathLike,
    user_prompt_path: PathLike,
    placeholders_json_path: PathLike,
    use_mock: bool = False,
) -> dict:
    """Improve a mnemonic using LLM API with structured output.

    Args:
        term (str): The vocabulary term
        mnemonic (str): The mnemonic aid for the term
        config_path (PathLike): Path to the LLM configuration
        improve_system_prompt_path (PathLike): Path to the system prompt for improving mnemonics
        classify_system_prompt_path (PathLike): Path to the system prompt for classify mnemonic linguistic features
        user_prompt_path (PathLike): Path to the user prompt for mnemonics
        placeholders_json_path (PathLike): Path to the JSON file containing placeholders for system prompts
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
        improve_system_prompt_path, vars_json_path=placeholders_json_path
    )
    classify_system_prompt = read_prompt(
        classify_system_prompt_path, vars_json_path=placeholders_json_path
    )
    improve_user_prompt = read_prompt(
        user_prompt_path, vars={"term": term, "mnemonic": mnemonic}
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
            user_prompt_path, vars={"term": term, "mnemonic": improved_mnemonic}
        )
        classify_messages = [
            {"role": "system", "content": classify_system_prompt},
            {
                "role": "user",
                "content": classify_user_prompt,
            },
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
