"""Module for improving and classifying mnemonics using LLM APIs."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import pandas as pd
from sqlmodel import Session, select

from src._data_prep.init_db import engine
from src._data_prep.mnemonic_schemas import (
    ImprovedMnemonic,
    Mnemonic,
    MnemonicClassification,
)
from src.llms.client import batch_complete, complete
from src.utils import check_file_path, read_prompt

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
    improve_system_prompt_path: PathLike,
    classify_system_prompt_path: PathLike,
    config_path: PathLike,
    use_mock: bool = False,
) -> dict:
    """Improve a mnemonic using LLM API with structured output.

    Args:
        term (str): The vocabulary term
        mnemonic (str): The mnemonic aid for the term
        improve_system_prompt_path (PathLike): Path to the system prompt for improving mnemonics
        classify_system_prompt_path (PathLike): Path to the system prompt for classify mnemonic linguistic features
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

    # Check prompts and configuration sources
    improve_system_prompt_path = check_file_path(
        improve_system_prompt_path, extensions=[".txt"]
    )
    classify_system_prompt_path = check_file_path(
        classify_system_prompt_path, extensions=[".txt"]
    )

    # Read prompts and configuration
    improve_system_prompt = read_prompt(improve_system_prompt_path)
    classify_system_prompt = read_prompt(classify_system_prompt_path)

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
            {
                "role": "user",
                "content": improve_system_prompt.format(term=term, mnemonic=mnemonic),
            },
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
        classify_messages = [
            {
                "role": "user",
                "content": classify_system_prompt.format(mnemonic=improved_mnemonic),
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
