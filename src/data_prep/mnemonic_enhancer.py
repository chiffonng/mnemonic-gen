"""Module for improving and classifying mnemonics using LLM APIs."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import pandas as pd
from sqlmodel import Session

from src.data_prep.init_db import engine
from src.data_prep.mnemonic_schemas import (
    ImprovedMnemonic,
    Mnemonic,
    MnemonicClassification,
)
from src.llms.client import complete
from src.utils import check_file_path, read_prompt

if TYPE_CHECKING:
    from src.utils import PathLike

# Set up logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
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

    # Build input messages
    messages = [
        {
            "role": "user",
            "content": improve_system_prompt.format(term=term, mnemonic=mnemonic),
        },
        {"role": "user", "content": classify_system_prompt.format(mnemonic=mnemonic)},
    ]

    # Call the LLM API
    if use_mock:
        improve_mock_response = "Mock response for improving mnemonic."
        classify_mock_response = "Mock response for classifying mnemonic."
    else:
        improve_mock_response = None
        classify_mock_response = None

    with Session(engine) as session:
        # Check if the mnemonic already exists in the database
        existing_mnemonic = session.exec(
            "SELECT * FROM mnemonic WHERE term = :term", {"term": term}
        ).first()
        # If it exists, return the existing mnemonic
        if existing_mnemonic:
            return existing_mnemonic

        else:
            # Send the request to the LLM API and get the response
            # Improve the mnemonic
            improved_mnemonic_obj = complete(
                messages=messages,
                config_path=config_path,
                default_config_path=config_path,
                output_schema=ImprovedMnemonic,
                mock_response=improve_mock_response,
            )
            improve_mnemonic = improved_mnemonic_obj.improved_mnemonic
            linguistic_reasoning = improved_mnemonic_obj.linguistic_reasoning

            # Classify the mnemonic
            classification_obj = complete(
                messages=messages,
                config_path=config_path,
                default_config_path=config_path,
                output_schema=MnemonicClassification,
                mock_response=classify_mock_response,
            )
            main_type = classification_obj.main_type
            sub_type = classification_obj.sub_type

            # Create the final response
            mnemonic_entry = Mnemonic(
                term=term,
                mnemonic=mnemonic,
                improved_mnemonic=improve_mnemonic,
                linguistic_reasoning=linguistic_reasoning,
                main_type=main_type,
                sub_type=sub_type,
            )

            # Save the mnemonic to the database
            session.add(mnemonic_entry)
            session.commit()
            session.refresh(mnemonic_entry)  # get the latest state of the object

    return mnemonic_entry
