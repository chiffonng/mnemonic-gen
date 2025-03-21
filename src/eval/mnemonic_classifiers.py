# src/data_prep/mnemonic_classifier.py
"""Module for classifying mnemonics by linguistic features."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

import structlog

from src.llms.client import complete
from src.utils import constants as const
from src.utils.common import read_prompt
from src.utils.types import MnemonicClassification

if TYPE_CHECKING:
    from typing import Optional

    from structlog.stdlib import BoundLogger

    from src.utils.types import MnemonicType, PathLike

logger: BoundLogger = structlog.getLogger(__name__)


def classify_by_keywords(mnemonic: str) -> tuple[MnemonicType, Optional[MnemonicType]]:
    """Match keywords in the mnemonic to classify it.

    Args:
        mnemonic: The mnemonic to classify

    Returns:
        tuple: The main type and sub type of the mnemonic
    """
    if not mnemonic:
        raise ValueError("Mnemonic cannot be empty")
    elif not isinstance(mnemonic, str):
        raise TypeError("Mnemonic must be a string")

    mnemonic = mnemonic.lower()

    keywords_by_type_dict = {
        MnemonicType.etymology: [
            "comes from",
            "from",
            "is derived",
            "latin",
            "freek",
            "french",
            "root",
        ],
        MnemonicType.morphology: [
            "formed from",
            "formed by",
            "made of",
            "composed of",
            "compound",
            "morpheme",
        ],
        MnemonicType.semantic_field: [
            "meaning",
            "refers to",
            "related to",
            "similar to",
            "synonym",
            "antonym",
            "spectrum",
            ">",
        ],
        MnemonicType.orthography: ["looks like", "spell", "divide"],
        MnemonicType.phonetics: ["sounds like", "pronounced as", "read as", "rhymes"],
        MnemonicType.context: ["used in", "example", "sentence"],
    }

    # Count keyword matches
    type_scores = {mtype: 0 for mtype in MnemonicType}
    for mtype, keywords in keywords_by_type_dict.items():
        for keyword in keywords:
            if keyword.lower() in keywords_by_type_dict:
                type_scores[mtype] += 1

    # Sort by score
    sorted_types = sorted(type_scores.items(), key=lambda x: x[1], reverse=True)

    # Assign main and sub types
    main_type = sorted_types[0][0] if sorted_types[0][1] > 0 else MnemonicType.unknown
    sub_type = (
        sorted_types[1][0] if len(sorted_types) > 1 and sorted_types[1][1] > 0 else None
    )

    for mnemonic_type, keywords in keywords_by_type_dict.items():
        if any(keyword in mnemonic for keyword in keywords):
            return mnemonic_type

    # morphology: search for pattern "term = morpheme1 + morpheme2", e.g. "unhappiness = un + happiness"
    pattern = re.compile(r"(\w+)\s*=\s*(\w+)\s*\+\s*(\w+)")
    if pattern.search(mnemonic):
        main_type = MnemonicType.morphology

    return main_type, sub_type


def classify_with_llm(
    term: str,
    mnemonic: str,
    config_path: PathLike,
    system_prompt_path: PathLike,
    user_prompt_path: Optional[PathLike] = None,
    use_mock: bool = False,
) -> tuple[MnemonicType, Optional[MnemonicType]]:
    """Classify a mnemonic by its linguistic features using an LLM.

    Args:
        term: The vocabulary term
        mnemonic: The mnemonic to classify
        config_path: Path to LLM configuration file
        system_prompt_path: Path to the system prompt file
        user_prompt_path: Path to the user prompt file
        use_mock: Whether to use mock responses for testing

    Returns:
        tuple: The main type and sub type of the mnemonic

    Raises:
        Exception: If an error occurs during src.llms.client.complete, or if the mnemonic is empty
    """
    if not mnemonic or not term:
        raise ValueError("Mnemonic and term cannot be empty")
    elif not isinstance(mnemonic, str) or not isinstance(term, str):
        raise TypeError("Mnemonic and term must be strings")

    # Prepare the prompts for classification
    system_prompt = read_prompt(system_prompt_path)
    if user_prompt_path:
        user_prompt = read_prompt(user_prompt_path)
    else:
        user_prompt = f"Classify the mnemonic '{mnemonic}' for the term '{term}' by its linguistic features."

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    # Mock response for testing
    mock_response = (
        "Mock response for classifying mnemonic with llm." if use_mock else None
    )

    # Get classification from LLM
    try:
        mnemonic_classes_obj = complete(
            messages=messages,
            config_path=config_path,
            output_schema=MnemonicClassification,
            mock_response=mock_response,
        )

        # Parse the result
        return mnemonic_classes_obj.main_type, mnemonic_classes_obj.sub_type

    except Exception as e:
        logger.exception(
            "Error classifying mnemonic for term",
            term=term,
            mnemonic=mnemonic,
        )
        raise e
