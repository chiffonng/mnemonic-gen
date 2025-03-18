# src/data_prep/mnemonic_classifier.py
"""Module for classifying mnemonics by linguistic features."""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING

from src.llms.client import complete
from src.utils.common import read_config
from src.utils.error_handlers import check_file_path

if TYPE_CHECKING:
    from typing import Optional

    from src.data_prep.mnemonic_schemas import Mnemonic, MnemonicType
    from src.utils.aliases import PathLike

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def keyword_match(mnemonic: str) -> tuple[MnemonicType, Optional[MnemonicType]]:
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
            if keyword.lower() in keywords_by_type_dict.lower():
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
    config_path: PathLike = "config/openai_conf.json",
    prompt_path: PathLike = "prompts/classify_5shot.txt",
    use_mock: bool = False,
) -> Mnemonic:
    """Classify a mnemonic by its linguistic features using an LLM.

    Args:
        term: The vocabulary term
        mnemonic: The mnemonic to classify
        config_path: Path to LLM configuration file
        prompt_path: Path to the system prompt file
        use_mock: Whether to use mock responses for testing

    Returns:
        MnemonicClassification: The classified linguistic features
    """
    # Validate config path
    config_path = check_file_path(config_path, extensions=["json"])

    # Prepare the prompt for classification
    system_prompt = read_config(prompt_path)
    user_prompt = f"Term: {term}\nMnemonic: {mnemonic}"

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    # Mock response for testing
    if use_mock:
        mock_response = "Mock response for classifying mnemonic with llm."
    else:
        mock_response = None

    # Get classification from LLM
    try:
        response = complete(
            messages=messages,
            config_path=config_path,
            output_schema=Mnemonic,
            mock_response=mock_response,
        )
        result = response.choices[0].message.content

        # Parse the result
        return Mnemonic.model_validate_json(result)

    except Exception as e:
        logger.error(f"Error classifying mnemonic for term '{term}': {e}")
        # Fallback classification
        return Mnemonic(
            term=term,
            mnemonic=mnemonic,
            linguistic_reasoning="Failed to classify automatically",
            main_type=MnemonicType.unknown,
            sub_type=None,
        )
