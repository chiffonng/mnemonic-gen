"""Analyze synthetic data and compute statistics (on diversity of linguistic features, distribution of each judge score, etc.)."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from structlog import getLogger

from src.data_gen.mnemonic_models import MnemonicType

if TYPE_CHECKING:
    from typing import Optional

    from structlog.stdlib import BoundLogger


logger: BoundLogger = getLogger(__name__)


def extract_mnemonic_types(mnemonic: str) -> tuple[str, Optional[str]]:
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
            "etymology",
            "comes from",
            "from",
            "is derived",
            "latin",
            "freek",
            "french",
            "root",
        ],
        MnemonicType.morphology: [
            "morphology",
            "formed from",
            "formed by",
            "made of",
            "composed of",
            "compound",
            "morpheme",
        ],
        MnemonicType.semantic_field: [
            "semantic field",
            "refers to",
            "related to",
            "similar to",
            "synonym",
            "antonym",
            "spectrum",
            ">",
        ],
        MnemonicType.orthography: ["orthography", "looks like", "spell", "divide"],
        MnemonicType.phonetics: [
            "phonetic",
            "sounds like",
            "pronounced as",
            "read as",
            "rhymes",
        ],
        MnemonicType.context: ["context", "used in", "example", "sentence"],
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


def compute_mnemonic_statistics(mnemonics) -> dict:
    """Compute statistics from a list of mnemonics, including diversity of linguistic features and distribution of judge scores.

    Args:
        mnemonics (list): A list of mnemonic objects or strings.

    Returns:
        dict: A dictionary containing statistics such as counts of each mnemonic type, average scores, etc.
    """
    pass


def analyze_mnemonic_diversity(mnemonics) -> dict:
    """Analyze the diversity of mnemonics based on their inguistic features, and plot the results.

    Args:
        mnemonics (list): A list of mnemonic objects or strings.

    Returns:
        dict: A dictionary containing diversity metrics such as unique types, average length, etc.
    """
    pass
