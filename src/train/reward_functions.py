"""Reward functions for GRPO training on mnemonic generation tasks."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from structlog import getLogger

from data_gen.reasoner import reason

if TYPE_CHECKING:
    from typing import Any

logger = getLogger(__name__)

# Constants
REASONING_START = "<think>"
REASONING_END = "</think>"
SOLUTION_START = "<answer>"
SOLUTION_END = "</answer>"

LINGUISTIC_FEATURES = [
    "phonetics",
    "orthography",
    "etymology",
    "morphology",
    "semantics",
    "custom",
]

# Regex patterns
FORMAT_REGEX = re.compile(
    rf"{REASONING_START}.+?{REASONING_END}.*?{SOLUTION_START}.+?{SOLUTION_END}",
    re.DOTALL | re.IGNORECASE,
)
REASONING_REGEX = re.compile(
    rf"{REASONING_START}(.+?){REASONING_END}", re.DOTALL | re.IGNORECASE
)
ANSWER_REGEX = re.compile(
    rf"{SOLUTION_START}(.+?){SOLUTION_END}", re.DOTALL | re.IGNORECASE
)
ACRONYM_REGEX = re.compile(r"\b(?:[A-Z]\.){2,}|[A-Z]{2,}\b")


# Extraction functions
def extract_linguistic_feature(text: str) -> str | None:
    """Extract the linguistic feature mentioned in the response.

    Args:
        text: The model's response text

    Returns:
        The linguistic feature if found, None otherwise
    """
    text = text.lower()
    if "linguistic_feature:" in text:
        # first word after "linguistic_feature:" is the feature
        return text.split("linguistic_feature:")[1].split()[0].strip("*")
    return None


def extract_mnemonic(text: str) -> str | None:
    """Extract the mnemonic from the response.

    Args:
        text: The model's response text

    Returns:
        The mnemonic if found, None otherwise
    """
    text = text.lower()
    if "mnemonic:" in text:
        return text.split("mnemonic:")[1].split("\n")[0].strip()
    return None


def extract_example(text: str) -> str | None:
    """Extract the example from the response.

    Args:
        text: The model's response text

    Returns:
        The example if found, None otherwise
    """
    text = text.lower()
    if "example:" in text:
        return text.split("example:")[1].split("\n")[0].strip()
    return None


def extract_reasoning(text: str) -> str | None:
    """Extract the reasoning from the response between <think> and </think> tags.

    Args:
        text: The model's response text

    Returns:
        The reasoning if found, None otherwise
    """
    # get group that matches the middle group of REASONING_REGEX
    match = REASONING_REGEX.search(text)
    if match:
        return match.group(1).strip()
    return None


def extract_solution(text: str) -> str | None:
    """Extract the solution from the response between <solution> and </solution> tags.

    Args:
        text: The model's response text

    Returns:
        The solution if found, None otherwise
    """
    match = ANSWER_REGEX.search(text)
    if match:
        return match.group(1).strip()
    return None


def check_essential_format(completions: list[Any], **kwargs):
    """Checks if responses follow the required format structure."""
    rewards = []

    for completion in completions:
        response = completion[0]["content"]
        response_lower = response.lower()
        score = 0.0

        if FORMAT_REGEX.search(response):
            score += 1.0
            # Use simple string checks for required sections
            if "linguistic" in response_lower:
                score += 0.5
            if "mnemonic:" in response_lower:
                score += 0.5
            if "example:" in response_lower:
                score += 0.1
        else:
            score -= 1.0  # Strong penalty for missing format

        rewards.append(score)

    return rewards


def mnemonic_contains_term_no_acronyms(completions, term, **kwargs):
    """Evaluates term inclusion in mnemonic and penalizes acronyms."""
    rewards = []

    for i, completion in enumerate(completions):
        response = completion[0]["content"]
        score = 0.0
        current_term = term[i].lower() if i < len(term) else ""

        mnemonic_text = extract_mnemonic(response)

        if mnemonic_text:
            # Check if term appears in the mnemonic
            if current_term and current_term in mnemonic_text:
                score += 1.5
            else:
                score -= 1.0

            # Check for acronyms
            acronym_matches = ACRONYM_REGEX.findall(response)
            if acronym_matches:
                score -= 1.0

        rewards.append(score)

    return rewards


def contains_linguistic_feature(completions, **kwargs):
    """Reward function that scores whether reasoning and mnemonic contain linguistic features.

    Args:
        completions: List of model completion objects
        **kwargs: Additional arguments

    Returns:
        List of reward scores
    """
    rewards = []
    for completion in completions:
        response = completion[0]["content"].lower()
        score = 0.0

        # Simply count the linguistic features present
        feature_count = 0
        for feature in LINGUISTIC_FEATURES:
            if feature in response:
                feature_count += 1

        # Optimal: 1-2 features
        if 1 <= feature_count <= 3:
            score = 1.0
        # Good: 3 features
        elif feature_count == 4:
            score = 0.5
        # Too many features: diminishing returns
        elif feature_count > 3:
            score = 0.2
        # No features: penalty
        else:
            score = -0.5

        rewards.append(score)

    return rewards
