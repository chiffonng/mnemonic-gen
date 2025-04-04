"""Module for decontaminating vocab and reasoning traces, preventing leakage from training to test data."""

from __future__ import annotations

import multiprocessing as mp
from collections import defaultdict
from functools import partial
from multiprocessing import Pool
from typing import TYPE_CHECKING

from datasets import Dataset, load_dataset
from rapidfuzz import fuzz, process
from structlog import getLogger
from tqdm import tqdm

from src.utils import constants as const

if TYPE_CHECKING:
    from typing import Optional

    from structlog.stdlib import BoundLogger

logger: BoundLogger = getLogger(__name__)


def fuzz_string_pair(
    str1: str, values2: list[str], similarity_threshold: float
) -> list[tuple]:
    """Find fuzzy matches between a string and a list of strings.

    Args:
        str1: The string to match against
        values2: List of strings to compare with
        similarity_threshold: Minimum similarity score (0-100) to consider a match

    Returns:
        List of tuples containing (str1, matched_string, similarity_score)
    """
    matches_with_scores = process.extract(
        str1, values2, scorer=fuzz.ratio, score_cutoff=similarity_threshold
    )
    return [
        (str1, match_tuple[0], match_tuple[1]) for match_tuple in matches_with_scores
    ]


def deduplicate(
    dataset: Dataset, column="term", similarity_threshold: float = 95.0
) -> Dataset:
    """Deduplicate dataset rows based on fuzzy string matching within specified column.

    Args:
        dataset: Input dataset to deduplicate
        column: Column to check for duplicates
        similarity_threshold: Fuzzy matching threshold (0-100)

    Returns:
        Deduplicated dataset
    """
    # Extract all values from the specified column
    values = [str(x) for x in dataset[column] if x is not None]
    unique_values = list(set(values))

    # Set up multiprocessing for parallel fuzzy matching
    n_processes = mp.cpu_count()

    # Map values to their indices in the dataset
    str_to_indices = defaultdict(list)
    for i, val in enumerate(values):
        str_to_indices[val].append(i)

    # Create a partial function with the threshold
    process_pair = partial(
        fuzz_string_pair,
        values2=unique_values,
        similarity_threshold=similarity_threshold,
    )

    # Run fuzzy matching in parallel
    with Pool(n_processes) as pool:
        all_matches = list(
            tqdm(
                pool.imap(process_pair, unique_values, chunksize=100),
                total=len(unique_values),
                desc="Finding fuzzy duplicates",
            )
        )

    # Identify indices to remove (keeping only the first occurrence of similar items)
    indices_to_remove = set()

    for matches_list in all_matches:
        for str1, str2, score in matches_list:
            if str1 != str2 and score >= similarity_threshold:
                # Found a fuzzy match between str1 and str2
                indices1 = str_to_indices[str1]
                indices2 = str_to_indices[str2]

                # Keep the first occurrence, remove all others
                all_indices = list(set(indices1 + indices2))
                all_indices.sort()
                indices_to_remove.update(all_indices[1:])

    # Filter the dataset to keep only non-duplicate rows
    keep_mask = [i for i in range(len(dataset)) if i not in indices_to_remove]
    clean_dataset = dataset.select(keep_mask)

    logger.info(f"Removed {len(indices_to_remove)} fuzzy duplicate rows")
    logger.info(f"Original size: {len(dataset)}, New size: {len(clean_dataset)}")

    return clean_dataset


def decontaminate(
    dataset: Dataset, excluded_terms: Optional[list[str]] = None, column: str = "term"
) -> Dataset:
    """Decontaminate dataset against a list of terms to avoid.

    Args:
        dataset: Input dataset to decontaminate
        excluded_terms: List of terms to exclude
        column: Column to check for contamination
    Returns:
        Decontaminated dataset
    """
    if excluded_terms is None:
        excluded_terms = load_dataset(const.HF_CONST.TESTSET_NAME, split="test")

    # Create a mask for terms not in the test set
    mask = []
    for term in dataset[column]:
        if term in excluded_terms:
            mask.append(False)
        else:
            mask.append(True)

    # Filter the dataset based on the mask
    filtered_dataset = dataset.select([i for i, keep in enumerate(mask) if keep])
    logger.info(f"Removed {len(mask) - sum(mask)} contaminated terms")

    return filtered_dataset
