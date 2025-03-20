"""Sample data from a DataFrame based on stratified sampling."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd
import structlog

if TYPE_CHECKING:
    from typing import Optional

logger = structlog.getLogger(__name__)


def stratify_by_column(
    df: pd.DataFrame,
    col_name_to_stratify: str,
    sample_size: int,
    min_samples_per_type: Optional[int] = None,
    seed: Optional[int] = 42,
) -> pd.DataFrame:
    """Stratified sampling of df by col_name_to_stratify, with at least min_samples_per_type samples per type.

    Args:
        df: DataFrame containing at least a col_name_to_stratify column.
        col_name_to_stratify: Column name to stratify by.
        sample_size: Total number of samples desired.
        min_samples_per_type: Minimum samples per type. If not provided, will use
        sample_size // num_types.
        seed: Random seed for reproducibility.

    Returns:
        A DataFrame sampled accordingly.
    """
    # Make sure df is a DataFrame and not empty.
    if not isinstance(df, pd.DataFrame) or df.empty:
        raise ValueError("Input df must be a non-empty DataFrame.")

    # Make sure df has the column to stratify by.
    if col_name_to_stratify not in df.columns:
        raise ValueError(f"Column '{col_name_to_stratify}' not found in DataFrame.")

    # Ensure sample_size is a positive integer.
    if not isinstance(sample_size, int) or sample_size <= 0:
        raise ValueError("sample_size must be a positive integer.")

    if sample_size > len(df):
        logger.warning(
            f"Requested sample size {sample_size} is larger than the DataFrame size {len(df)}. "
            "Returning the entire DataFrame."
        )
        return df.copy()

    # Guarantee at least (sample_size / num_types) samples per type.
    types = df[col_name_to_stratify].dropna().unique()
    num_types = len(types)
    baseline = int(sample_size // num_types)

    # If min_samples_per_type is provided, ensure it's not larger than baseline.
    if min_samples_per_type is None:
        min_samples_per_type = baseline
    elif min_samples_per_type > baseline:
        logger.warning(
            f"min_samples_per_type {min_samples_per_type} is larger than baseline {baseline}. Setting min_samples_per_type to baseline {baseline}."
        )
        min_samples_per_type = baseline

    # Sample at least min_samples_per_type from each type.
    guaranteed_sample = (
        df.groupby(col_name_to_stratify)
        .apply(
            lambda x: x.sample(min(len(x), min_samples_per_type), random_state=seed),
            include_groups=False,
        )
        .reset_index(drop=True)
    )

    # Randomly sample the remaining needed rows from the leftover data.
    remaining_sample_size = sample_size - len(guaranteed_sample)
    remaining_df = df.drop(guaranteed_sample.index)

    if remaining_sample_size > 0 and not remaining_df.empty:
        remaining_sample = remaining_df.sample(
            n=remaining_sample_size, random_state=seed
        )
    else:
        remaining_sample = pd.DataFrame()

    # Combine the guaranteed and remaining samples, shuffle.
    sampled_df = pd.concat([guaranteed_sample, remaining_sample], ignore_index=True)
    sampled_df.sample(frac=1, random_state=seed).reset_index(drop=True)

    return sampled_df
