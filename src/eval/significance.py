# src/eval/significance.py
"""Module for statistical significance tests on mnemonic evaluation data."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from scipy import stats
from structlog import getLogger

if TYPE_CHECKING:
    from typing import Optional

    from pandas import DataFrame, Series
    from structlog.stdlib import BoundLogger

logger: BoundLogger = getLogger(__name__)


def compare_likert_scores(
    base_scores: Series | np.ndarray,
    finetuned_scores: Series | np.ndarray,
    alpha: float = 0.05,
) -> tuple[float, float, str]:
    """Compare Likert scale scores between base and finetuned models.

    Args:
        base_scores: Scores from base model
        finetuned_scores: Scores from finetuned model
        alpha: Significance level

    Returns:
        Tuple containing:
        - p_value: p-value of the test
        - effect_size: Cohen's d effect size
        - interpretation: Human-readable interpretation of the result
    """
    # Check if data length matches
    if len(base_scores) != len(finetuned_scores):
        raise ValueError(
            f"Length mismatch: base_scores ({len(base_scores)}) vs "
            f"finetuned_scores ({len(finetuned_scores)})"
        )

    # Calculate difference
    diff = np.array(finetuned_scores) - np.array(base_scores)

    # Use Wilcoxon signed-rank test
    stat, p_value = stats.wilcoxon(finetuned_scores, base_scores)

    is_significant = p_value < alpha

    # Compute effect size - Cohen's d for paired samples
    effect_size = (
        np.mean(diff) / np.std(diff, ddof=1) if np.std(diff, ddof=1) != 0 else 0
    )

    # Create interpretation
    mean_diff = np.mean(finetuned_scores) - np.mean(base_scores)
    if is_significant:
        if mean_diff > 0:
            direction = "higher"
        else:
            direction = "lower"

        if abs(effect_size) < 0.2:
            magnitude = "negligible"
        elif abs(effect_size) < 0.5:
            magnitude = "small"
        elif abs(effect_size) < 0.8:
            magnitude = "medium"
        else:
            magnitude = "large"

        interpretation = (
            f"The finetuned model scored significantly {direction} than the base model "
            f"(p={p_value:.4f}). The effect size is {magnitude} (Cohen's d={effect_size:.2f})."
        )
    else:
        interpretation = (
            f"No significant difference between models (p={p_value:.4f}). "
            f"Mean difference: {mean_diff:.2f}, Cohen's d={effect_size:.2f}"
        )

    return p_value, effect_size, interpretation


def compare_boolean_outcomes(
    base_outcomes: Series | np.ndarray,
    finetuned_outcomes: Series | np.ndarray,
    alpha: float = 0.05,
) -> tuple[float, str]:
    """Compare boolean outcomes between base and finetuned models using McNemar's test.

    Args:
        base_outcomes: Boolean outcomes from base model
        finetuned_outcomes: Boolean outcomes from finetuned model
        alpha: Significance level

    Returns:
        Tuple containing:
        - p_value: p-value of the test
        - interpretation: Human-readable interpretation of the result
    """
    # Ensure inputs are proper boolean arrays
    base = np.array(base_outcomes, dtype=bool)
    finetuned = np.array(finetuned_outcomes, dtype=bool)

    # Create contingency table
    # b01: base=False, finetuned=True
    # b10: base=True, finetuned=False
    b01 = np.sum(~base & finetuned)
    b10 = np.sum(base & ~finetuned)

    # McNemar's test
    if b01 + b10 < 20:
        # Use exact binomial test for small samples
        p_value = stats.binomtest(min(b01, b10), b01 + b10, p=0.5).pvalue
    else:
        # Use chi-square approximation
        stat = ((b01 - b10) ** 2) / (b01 + b10)
        p_value = stats.chi2.sf(stat, df=1)

    is_significant = p_value < alpha

    # Create interpretation
    if is_significant:
        if b01 > b10:
            interpretation = (
                f"The finetuned model produced significantly more correct responses "
                f"than the base model (p={p_value:.4f}). "
                f"Improved in {b01} cases, declined in {b10} cases."
            )
        else:
            interpretation = (
                f"The base model produced significantly more correct responses "
                f"than the finetuned model (p={p_value:.4f}). "
                f"Improved in {b10} cases, declined in {b01} cases."
            )
    else:
        interpretation = (
            f"No significant difference in correctness between models (p={p_value:.4f}). "
            f"Finetuned better in {b01} cases, base better in {b10} cases."
        )

    return p_value, interpretation


def analyze_mnemonic_evaluation(
    base_results: DataFrame,
    finetuned_results: DataFrame,
    matched_by: str = "term",
    metrics: Optional[list[str]] = None,
) -> DataFrame:
    """Analyze differences between base and finetuned mnemonic evaluation results.

    Args:
        base_results: DataFrame with evaluation results from base model
        finetuned_results: DataFrame with evaluation results from finetuned model
        matched_by: Column to match results (default: "term")
        metrics: List of metrics to analyze. If None, uses all numeric and boolean columns
            except the matching column and model column.

    Returns:
        DataFrame with statistical test results for each metric
    """
    # Ensure we have matching terms in both dataframes
    matched_terms = set(base_results[matched_by]).intersection(
        set(finetuned_results[matched_by])
    )
    if len(matched_terms) == 0:
        raise ValueError(f"No matching terms found using column '{matched_by}'")

    # Filter to matching terms and sort to ensure alignment
    base_df = base_results[base_results[matched_by].isin(matched_terms)].sort_values(
        matched_by
    )
    finetuned_df = finetuned_results[
        finetuned_results[matched_by].isin(matched_terms)
    ].sort_values(matched_by)

    # If metrics not specified, use all numeric and boolean columns except matching column
    # remove any columns that are not in both dataframes, not of type int/float/bool
    if metrics is None:
        metrics = []
        for col in base_df.columns:
            if col != matched_by and base_df[col].dtype in ["int64", "float64", "bool"]:
                metrics.append(col)

    # Initialize results dataframe
    results = []

    # Analyze each metric
    for metric in metrics:
        # Skip if metric not in both dataframes
        if metric not in base_df.columns or metric not in finetuned_df.columns:
            logger.warning(f"Metric '{metric}' not found in both dataframes. Skipping.")
            continue

        if base_df[metric].dtype == bool:
            # Boolean metrics (e.g., correctness)
            p_value, interpretation = compare_boolean_outcomes(
                base_df[metric], finetuned_df[metric]
            )
            effect_size = None
        else:
            # Likert scale metrics
            p_value, effect_size, interpretation = compare_likert_scores(
                base_df[metric], finetuned_df[metric]
            )

        # Add to results
        results.append(
            {
                "metric": metric,
                "base_mean": base_df[metric].mean(),
                "linksys_mean": finetuned_df[metric].mean(),
                "mean_diff": finetuned_df[metric].mean() - base_df[metric].mean(),
                "base_std": base_df[metric].std(),
                "linksys_std": finetuned_df[metric].std(),
                "p_value": p_value,
                "effect_size": effect_size,
                "n_samples": len(matched_terms),
                "interpretation": interpretation,
            }
        )

    return pd.DataFrame(results)


if __name__ == "__main__":
    base_results = pd.read_csv("data/eval/it_model_evaluation.csv")
    finetuned_results = pd.read_csv("data/eval/rl_model_evaluation.csv")

    # Analyze all metrics
    results = analyze_mnemonic_evaluation(base_results, finetuned_results)

    # don't print last two columns
    print(results.drop(columns=["interpretation", "n_samples"]))

    # Save results to file
    results.to_csv("data/eval/significance_results.csv", index=False)
