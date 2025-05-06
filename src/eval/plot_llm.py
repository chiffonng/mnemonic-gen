"""Plotting functions for LLM evaluation results. This module contains functions to visualize the results of LLM evaluations, including Likert scale distributions, boolean metric comparisons, and summary visualizations."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from typing import Optional

    from matplotlib.figure import Figure
    from pandas import DataFrame

    from src.utils.types import PathLike

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def create_likert_distribution(
    base_results: DataFrame,
    finetuned_results: DataFrame,
    metrics: list[str],
    match_column: str = "term",
    save_path: Optional[PathLike] = None,
) -> Figure:
    """Create stacked horizontal bar charts for Likert scale metrics.

    Args:
        base_results: DataFrame with base model results
        finetuned_results: DataFrame with finetuned model results
        metrics: List of metrics to plot
        match_column: Column to match examples between datasets
        save_path: Path to save the figure. If None, figure is not saved.

    Returns:
        The created figure object
    """
    # Ensure we have matching terms
    matched_terms = set(base_results[match_column]).intersection(
        set(finetuned_results[match_column])
    )
    base_df = base_results[base_results[match_column].isin(matched_terms)].sort_values(
        match_column
    )
    finetuned_df = finetuned_results[
        finetuned_results[match_column].isin(matched_terms)
    ].sort_values(match_column)

    # Filter metrics to Likert scale metrics
    likert_metrics = [
        m
        for m in metrics
        if m in base_df.columns
        and m in finetuned_df.columns
        and base_df[m].dtype in ["int64", "float64"]
        and not base_df[m].dtype == "bool"
    ]

    if not likert_metrics:
        raise ValueError("No valid Likert scale metrics found for plotting")

    # Get possible Likert values
    all_values = set()
    for metric in likert_metrics:
        all_values.update(base_df[metric].unique())
        all_values.update(finetuned_df[metric].unique())

    likert_values = sorted(all_values)

    # Set up the figure
    fig, ax = plt.subplots(
        figsize=(8, len(likert_metrics) * 0.8 + 1), constrained_layout=True
    )
    # Set up colors for different scores - from red to green with score 3 being gray
    colors = {1: "tab:red", 2: "lightcoral", 3: "gray", 4: "lightgreen", 5: "tab:green"}

    # Prepare data for plotting
    y_labels = []
    percentages_data = []

    for metric in likert_metrics:
        # Get data distributions for base model
        base_counts = (
            base_df[metric].value_counts().reindex(likert_values, fill_value=0)
        )
        base_pct = base_counts / base_counts.sum() * 100

        # Get data distributions for finetuned model
        finetuned_counts = (
            finetuned_df[metric].value_counts().reindex(likert_values, fill_value=0)
        )
        finetuned_pct = finetuned_counts / finetuned_counts.sum() * 100

        # Add to our data structure
        y_labels.extend([f"{metric} (base)", f"{metric} (links)"])
        percentages_data.extend([base_pct, finetuned_pct])

    # Create the plot - working bottom to top
    y_pos = np.arange(len(y_labels))

    # For each likert value, create a segment in the stacked bar
    left_positions = np.zeros(len(y_labels))

    # For the legend
    legend_handles = []
    legend_labels = []

    for score in likert_values:
        # Get percentages for this score for all metrics and models
        score_percentages = [data[score] for data in percentages_data]

        # Create bar segment
        bars = ax.barh(
            y_pos,
            score_percentages,
            left=left_positions,
            color=colors.get(score, "black"),
            edgecolor="white",
            height=0.7,
        )

        # Add a handle for the legend
        legend_handles.append(bars[0])
        legend_labels.append(f"Score {score}")

        # Update left positions for next segment
        left_positions = left_positions + score_percentages

        # Add percentage labels inside the bars if there's enough space
        for i, (p, left) in enumerate(
            zip(score_percentages, left_positions - score_percentages, strict=False)
        ):
            if p >= 5:  # Only add text if percentage is big enough to be visible
                # Position the text in the middle of the segment
                text_x = left + p / 2

                # Add the text with percentage
                ax.text(
                    text_x,
                    y_pos[i],
                    f"{p:.1f}%",
                    va="center",
                    ha="center",
                    color="black" if score == 3 else "white",
                    fontweight="bold" if p > 20 else "normal",
                )

    # Customize the plot
    ax.set_yticks(y_pos)
    ax.set_yticklabels(y_labels)
    ax.set_xlabel("Percentage (%)")
    ax.set_xlim(0, 100)

    # Add grid lines
    ax.grid(axis="x", linestyle="--", alpha=0.3)

    # Add legend
    ax.legend(
        legend_handles,
        legend_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.1),
        ncol=len(likert_values),
    )

    # Add title
    # plt.suptitle("Likert Scale Distributions by Metric and Model", fontsize=16)

    # Adjust layout to make room for legend
    plt.tight_layout(rect=(0, 0.1, 1, 0.95))

    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


def create_boolean_comparison(
    base_results: DataFrame,
    finetuned_results: DataFrame,
    boolean_metrics: list[str],
    match_column: str = "term",
    save_path: Optional[str] = None,
) -> Figure:
    """Create horizontal bar chart comparison for boolean metrics.

    Args:
        base_results: DataFrame with base model results
        finetuned_results: DataFrame with finetuned model results
        boolean_metrics: List of boolean metrics to visualize
        match_column: Column to match examples between datasets
        save_path: Path to save the figure. If None, figure is not saved.

    Returns:
        The created figure object
    """
    # Ensure we have matching terms
    matched_terms = set(base_results[match_column]).intersection(
        set(finetuned_results[match_column])
    )
    base_df = base_results[base_results[match_column].isin(matched_terms)].sort_values(
        match_column
    )
    finetuned_df = finetuned_results[
        finetuned_results[match_column].isin(matched_terms)
    ].sort_values(match_column)

    # Filter to boolean metrics
    valid_metrics = [
        m
        for m in boolean_metrics
        if m in base_df.columns
        and m in finetuned_df.columns
        and base_df[m].dtype == "bool"
    ]

    if not valid_metrics:
        raise ValueError("No valid boolean metrics found for plotting")

    # Create figure (width optimized for single column in two-column LaTeX)
    # Typical column width in two-column LaTeX is about 3.3 inches
    fig, ax = plt.subplots(figsize=(4, len(valid_metrics)), constrained_layout=True)

    # Colors for the different categories (with consistent legend order)
    colors = {
        "both_false": "lightgray",
        "base_only": "tab:blue",
        "finetuned_only": "tab:orange",
        "both_true": "tab:green",
    }

    # Prepare data for plotting
    y_positions = np.arange(len(valid_metrics))

    # Store data for annotations
    all_metrics_data = []

    # Create bars for each metric
    for i, metric in enumerate(valid_metrics):
        # Calculate contingency table
        b00 = sum((~base_df[metric]) & (~finetuned_df[metric]))  # Both False
        b01 = sum(
            (~base_df[metric]) & finetuned_df[metric]
        )  # Base False, Finetuned True
        b10 = sum(
            base_df[metric] & (~finetuned_df[metric])
        )  # Base True, Finetuned False
        b11 = sum(base_df[metric] & finetuned_df[metric])  # Both True

        # Total counts
        total = b00 + b01 + b10 + b11

        # Calculate percentages
        pct_both_false = b00 / total * 100
        pct_base_only = b10 / total * 100
        pct_finetuned_only = b01 / total * 100
        pct_both_true = b11 / total * 100

        # Store the data for annotations
        metric_data = {
            "metric": metric,
            "total": total,
            "both_false": pct_both_false,
            "base_only": pct_base_only,
            "finetuned_only": pct_finetuned_only,
            "both_true": pct_both_true,
            "base_true": pct_base_only + pct_both_true,
            "finetuned_true": pct_finetuned_only + pct_both_true,
            "disagreements": pct_base_only + pct_finetuned_only,
        }
        all_metrics_data.append(metric_data)

        # Plot horizontal stacked bar
        left = 0
        for category, color in colors.items():
            width = metric_data[category]
            ax.barh(
                i,
                width,
                left=left,
                color=color,
                edgecolor="white",
                height=0.6,
                label=category.replace("_", " ").title() if i == 0 else None,
            )

            # Add percentage labels if wide enough
            if width > 10:
                text_x = left + width / 2
                ax.text(
                    text_x,
                    i,
                    f"{width:.1f}%",
                    va="center",
                    ha="center",
                    color="black",
                    fontsize=8,
                    fontweight="bold",
                )

            left += width

    # Add metric names as y-tick labels
    ax.set_yticks(y_positions)
    ax.set_yticklabels([metric for metric in valid_metrics])

    # Set x-axis limits and label
    ax.set_xlim(0, 100)
    ax.set_xlabel("Percentage (%)")

    # Add grid lines
    ax.grid(axis="x", linestyle="--", alpha=0.3)

    # Add a legend outside the plot
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=4, fontsize=8)

    # Add model performance annotations
    for _, data in enumerate(all_metrics_data):
        # Add model performance comparison
        disagreements = data["disagreements"]
        if disagreements > 0:
            (
                "Finetuned better"
                if data["finetuned_only"] > data["base_only"]
                else "Base better"
                if data["base_only"] > data["finetuned_only"]
                else "Equal"
            )

            # Create text for disagreement details
            (
                f"Base: {data['base_true']:.1f}%, "
                f"Finetuned: {data['finetuned_true']:.1f}%"
            )

            # Add small annotation at the right side
            # ax.text(
            #     101,
            #     i,
            #     disagree_text,
            #     va="center",
            #     ha="left",
            #     fontsize=7,
            #     color="dimgray",
            # )

    # Add title
    # ax.set_title("Boolean Metrics Comparison", fontsize=10)

    # Adjust layout
    plt.tight_layout()

    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


if __name__ == "__main__":
    # Generate synthetic data for demonstration

    base_df = pd.read_csv("data/eval/it_model_evaluation.csv")
    rl_df = pd.read_csv("data/eval/rl_model_evaluation.csv")

    fig1 = create_likert_distribution(
        base_df,
        rl_df,
        ["association_score", "clarity_score", "memorability_score"],
        save_path="writeup/fig/likert_distribution.pdf",
    )

    fig2 = create_boolean_comparison(
        base_df,
        rl_df,
        ["use_correct", "is_linguistic_grounded"],
        save_path="writeup/fig/boolean_comparison.pdf",
    )

    plt.show()
