"""Evaluate reasoning traces and mnemonic quality using a judge model."""

from __future__ import annotations

from typing import TYPE_CHECKING

from bespokelabs import curator
from pydantic import BaseModel, Field
from structlog import getLogger

from src.utils.common import read_prompt

if TYPE_CHECKING:
    from datasets import Dataset
    from structlog.stdlib import BoundLogger

logger: BoundLogger = getLogger(__name__)


class JudgeResult(BaseModel):
    """Result of the judge's evaluation of a mnemonic."""

    correct: bool = Field(
        ..., description="Whether the vocabulary is used correctly in the mnemonic."
    )
    is_ling_grounded: bool = Field(
        ...,
        description="Whether the mnemonic is linguistically grounded.",
    )
    association_score: int = Field(
        ...,
        description="Score from 1-5 rating how strongly the mnemonic relates to the term.",
    )
    simplicity_score: int = Field(
        ...,
        description="Score from 1-5 rating how simple the mnemonic is.",
    )
    memorability_score: int = Field(
        ...,
        description="Score from 1-5 rating how memorable the mnemonic is.",
    )
    reasoning: str = Field(..., description="Explanation of the evaluation.")


class MnemonicJudge(curator.LLM):
    """Judge class for evaluating mnemonics."""

    response_format = JudgeResult

    def prompt(self, input):
        """Create a prompt for the judge to evaluate mnemonic quality."""
        return read_prompt(
            regex_pattern=r"*judge*system\.txt",
            vars={
                "term": input["term"],
                "mnemonic": input["mnemonic"],
                "reasoning": input["reasoning"],
            },
        )

    def parse(self, input, response):
        """Parse the judge's response to extract evaluation metrics."""
        return {
            "term": input["term"],
            "mnemonic": input["mnemonic"],
            # Extract the evaluation metrics from the response
            "correct": response.correct,
            "judge_reasoning": response.reasoning,
            "is_ling_grounded": response.is_ling_grounded,
            "association_score": response.association_score,
            "simplicity_score": response.simplicity_score,
            "memorability_score": response.memorability_score,
        }


def judge(ds: Dataset):
    """Evaluate a dataset of mnemonics using the Judge model.

    Args:
        ds (Dataset): The dataset containing mnemonics to be evaluated.

    Returns:
        Dataset: The original dataset with added evaluation metrics.
    """
    # Initialize the judge model
    judge_model = MnemonicJudge(model_name="o3-mini", backend="openai")

    evaluations = judge_model(ds)

    return evaluations.to_pandas()
