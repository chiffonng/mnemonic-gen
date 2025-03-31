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
    relevance_score: int = Field(
        ...,
        description="Score from 1-10 rating how relevant the mnemonic is to the vocabulary",
        ge=1,
        le=10,
    )
    linguistic_score: int = Field(
        ...,
        description="Score from 1-10 rating the linguistic richness",
        ge=1,
        le=10,
    )
    memorability_score: int = Field(
        ...,
        description="Score from 1-10 rating how memorable the mnemonic is.",
        ge=1,
        le=10,
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
            **input,
            "correct": response.correct,
            "judge_reasoning": response.reasoning,
            "relevance_score": response.relevance_score,
            "linguistic_score": response.linguistic_score,
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
    judge_model = MnemonicJudge()

    evaluations = judge_model(ds)

    return evaluations.to_pandas()
