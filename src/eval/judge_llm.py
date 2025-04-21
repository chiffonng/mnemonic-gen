"""Evaluate reasoning traces and mnemonic quality using a judge model."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from bespokelabs import curator
from pydantic import BaseModel, Field
from structlog import getLogger

from src.utils.common import read_prompt

if TYPE_CHECKING:
    from typing import Optional

    from datasets import Dataset
    from pandas import DataFrame
    from structlog.stdlib import BoundLogger

    from src.utils.types import PathLike

logger: BoundLogger = getLogger(__name__)


class JudgeResult(BaseModel):
    """Result of the judge's evaluation of a mnemonic."""

    use_correct: bool = Field(
        ..., description="Whether the term is used correctly in the mnemonic."
    )
    is_linguistic_grounded: bool = Field(
        ...,
        description="Whether the mnemonic is linguistically grounded, leveraging linguistic features like etymology, morphology, semantics, phonetics, orthography, etc.",
    )
    association_score: int = Field(
        ...,
        description="Score from 1-5 rating how strongly the mnemonic is associated with the term.",
    )
    clarity_score: int = Field(
        ...,
        description="Score from 1-5 rating how clear and understandable the mnemonic is.",
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
            "judge_reasoning": response.reasoning,
            "use_correct": response.use_correct,
            "is_linguistic_grounded": response.is_linguistic_grounded,
            "association_score": response.association_score,
            "clarity_score": response.clarity_score,
            "memorability_score": response.memorability_score,
        }


def judge(
    ds: Dataset,
    model_name: str = "o3-mini",
    save_dir: Optional[PathLike] = "outputs/judge_llm",
) -> DataFrame:
    """Evaluate a dataset of mnemonics using the Judge model.

    Args:
        ds (Dataset): The dataset containing mnemonics to be evaluated.
        model_name (str): The name of the judge model to use.
        save_dir (PathLike): Directory to save the evaluation results.

    Returns:
        Dataset: The original dataset with added evaluation metrics.
    """
    # Initialize the judge model
    judge_model = MnemonicJudge(model_name=model_name)

    evaluations: Dataset = judge_model(ds)

    evaluations_df = evaluations.to_pandas()

    if save_dir:
        save_path = Path(save_dir) / model_name
        save_path.parent.mkdir(parents=True, exist_ok=True)

        # Save the evaluations to a file
        evaluations_df.to_csv(save_path, index=False)
        logger.info(f"Saved evaluations to {save_dir}")

    return evaluations_df
