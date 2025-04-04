"""Module for generating reasoning traces for mnemonic generation."""

from __future__ import annotations

from typing import TYPE_CHECKING

from bespokelabs import curator
from structlog import getLogger

from src import constants as const
from src.data_gen.models import MnemonicResult
from src.utils.common import read_config, read_prompt

if TYPE_CHECKING:
    from typing import Any

    from datasets import Dataset
    from structlog.stdlib import BoundLogger

# Set up logging
logger: BoundLogger = getLogger(__name__)

SYSTEM_10SHOT_PROMPT = read_prompt("prompts/reason/system_10_shot.txt")
SYSTEM_100SHOT_PROMPT = read_prompt("prompts/reason/system_100_shot.txt")


class DeepSeekReasoner(curator.LLM):
    """class for generating reasoning traces for mnemonics."""

    return_completions_object = True

    def prompt(self, input: dict[str, str]) -> list[dict[str, Any]]:
        """Create a prompt for the LLM to reason about the vocab and user input.

        Args:
            input: Dictionary containing the input data for reasoning

        Returns:
            List of dictionaries containing the prompt for the LLM
        """
        return [
            {"role": "system", "content": SYSTEM_10SHOT_PROMPT},
            {"role": "user", "content": input["instruction"]},
        ]

    def parse(self, input: dict, response: dict[str, str]) -> dict[str, Any]:
        """Parse the LLM response to extract reasoning and solution."""
        return {
            "term": input["term"],  # The term being reasoned about
            "instruction": input["instruction"],
            "reasoning": response["choices"][0]["message"]["reasoning_content"],
            "answer": response["choices"][0]["message"]["content"],
        }


class O3MiniReasoner(curator.LLM):
    """Class for generating reasoning traces for mnemonics using O3 Mini."""

    response_format = MnemonicResult

    def prompt(self, input: dict[str, str]) -> list[dict[str, Any]]:
        """Create a prompt for the LLM to reason about the vocab and user input.

        Args:
            input: Dictionary containing the input data for reasoning

        Returns:
            List of dictionaries containing the prompt for the LLM
        """
        return [
            {"role": "system", "content": SYSTEM_100SHOT_PROMPT},
            {"role": "user", "content": input["instruction"]},
        ]

    def parse(self, input: dict, response: dict[str, str]) -> dict[str, Any]:
        """Parse the LLM response to extract reasoning and solution."""
        return {
            "term": input["term"],  # The term being reasoned about
            "instruction": input["instruction"],
            "reasoning": response.reasoning,
            "answer": response.answer,
        }


def reason(ds: Dataset, model_name: str = "deepseek-reasoner") -> Dataset:
    """Generate reasoning traces using the DeepSeekReasoner.

    Args:
        ds: Dataset containing the input data for reasoning
        model_name: Name of the reasoning model to use (default is "deepseek-reasoner")

    Returns:
        Dataset: Dataset with added reasoning traces and other fields
    """
    default_generation_params = read_config(const.CONFIG_PATH.DEFAULT_GENERATION)
    default_backend_params = read_config(const.CONFIG_PATH.DEFAULT_BACKEND)

    if model_name == "deepseek-reasoner":
        default_generation_params.update(
            read_config(const.CONFIG_PATH.DEEPSEEK_REASONER)
        )
        reasoner = DeepSeekReasoner(
            model_name="deepseek/deepseek-reasoner",
            generation_params=default_generation_params,
            backend_params=default_backend_params,
        )

    elif model_name == "o3-mini":
        default_generation_params.update(read_config(const.CONFIG_PATH.OPENAI))
        reasoner = O3MiniReasoner(
            model_name="openai/o3-mini",
            batch=True,
            generation_params=default_generation_params,
            backend_params=default_backend_params,
        )
    else:
        raise ValueError(f"Unknown model name: {model_name}")
    return reasoner(ds)
