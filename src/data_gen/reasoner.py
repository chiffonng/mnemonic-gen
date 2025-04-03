"""Module for generating reasoning traces for mnemonic generation."""

from __future__ import annotations

from typing import TYPE_CHECKING

from bespokelabs import curator
from structlog import getLogger

from src.data_gen.models import MnemonicResult
from src.data_gen.prompt import get_system_prompt
from src.utils import constants as const
from src.utils.common import read_config, read_prompt

if TYPE_CHECKING:
    from typing import Any

    from datasets import Dataset
    from structlog.stdlib import BoundLogger

# Set up logging
logger: BoundLogger = getLogger(__name__)


class DeepSeekReasoner(curator.LLM):
    """class for generating reasoning traces for mnemonics."""

    return_completions_object = True

    # TODO: add learning setting to the model
    def prompt(self, input: dict[str, str]) -> list[dict[str, Any]]:
        """Create a prompt for the LLM to reason about the vocab and user input.

        Args:
            input: Dictionary containing the input data for reasoning

        Returns:
            List of dictionaries containing the prompt for the LLM
        """
        return [
            {
                "role": "system",
                "content": read_prompt(const.PROMPT_PATH.REASON_SYSTEM),
            },
            {"role": "user", "content": input["instruction"]},
        ]

    def parse(self, input: dict, response: dict[str, str]) -> dict[str, Any]:
        """Parse the LLM response to extract reasoning and solution."""
        return {
            "term": input["term"],  # The term being reasoned about
            "instruction": input["instruction"],
            "reasoning": response["choices"][0]["message"]["reasoning_content"],
            "solution": response["choices"][0]["message"]["content"],
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
            {
                "role": "system",
                "content": read_prompt(const.PROMPT_PATH.REASON_SYSTEM),
            },
            {"role": "user", "content": input["instruction"]},
        ]

    def parse(self, input: dict, response: dict[str, str]) -> dict[str, Any]:
        """Parse the LLM response to extract reasoning and solution."""
        return {
            "term": input["term"],  # The term being reasoned about
            "instruction": input["instruction"],
            "reasoning": response.reasoning,
            "solution": response.solution,
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

    if model_name == "deepseek-reasoner":
        reasoner = DeepSeekReasoner(
            model_name="deepseek/deepseek-reasoner",
            generation_params=default_generation_params.update(
                read_config(const.CONFIG_PATH.DEEPSEEK_REASONER)
            ),
            backend_params=read_config(const.CONFIG_PATH.DEFAULT_BACKEND),
        )

    elif model_name == "o3-mini":
        reasoner = O3MiniReasoner(
            model_name="openai/o3-mini",
            batch=True,
            generation_params=default_generation_params.update(
                read_config(const.CONFIG_PATH.OPENAI)
            ),
            backend_params=read_config(const.CONFIG_PATH.DEFAULT_BACKEND_BATCH),
        )
    else:
        raise ValueError(f"Unknown model name: {model_name}")
    return reasoner(ds)
