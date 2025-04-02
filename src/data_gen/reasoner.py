"""Module for generating reasoning traces for mnemonic generation."""

from __future__ import annotations

from typing import TYPE_CHECKING

from bespokelabs import curator
from structlog import getLogger

from src.utils import constants as const
from src.utils.common import read_config, read_prompt

if TYPE_CHECKING:
    from typing import Any, Optional

    from datasets import Dataset
    from structlog.stdlib import BoundLogger

# Set up logging
logger: BoundLogger = getLogger(__name__)


class DeepSeekReasoner(curator.LLM):
    """Reasoner class for generating reasoning traces for mnemonics."""

    return_completions_object = True

    def __init__(
        self,
        model_name: str = "deepseek-reasoner",
        backend: str = "openai",
        batch: bool = True,
        generation_params: Optional[dict[str, Any]] = None,
        backend_params: Optional[dict[str, Any]] = None,
    ):
        """Initialize the DeepSeekReasoner class. See https://docs.bespokelabs.ai/bespoke-curator/api-reference/llm-api-documentation for more details.

        Args:
            model_name: Name of the LLM model to use
            backend: Backend to use for the LLM (default is "openai")
            batch: Whether to process instructions in batches (default is True)
            generation_params: Parameters for text generation
            backend_params: Parameters for backend configuration
        """
        default_generation_params = read_config(const.CONFIG_PATH.DEFAULT_GENERATION)

        if batch:
            default_backend_params = read_config(
                const.CONFIG_PATH.DEFAULT_BACKEND_BATCH
            )
        else:
            default_backend_params = read_config(const.CONFIG_PATH.DEFAULT_BACKEND)

        if generation_params is None:
            # Search for config "something-deepseek-something.json" in the config directory
            model_generation_params = read_config(regex_pattern=r".*deepseek.*\.json")
            default_generation_params.update(model_generation_params)
        else:
            # Update default generation params with the provided ones
            default_generation_params.update(generation_params)

        if backend_params:
            default_backend_params.update(backend_params)

        logger.debug(
            "DeepSeekReasoner initialized",
            model_name=model_name,
            backend=backend,
            batch=batch,
            generation_params=default_generation_params,
            backend_params=default_backend_params,
        )

        super().__init__(
            model_name=model_name,
            backend=backend,
            batch=batch,
            generation_params=default_generation_params,
            backend_params=default_backend_params,
        )

    def prompt(self, input: dict) -> list[dict[str, Any]]:
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

    def parse(self, input: dict, response: dict) -> dict[str, Any]:
        """Parse the LLM response to extract reasoning and solution."""
        return {
            "term": input["term"],  # The term being reasoned about
            "instruction": input["instruction"],
            "reasoning": response["choices"][0]["message"]["reasoning_content"],
            "completion": response["choices"][0]["message"]["content"],
        }


def reason(ds: Dataset) -> Dataset:
    """Generate reasoning traces using the DeepSeekReasoner.

    Args:
        ds: Dataset containing the input data for reasoning
    Returns:
        Dataset: Dataset with added reasoning traces and other fields
    """
    reasoner = DeepSeekReasoner(batch=True)
    return reasoner(ds)
