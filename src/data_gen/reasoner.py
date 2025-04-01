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
    from pydantic import BaseModel
    from structlog.stdlib import BoundLogger

# Set up logging
logger: BoundLogger = getLogger(__name__)


class DeepSeekReasoner(curator.LLM):
    """Reasoner class for generating reasoning traces for mnemonics."""

    return_completion_objects = True

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
        default_generation_params = read_config(
            const.CONFIG_FILES["DEFAULT_GENERATION"]
        )
        default_backend_params = read_config(const.CONFIG_FILES["DEFAULT_BACKEND"])

        if generation_params is None:
            # Search for config "something-deepseek-something.json" in the config directory
            generation_params = read_config(regex_pattern=r".*deepseek.*\.json")
            default_generation_params.update(generation_params)
            generation_params = default_generation_params

        if backend_params is None:
            backend_params = default_backend_params
        else:
            default_backend_params.update(backend_params)
            backend_params = default_backend_params

        super().__init__(
            model_name=model_name,
            backend=backend,
            batch=batch,
            generation_params=generation_params,
            backend_params=backend_params,
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
                "content": read_prompt(const.PROMPT_FILES["REASON_SYSTEM"]),
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


class ClaudeReasoner(curator.LLM):
    """Reasoner class for generating reasoning traces using Claude."""

    return_completion_objects = True

    def __init__(
        self,
        model_name: str,
        backend: str = "anthropic",
        batch: bool = True,
        generation_params: Optional[dict[str, Any]] = None,
        backend_params: Optional[dict[str, Any]] = None,
    ):
        """Initialize the Reasoner class for getting reasoning traces using Claude.

        Args:
            model_name: Name of the LLM model to use
            backend: Backend to use for the LLM (default is "anthropic")
            batch: Whether to process instructions in batches (default is True)
            generation_params: Parameters for text generation
            backend_params: Parameters for backend configuration
        """
        default_generation_params = read_config(
            const.CONFIG_FILES["DEFAULT_GENERATION"]
        )
        default_backend_params = read_config(const.CONFIG_FILES["DEFAULT_BACKEND"])

        if generation_params is None:
            # Search for config "anything-claude_anything.json" in the config directory
            generation_params = read_config(regex_pattern=r"claude.*\.json")
            default_generation_params.update(generation_params)
            generation_params = default_generation_params

        if backend_params is None:
            backend_params = default_backend_params
        else:
            default_backend_params.update(backend_params)
            backend_params = default_backend_params

        super().__init__(
            model_name=model_name
            or default_generation_params.get(
                "model_name", "claude-3-7-sonnet-20250219"
            ),
            backend=backend,
            batch=batch,
            generation_params=generation_params,
            backend_params=backend_params,
        )

    def prompt(self, input):
        """Create a prompt for the LLM to reason about the vocab and user input."""
        return [
            {
                "role": "system",
                "content": read_prompt(const.PROMPT_FILES["REASON_SYSTEM"]),
            },
            {"role": "user", "content": input["instruction"]},
        ]

    def parse(self, input, response):
        """Parse the LLM response to extract reasoning and solution."""
        content = response["content"]
        thinking = ""
        text = ""
        for content_block in content:
            if content_block["type"] == "thinking":
                thinking = content_block["thinking"]
            elif content_block["type"] == "text":
                text = content_block["text"]
            elif content_block["type"] == "redacted_thinking":
                print("Redacted thinking block! (notifying you for fun)")

        input["claude_thinking"] = thinking
        input["claude_attempt"] = text
        return input


def reason(ds: Dataset, reasoner: curator.LLM = DeepSeekReasoner) -> BaseModel:
    """Generate reasoning traces using the DeepSeekReasoner."""
    return reasoner(ds)
