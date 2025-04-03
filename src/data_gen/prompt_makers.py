"""Build prompts for the LLM as data generator, including prompts for few-shot and many-shot learning.

Usage:
    ```python
    from src.data_gen.prompt_makers import get_system_prompt

    prompt = get_system_prompt(prompt_path, learning_setting="few_shot")
    ```

    Set custom number of examples:
    ```python
    from src.data_gen.prompt_makers import get_system_prompt

    prompt = get_system_prompt(prompt_path, learning_setting="many_shot", num_examples=200)
    ```
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from structlog import getLogger

from src.data_prep.data_io import read_csv_file, read_json_file
from src.utils import constants as const
from src.utils.common import read_prompt
from src.utils.error_handlers import check_file_path

if TYPE_CHECKING:
    from typing import Any, Literal, Optional

    from structlog.stdlib import BoundLogger

    from src.utils.types import PathLike

# Set up logging
logger: BoundLogger = getLogger(__name__)


def get_system_prompt(
    prompt_path: PathLike,
    learning_setting: Literal["zero_shot", "few_shot", "many_shot"],
    **kwargs,
) -> str:
    """Read prompt from file.

    Args:
        prompt_path: Path to the prompt file.
        learning_setting: Learning setting for the prompt. Options are:
            - "zero_shot": No examples (0-shot learning)
            - "few_shot": Few examples (10-shot learning)
            - "many_shot": Many examples (100-shot learning)
        kwargs: Additional keyword arguments for get_system_prompt_examples. Accepts:
            - examples_path (PathLike): Path to the examples file.
            - num_examples (int): Number of examples to include in the prompt.

    Returns:
        str: Prompt string.

    Raises:
        ValueError: If the learning setting is not recognized.
    """
    num_example_dict = {
        "zero_shot": 0,
        "few_shot": 10,
        "many_shot": 100,
    }
    if "num_examples" not in kwargs and learning_setting not in num_example_dict.keys():
        raise ValueError(
            f"Learning setting '{learning_setting}' is not recognized, and 'num_examples' is not provided. "
            f"Please use one of {num_example_dict.keys()} or provide 'num_examples' in kwargs."
        )
    # Add num_examples to kwargs so it's passed to get_system_prompt_examples
    elif "num_examples" in kwargs:
        num_examples = kwargs["num_examples"]
    else:
        num_examples = num_example_dict[learning_setting]
        kwargs["num_examples"] = num_examples
    logger.debug(
        "Getting system prompt",
        prompt_path=prompt_path,
        learning_setting=learning_setting,
        num_examples=num_examples,
    )

    return get_system_prompt_examples(prompt_path, **kwargs)


def get_system_prompt_examples(
    prompt_path: PathLike,
    examples_path: Optional[PathLike] = None,
    num_examples: Optional[int] = 0,
) -> str:
    """Read prompt and add k examples to it, used for 0-shot, few-shot, and many-shot learning.

    Args:
        prompt_path: Path to the prompt file.
        examples_path: Path to the examples file.
        num_examples: Number of examples to include in the prompt. Default is 0, which means no examples (zero-shot). If num_examples is greater than the number of examples in the file, all examples will be included.
            num_examples = 0: zero-shot
            num_examples 0-50: few-shot
            num_examples > 100: many-shot

    Returns:
        str: Prompt with examples.
    """
    system_prompt = read_prompt(prompt_path)

    if num_examples < 0:
        logger.warning(
            f"Requested number of examples is negative ({num_examples}). Using 0 examples instead."
        )
        num_examples = 0

    if num_examples == 0:
        return system_prompt

    # If no examples path is provided, try common locations
    if examples_path is None:
        logger.debug("examples_path is None, attempting to find examples")
        possible_paths = [const.DATA_PATH.EXAMPLES_JSONL, const.DATA_PATH.EXAMPLES]

        # Check if examples files exist
        found_examples_path = None
        for path in possible_paths:
            try:
                if path.exists():
                    logger.debug("Found examples file", path=path)
                    found_examples_path = path
                    break
            except Exception as e:
                logger.warning("Error checking path", path=path, error=str(e))

        # If we didn't find a valid path, return just the system prompt
        if found_examples_path is None:
            logger.warning("No examples found. Using system prompt without examples.")
            return system_prompt

        # Set the found path to examples_path
        examples_path = found_examples_path

    # Now examples_path should not be None
    try:
        examples = load_examples(examples_path)
    except Exception as e:
        logger.warning(f"Failed to load examples from {examples_path}: {e}")
        return system_prompt

    actual_num_examples = len(examples)
    if num_examples > actual_num_examples:
        logger.warning(
            f"Requested number of examples exceeds available ({num_examples} > {actual_num_examples}). Usiing all available examples instead."
        )
        num_examples = actual_num_examples

    logger.info(
        "Examples loaded successfully",
        path=examples_path,
        size=actual_num_examples,
    )

    # Prepare the examples for inclusion in the prompt
    formatted_examples = [
        f"{i + 1}. {example}\n" for i, example in enumerate(examples[:num_examples])
    ]
    formatted_examples_str = "\n".join(formatted_examples)

    # Add the examples to the system prompt
    return system_prompt + "\n\nEXAMPLE SOLUTIONS:\n\n" + formatted_examples_str


def load_examples(
    examples_path: PathLike,
) -> list[dict[str, Any]]:
    """Load examples from a CSV or JSONL file.

    Args:
        examples_path: Path to the examples file.

    Returns:
        List of dictionaries containing the examples.
    """
    examples_path = check_file_path(
        examples_path,
        extensions=[const.Extension.CSV, const.Extension.JSONL],
    )
    if examples_path.suffix == const.Extension.CSV:
        return read_csv_file(examples_path, to_list_of_dicts=True)
    elif examples_path.suffix == const.Extension.JSONL:
        return read_json_file(examples_path)
    else:
        raise ValueError(
            "Examples file must be in CSV or JSONL format. "
            f"Got {examples_path.resolve()} instead."
        )


if __name__ == "__main__":
    prompt_path = const.PROMPT_PATH.REASON_SYSTEM

    prompt = get_system_prompt(prompt_path, learning_setting="few_shot")
    print(prompt[-1000:])
