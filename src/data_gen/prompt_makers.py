"""Build prompts for the LLM as data generator, including prompts for few-shot and many-shot learning."""

from __future__ import annotations

from typing import TYPE_CHECKING

from structlog import getLogger

from src import constants as const
from src.data_prep.data_io import read_csv_file, read_json_file
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
    to_export: bool = True,
    **kwargs,
) -> str:
    """Read prompt from file.

    Args:
        prompt_path: Path to the prompt file.
        learning_setting: Learning setting for the prompt. Options are:
            - "zero_shot": No examples (0-shot learning)
            - "few_shot": Few examples (10-shot learning)
            - "many_shot": Many examples (100-shot learning)
        to_export: Whether to export the prompt to a file. Default is True.
        kwargs: Additional keyword arguments for get_system_prompt_examples. Accepts:
            - examples_path (PathLike): Path to the examples file.
            - num_examples (int): Number of examples to include in the prompt.

    Returns:
        str: Prompt string, or the path to the exported prompt file if to_export is True.

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

    prompt_path = check_file_path(
        prompt_path,
        extensions=[const.Extension.TXT],
    )
    logger.debug(
        "Getting system prompt",
        prompt_path=prompt_path,
        learning_setting=learning_setting,
        num_examples=num_examples,
    )

    # Get the system prompt with examples
    prompt_with_examples, num_examples = get_system_prompt_examples(
        prompt_path, **kwargs
    )

    if to_export and num_examples == 0:
        logger.info("No examples requested, not exporting prompt")
        return prompt_path

    elif to_export and num_examples > 0:
        export_path = prompt_path.parent / f"{prompt_path.stem}_{num_examples}_shot.txt"

        with export_path.open("w", encoding="utf-8") as f:
            f.write(prompt_with_examples)
        logger.info("Prompt exported", path=export_path.resolve())

        # Add export_path to constants.py
        setattr(const.PROMPT_PATH, f"REASON_SYSTEM_{num_examples}SHOT", export_path)

        return export_path

    return get_system_prompt_examples(prompt_path, **kwargs)


def get_system_prompt_examples(
    prompt_path: PathLike,
    examples_path: Optional[PathLike] = None,
    num_examples: Optional[int] = 0,
) -> tuple[str, int]:
    """Read prompt and add k examples to it, used for 0-shot, few-shot, and many-shot learning.

    Args:
        prompt_path: Path to the prompt file.
        examples_path: Path to the examples file.
        num_examples: Number of examples to include in the prompt. Default is 0, which means no examples (zero-shot). If num_examples is greater than the number of examples in the file, all examples will be included.
            num_examples = 0: zero-shot
            num_examples 0-50: few-shot
            num_examples > 100: many-shot

    Returns:
        tuple[str, int]: Tuple containing the prompt string with examples AND the number of examples included.
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
    return (
        system_prompt + "\n\nEXAMPLE SOLUTIONS:\n\n" + formatted_examples_str,
        num_examples,
    )


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


def combine_prompt_examples(prompt_path: PathLike, examples_path: PathLike) -> str:
    """Combine the prompt with examples for a specific learning setting.

    Args:
        prompt_path: Path to the prompt file (.txt file)
        examples_path: Path to the examples file (.txt file)

    Returns:
        str: Combined prompt with examples.
    """
    # Load the system prompt
    system_prompt = read_prompt(prompt_path)

    # Load the examples
    examples = read_prompt(examples_path)

    # Combine the system prompt with the examples
    combined_prompt = system_prompt + "\n\nEXAMPLE SOLUTIONS:\n\n"
    for i, example in enumerate(examples):
        combined_prompt += f"{i + 1}. {example}\n"

    return combined_prompt


if __name__ == "__main__":
    prompt_path = const.PROMPT_PATH.REASON_SYSTEM

    few_shot_prompt_path = get_system_prompt(
        prompt_path, learning_setting="few_shot", to_export=True
    )
    many_shot_prompt_path = get_system_prompt(
        prompt_path, learning_setting="many_shot", to_export=True
    )
    full_shot_prompt_path = get_system_prompt(
        prompt_path, learning_setting="many_shot", to_export=True, num_examples=200
    )
