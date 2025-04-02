"""Build prompts for the LLM as data generator, including prompts for few-shot and many-shot learning."""

from __future__ import annotations

from typing import TYPE_CHECKING

from structlog import getLogger

from src.data_prep.data_io import read_csv_file, read_json_file, read_txt_file
from src.utils import constants as const
from src.utils.common import read_prompt

if TYPE_CHECKING:
    from typing import Literal, Optional

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
        kwargs: Additional keyword arguments for get_system_prompt_examples.

    Returns:
        str: Prompt string.

    Raises:
        ValueError: If the learning setting is not recognized.
    """
    if learning_setting == "zero_shot":
        return get_system_prompt_examples(prompt_path, num_examples=0, **kwargs)
    elif learning_setting == "few_shot":
        return get_system_prompt_examples(prompt_path, num_examples=10, **kwargs)
    elif learning_setting == "many_shot":
        return get_system_prompt_examples(prompt_path, num_examples=100, **kwargs)
    else:
        raise ValueError(
            f"Learning setting '{learning_setting}' is not recognized. "
            "Please use 'zero_shot', 'few_shot', or 'many_shot'."
        )


def get_system_prompt_examples(
    prompt_path: PathLike,
    examples_path: PathLike = const.DATA_PATH.EXAMPLES,
    num_examples: Optional[int] = 0,
) -> str:
    """Read prompt and add k examples to it, used for 0-shot, few-shot, and many-shot learning.

    Args:
        prompt_path: Path to the prompt file.
        examples_path: Path to the examples file.
        num_examples: Number of examples to include in the prompt. Default is 0, which means no examples (zero-shot). If num_examples is greater than the number of examples in the file, all examples will be included.
            num_examples = 0: zero-shot
            num_examples 0-100: few-shot
            num_examples > 100: many-shot

    Returns:
        str: Prompt with examples.
    """
    pass
