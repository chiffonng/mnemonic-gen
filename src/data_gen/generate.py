"""Usable functions for generating mnemonics and reasoning traces."""

from __future__ import annotations

from typing import TYPE_CHECKING

from datasets import Dataset, DatasetDict, concatenate_datasets, load_dataset
from structlog import getLogger

from src import constants as const
from src.data_gen.reasoner import reason
from src.data_prep.data_hf import load_txt_by_lines, push_data_to_hf
from src.data_prep.dedup import decontaminate, deduplicate
from src.utils.common import read_prompt

if TYPE_CHECKING:
    from typing import Optional

    from structlog.stdlib import BoundLogger

    from src.utils.types import PathLike

# Set up logging
logger: BoundLogger = getLogger(__name__)


def prepare_user_instructions(
    dataset: Dataset,
    instruction_prompt_path: PathLike,
) -> Dataset:
    """Add "instruction" column to the dataset based on a prompt file.

    Args:
        dataset (Dataset): Input dataset to prepare instructions for
        instruction_prompt_path (PathLike): Path to the instruction prompt file
        instruction_vars (Optional[dict[str, Any]]): Variables to substitute in the prompt

    Returns:
        Dataset with added instruction column
    """

    def map_row(row):
        """Map function to prepare user instructions for each row in the dataset."""
        try:
            # TODO: Sample prompts from a pool of prompts later.
            user_instruction_template = read_prompt(prompt_path=instruction_prompt_path)
            user_instruction = user_instruction_template.format(term=row["term"])
            return {"instruction": user_instruction}
        except Exception as e:
            logger.error(
                "Failed to prepare instruction for term",
                prompt_source=instruction_prompt_path,
                term=row["term"],
                error=str(e),
            )
            raise e

    return dataset.map(map_row)


# TODO: Add argparse and refactor to allow CLI usage
def generate_ds(
    reasoner_name: str,
    input_path: PathLike,
    excluded_terms: Optional[list[str]] = None,
    output_repo_id: Optional[str] = None,
    dry_run: bool = True,
) -> Dataset | DatasetDict:
    """Generate mnemonics for vocabulary terms using OpenThoughts approach.

    Args:
        reasoner_name: Name of the reasoning model to use
        input_path: Path to input vocabulary dataset
        excluded_terms: Path to a file with terms to exclude from the dataset
        output_repo_id: Hugging Face repo ID to push results. <user>/<repo>
        dry_run: If True, run with minimal samples for testing, and return Dataset.

    Returns:
        Dataset or DatasetDict: Generated mnemonics dataset
    """
    # Load vocabulary dataset
    ds = load_txt_by_lines(input_path)

    # Deduplicate terms
    ds = deduplicate(ds)

    # Decontaminate against potential test sets
    ds = decontaminate(ds, excluded_terms=excluded_terms)

    # Prepare instructions
    ds = prepare_user_instructions(
        ds, instruction_prompt_path=const.PROMPT_PATH.REASON_USER
    )

    # Generate reasoning and mnemonics
    ds = reason(ds, model_name=reasoner_name)

    # Push to Hugging Face
    if not dry_run and not output_repo_id:
        raise ValueError(
            "Please provide an output_repo_id to push the dataset to Hugging Face."
        )
    elif output_repo_id and not dry_run:
        # Split into train and val sets
        # TODO: Stratify by linguistic features
        ds_dict: DatasetDict = ds.train_test_split(test_size=0.2, seed=42, shuffle=True)
        ds_dict = ds_dict.rename_column("test", "val")

        if dry_run:
            logger.info("==== MNEMONIC DATASET (DRY RUN) ====")
            logger.info("Dataset summary:", ds_summary=ds_dict)
            push_data_to_hf(ds_dict, output_repo_id, private=True)
        else:
            logger.info("==== MNEMONIC DATASET ====")
            push_data_to_hf(ds_dict, output_repo_id, private=False)
        return ds_dict

    elif not output_repo_id and dry_run:
        logger.info("==== MNEMONIC DATASET (DRY RUN) ====")
        logger.info("Dataset summary:", ds_summary=ds)
        logger.info("Dataset preview:", ds_preview=ds[0])

    return ds


if __name__ == "__main__":
    structured_ds_repo_id = const.HF_CONST.DATASET_NAME
    rl_ds_repo_id = const.HF_CONST.RL_DATASET_NAME

    structured_ds = generate_ds(
        reasoner_name="deepseek-reasoner",
        input_path="data/raw/train.txt",
        # output_repo_id=structured_ds_repo_id,
        dry_run=True,
    )
