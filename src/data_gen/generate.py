"""Usable functions for generating mnemonics and reasoning traces."""

from __future__ import annotations

from typing import TYPE_CHECKING

from datasets import Dataset, DatasetDict, concatenate_datasets, load_dataset
from structlog import getLogger

from src.data_gen.reasoner import reason
from src.data_prep.data_hf import load_txt_by_lines, push_data_to_hf
from src.data_prep.dedup import decontaminate, deduplicate
from src.utils import constants as const
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
def generate_mnemonics(
    reasoner_name: str,
    input_path: PathLike,
    excluded_terms: Optional[list[str]] = None,
    output_repo_id: Optional[str] = None,
    sample_size: Optional[int] = None,
    dry_run: bool = True,
) -> Dataset:
    """Generate mnemonics for vocabulary terms using OpenThoughts approach.

    Args:
        reasoner_name: Name of the reasoning model to use
        input_path: Path to input vocabulary dataset
        excluded_terms: Path to a file with terms to exclude from the dataset
        output_repo_id: Hugging Face repo ID to push results
        sample_size: Number of samples to process. If None, process all samples.
        dry_run: If True, run with minimal samples for testing

    Returns:
        Dataset with generated mnemonics and reasoning traces
    """
    # 1. Load vocabulary dataset
    ds = load_txt_by_lines(input_path, sample_size=sample_size)

    # 2. Deduplicate terms
    ds = deduplicate(ds)

    # 3. Decontaminate against potential test sets
    ds = decontaminate(ds, excluded_terms=excluded_terms)

    # 4. Prepare instructions
    ds = prepare_user_instructions(
        ds, instruction_prompt_path=const.PROMPT_PATH.REASON_USER
    )

    # 5. Generate reasoning and mnemonics
    ds = reason(ds, model_name=reasoner_name)

    # 7. Push to Hugging Face
    if not dry_run and not output_repo_id:
        raise ValueError(
            "Please provide an output_repo_id to push the dataset to Hugging Face."
        )
    elif output_repo_id:
        repo_id = f"{const.HF_CONST.USER}/{output_repo_id}"
        ds_dict = DatasetDict({"train": ds})

        if dry_run:
            logger.info("==== MNEMONIC DATASET (DRY RUN) ====")
            logger.info("Dataset summary:", ds_summary=ds)
            push_data_to_hf(ds_dict, repo_id, private=True)
        else:
            logger.info("==== MNEMONIC DATASET ====")
            push_data_to_hf(ds_dict, repo_id, private=False)

    elif not output_repo_id and dry_run:
        logger.info("==== MNEMONIC DATASET (DRY RUN) ====")
        logger.info("Dataset summary:", ds_summary=ds)
        logger.info("Dataset preview:", ds_preview=ds[0])

    return ds


if __name__ == "__main__":
    repo_id = "chiffonng/en-vocab-thoughts-mnemonics"
    old_dataset = load_dataset(repo_id, split="train")
    excluded_terms = old_dataset["term"]
    new_dataset = generate_mnemonics(
        reasoner_name="deepseek-reasoner",
        input_path="data/raw/train.txt",
        excluded_terms=excluded_terms,
        dry_run=False,
        sample_size=None,
    )
    combined_dataset = concatenate_datasets(
        [old_dataset, new_dataset["train"]], axis=0, info=old_dataset.info
    )
    combined_dataset = deduplicate(combined_dataset)
    combined_dataset.shuffle(seed=42).train_test_split(test_size=0.2).push_to_hub(
        "chiffonng/en-vocab-mnemonics",
        private=False,
        split={"train": "train", "val": "test"},
    )
