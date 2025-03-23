"""Fine-tune OpenAI model with seed mnemonics.

Examples:
    Run the script with default parameters
    ```bash
    python scripts/_finetune_openai.py
    ```

    Run the script with custom parameters
    ```bash
    python scripts/_finetune_openai.py --to-prepare --to-upload --skip-finetune --train-file-path train.jsonl --val-file-path val.jsonl --split-ratio 0.8 --seed 42 --finetune-config-path config.json --completion-config-path completion_config.json
    ```
"""

from __future__ import annotations

import argparse
from typing import TYPE_CHECKING

from dotenv import load_dotenv
from openai import OpenAI
from src._mnemonic_enhancer.mnemonic_ft import (
    prepare_finetune_data,
    split_export_finetune_data,
    upload_finetune_data,
)
from src.llms.openai.openai_ft import finetune_from_config
from src.utils import constants as const
from structlog import getLogger

if TYPE_CHECKING:
    from structlog.stdlib import BoundLogger

# Set up logging
logger: BoundLogger = getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Prepare finetune data and run finetuning with OpenAI."
    )
    parser.add_argument(
        "--to-prepare",
        action="store_true",
        help="If set, prepare finetune data (default: False, use existing data)",
    )
    parser.add_argument(
        "--to-upload",
        action="store_true",
        help="If set, upload finetune data to OpenAI (default: False, skip upload)",
    )
    parser.add_argument(
        "--skip-finetune",
        action="store_true",
        default=False,
        help="If set, skip the fine-tuning step (default: False, run fine-tuning)",
    )
    parser.add_argument(
        "--train-file-path",
        type=str,
        default=const.SFT_IMPROVE_TRAIN,
        help="Path to output training JSONL file",
    )
    parser.add_argument(
        "--val-file-path",
        type=str,
        default=const.SFT_IMPROVE_VAL,
        help="Path to output validation JSONL file",
    )
    parser.add_argument(
        "--split-ratio",
        type=float,
        default=0.8,
        help="Split ratio for train/validation (0.0-1.0)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--finetune-config-path",
        type=str,
        default=const.CONF_OPENAI_SFT,
        help="Path to OpenAI finetuning config file",
    )
    parser.add_argument(
        "--completion-config-path",
        type=str,
        default=const.CONF_OPENAI_SFT_COMPLETION,
        help="Path to OpenAI completion config file",
    )
    return parser.parse_args()


def main():
    """Run the full pipeline for fine-tuning a model with improved mnemonics."""
    args = parse_args()
    logger.debug("Parsed arguments", args=vars(args))

    if args.to_prepare:
        logger.info("Preparing finetune data...")
        data = prepare_finetune_data(
            input_path=const.SEED_IMPROVED_CSV,
            system_prompt_path=const.FILE_PROMPT_IMPROVE_SFT_SYSTEM,
            user_prompt_path=const.FILE_PROMPT_USER,
        )
        split_export_finetune_data(
            data,
            output_train_jsonl=args.train_file_path,
            output_val_jsonl=args.val_file_path,
            split_ratio=args.split_ratio,
            seed=args.seed,
        )
    else:
        logger.info("Skipping data preparation step.")

    load_dotenv()
    client = OpenAI()
    if args.to_upload:
        logger.info("Uploading files to OpenAI...")
        upload_finetune_data(
            client,
            input_path=args.train_file_path,
            config_file_path=args.finetune_config_path,
            file_type="train",
        )
        upload_finetune_data(
            client,
            input_path=args.val_file_path,
            config_file_path=args.finetune_config_path,
            file_type="val",
        )
    else:
        logger.info("Skipping file upload step.")

    if not args.skip_finetune:
        logger.info("Starting fine-tuning process...")
        finetune_from_config(
            client,
            finetune_config_path=args.finetune_config_path,
            completion_config_path=args.finetune_config_path,
        )
    else:
        logger.info("Skipping fine-tuning step.")


if __name__ == "__main__":
    main()
