"""Fine-tune OpenAI model with seed mnemonics."""

from __future__ import annotations

import argparse
from typing import TYPE_CHECKING

from dotenv import load_dotenv
from openai import OpenAI
from src._mnemonic_enhancer.mnemonic_ft import (
    prepare_split_finetune_data,
    upload_finetune_data,
)
from src.llms.openai.openai_ft import finetune_from_config
from src.utils import constants as const
from structlog import getLogger

if TYPE_CHECKING:
    from src.utils.types import Mnemonic
    from structlog.stdlib import BoundLogger

# Set up logging
logger: BoundLogger = getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Prepare finetune data and run finetuning"
    )
    parser.add_argument(
        "--input-path",
        type=str,
        default=const.SEED_IMPROVED_CSV,
        help="Path to input CSV file",
    )
    parser.add_argument(
        "--system-prompt-path",
        type=str,
        default=const.FILE_PROMPT_IMPROVE_SFT_SYSTEM,
        help="Path to system prompt file",
    )
    parser.add_argument(
        "--user-prompt-path",
        type=str,
        default=const.FILE_PROMPT_IMPROVE_SFT_USER,
        help="Path to user prompt file",
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
        "--config-path",
        type=str,
        default=const.CONF_OPENAI_SFT,
        help="Path to OpenAI finetuning config file",
    )
    parser.add_argument(
        "--split-ratio",
        type=float,
        default=0.8,
        help="Split ratio for train/validation (0.0-1.0)",
    )
    parser.add_argument(
        "--skip-prepare",
        action="store_true",
        default=True,
        help="Skip data preparation and use existing files",
    )
    parser.add_argument(
        "--skip-upload",
        action="store_true",
        default=True,
        help="Skip file upload and use IDs in config",
    )
    parser.add_argument(
        "--skip-finetune",
        action="store_true",
        default=False,
        help="Skip finetuning step",
    )
    return parser.parse_args()


def main():
    """Run the full pipeline for fine-tuning a model with improved mnemonics."""
    args = parse_args()
    logger.debug("Parsed arguments", args=vars(args))

    if not args.skip_prepare:
        logger.info("Preparing finetune data...")
        try:
            prepare_split_finetune_data(
                input_path=args.input_path,
                system_prompt_path=args.system_prompt_path,
                user_prompt_path=args.user_prompt_path,
                output_train_jsonl=args.train_file_path,
                output_val_jsonl=args.val_file_path,
                split_ratio=args.split_ratio,
                validation_schema=Mnemonic,
            )
        except Exception as e:
            logger.exception("Error preparing finetune data")
            raise e
    else:
        logger.info("Skipping data preparation step.")

    load_dotenv()
    client = OpenAI()
    if not args.skip_upload:
        logger.info("Uploading files to OpenAI...")
        try:
            upload_finetune_data(
                client,
                input_path=args.train_file_path,
                config_file_path=args.config_path,
                file_type="train",
            )
            upload_finetune_data(
                client,
                input_path=args.val_file_path,
                config_file_path=args.config_path,
                file_type="val",
            )
        except Exception as e:
            logger.exception("Error uploading files to OpenAI")
            raise e

    if not args.skip_finetune:
        logger.info("Starting fine-tuning process...")
        finetune_from_config(
            client,
            config_file_path=args.config_path,
        )
    else:
        logger.info("Skipping fine-tuning step.")
