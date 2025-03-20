"""Script to generate full dataset of improved mnemonics and classified linguistic reasoning. From BasicMnemonic to Mnemonic (see src/data_prep/mnemonic_schemas.py)."""

import argparse

import structlog
from src import const
from src._mnemonic_enhancer.mnemonic_enhancer import improve_mnemonic

# Set up logging
logger = structlog.getLogger(__name__)


def improve_single_mnemonic():
    """Test the mnemonic enhancement functionality with a single term."""
    parser = argparse.ArgumentParser(
        description="Test mnemonic enhancement with OpenAI API"
    )
    parser.add_argument(
        "--term",
        type=str,
        default="anachronism",
        help="Term to generate an improved mnemonic for",
    )
    parser.add_argument(
        "--mnemonic",
        type=str,
        default="You know about 'chronometer,' which means clock. Anything with 'chron' relates to time. Anachronism (an+chron+ism) uses 'an' in a negative sense, so anachronism means something or someone not in its correct time period.",
        help="Current basic mnemonic for the term",
    )
    parser.add_argument(
        "--model",
        choices=["claude", "openai", "openai_sft"],
        default="openai",
        help="Model to use for enhancement",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        default=False,
        help="Use mock responses instead of calling API",
    )

    args = parser.parse_args()

    # Set up paths based on the selected model
    if args.model == "openai":
        config_path = const.CONF_OPENAI
    elif args.model == "openai_sft":
        config_path = const.CONF_OPENAI_SFT
    elif args.model == "claude":
        config_path = const.CONF_CLAUDE
    else:
        raise ValueError(f"Unsupported model: {args.model}")

    try:
        logger.info(f"Testing mnemonic enhancement for term: '{args.term}'")
        result = improve_mnemonic(
            term=args.term,
            mnemonic=args.mnemonic,
            config_path=config_path,
            use_mock=args.mock,
        )

        logger.info("Enhancement successful!")
        logger.info(f"Term: {result.term}")
        logger.info(f"Improved Mnemonic: {result.mnemonic}")
        logger.info(f"Linguistic Reasoning: {result.linguistic_reasoning}")
        logger.info(f"Main Type: {result.main_type}")
        logger.info(f"Sub Type: {result.sub_type if result.sub_type else 'None'}")

    except Exception as e:
        logger.error(f"Error during mnemonic enhancement: {e}")
        raise e


if __name__ == "__main__":
    improve_single_mnemonic()
