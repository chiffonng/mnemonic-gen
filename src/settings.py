"""Global settings for the application, including logging configuration.

structlog reference: https://www.structlog.org/en/stable/standard-library.html#rendering-within-structlog
"""

import logging
import logging.config
from pathlib import Path

import structlog

# Set up logging directory and file
log_file = "logs/app.log"  # Default log file path
log_filepath = Path(log_file)  # Convert to Path object
# Ensure the log directory exists
log_dir = log_filepath.parent
log_dir.mkdir(parents=True, exist_ok=True)
log_filepath.touch()


# Create a custom filter to ignore litellm errors about __annotations__
class IgnoreFilter(logging.Filter):
    """Filter to ignore specific litellm errors about __annotations__."""

    def filter(self, record: logging.LogRecord) -> bool:
        """Filter out logs matching specific patterns.

        Args:
            record: The log record to check

        Returns:
            bool: True if the record should be processed, False if it should be filtered out
        """
        # Check if it's a LiteLLM error about __annotations__
        if hasattr(record, "msg") and isinstance(record.msg, str):
            if "LiteLLM:ERROR" in record.msg and "__annotations__" in record.msg:
                return False

        # Allow the log record
        return True


LOGGING_CONF = {
    "version": 1,
    "formatters": {
        "json_formatter": {
            "()": structlog.stdlib.ProcessorFormatter,
            "processor": structlog.processors.JSONRenderer(),
        },
        "plain_console": {
            "()": structlog.stdlib.ProcessorFormatter,
            "processor": structlog.dev.ConsoleRenderer(),
        },
    },
    "filters": {
        "ignore_litellm_annotations": {
            "()": IgnoreFilter,
        }
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "plain_console",
            "filters": ["ignore_litellm_annotations"],
        },
        "json_file": {
            "class": "logging.handlers.WatchedFileHandler",
            "filename": log_file,
            "formatter": "json_formatter",
            "filters": ["ignore_litellm_annotations"],
        },
    },
    "loggers": {
        "src": {
            "handlers": ["console", "json_file"],
            "level": "DEBUG",
        },
        "scripts": {
            "handlers": ["console", "json_file"],
            "level": "ERROR",
        },
    },
}

logging.config.dictConfig(LOGGING_CONF)

structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.processors.TimeStamper(fmt="iso", utc=False),
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.StackInfoRenderer(),
        structlog.processors.UnicodeDecoder(),  # decode bytes to str
        structlog.stdlib.PositionalArgumentsFormatter(),  # perform %-style formatting.
        structlog.processors.CallsiteParameterAdder(
            [
                structlog.processors.CallsiteParameter.FUNC_NAME,
                structlog.processors.CallsiteParameter.LINENO,
            ]
        ),
        structlog.dev.ConsoleRenderer(
            exception_formatter=structlog.dev.RichTracebackFormatter()
        ),
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    #  wrapper_class=structlog.make_filtering_bound_logger(logging.ERROR) # for filtering errors or higher
    cache_logger_on_first_use=True,
)
