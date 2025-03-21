"""Global settings for the application."""

import logging
import logging.config

import structlog

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
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "plain_console",
        },
        "json_file": {
            "class": "logging.handlers.WatchedFileHandler",
            "filename": "logs/json.log",
            "formatter": "json_formatter",
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
        structlog.contextvars.merge_contextvars,
        # Add module name and line number from the call site
        structlog.stdlib.filter_by_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.CallsiteParameterAdder(
            [
                structlog.processors.CallsiteParameter.MODULE,
                structlog.processors.CallsiteParameter.FUNC_NAME,
                structlog.processors.CallsiteParameter.LINENO,
            ]
        ),
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        structlog.dev.ConsoleRenderer(),
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    #  wrapper_class=structlog.make_filtering_bound_logger(logging.ERROR) # for filtering errors or higher
    cache_logger_on_first_use=True,
)
