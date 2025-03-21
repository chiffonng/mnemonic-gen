"""Module for processing LLM requests and responses with litellm, endpoint: chat completion."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import structlog
from litellm import (
    batch_completion,
    completion,
    supports_response_schema,
    validate_environment,
)
from pydantic import BaseModel, ValidationError
from tqdm import tqdm

from src.utils import constants as const
from src.utils import read_config

if TYPE_CHECKING:
    from typing import Any, Optional, Sequence, TypeVar

    from structlog.stdlib import BoundLogger

    from src.utils.aliases import PathLike

    ModelT = TypeVar("ModelT", bound=BaseModel)  # subclass of BaseModel
    ResponseType = str | ModelT | dict[str, Any]
    BatchResponseType = list[ResponseType]

# Set up logging
logger: BoundLogger = structlog.getLogger(__name__)


def build_input_params(
    messages: list[dict[str, Any]] | list[list[dict[str, Any]]],
    config_path: Optional[PathLike] = None,
    default_config_path: Optional[PathLike] = const.CONF_DEFAULT,
    output_schema: Optional[type[ModelT]] = None,
    mock_response: Optional[str] = None,
    **kwargs,
) -> dict[str, Any]:
    """Build input parameters for the LLM request.

    Args:
        messages: List of messages or batch of messages to send to the LLM
        config_path: Path to configuration file
        default_config_path: Path to default configuration file
        output_schema: Pydantic model for validation. If set, the model will be used to validate the response.
            INCOMPATIBLE with mock_response
        mock_response: Mock response to use instead of LLM response. If set, the model will not be called.
            INCOMPATIBLE with output_schema
        **kwargs: Additional keyword arguments for litellm.completion

    Returns:
        Dictionary of input parameters

    Raises:
        ValueError: If mock response is used with JSON schema output
    """
    # Load configuration if provided
    config = {}
    default_config = {}

    if config_path:
        config = read_config(config_path)

    if default_config_path:
        default_config = read_config(default_config_path)

    # Check for mock response and output schema compatibility
    if mock_response and output_schema:
        raise ValueError("Cannot use mock_response and output_schema at the same time.")
    elif output_schema:
        if not issubclass(output_schema, BaseModel):
            raise TypeError("output_schema must be a subclass of pydantic.BaseModel")

        if "model" in config and not supports_response_schema(model=config["model"]):
            raise ValueError(
                f"Model {config['model']} does not support JSON schema output."
            )

        config["response_format"] = output_schema
        logger.debug(
            "Using JSON schema output:", model=config.get("model"), schema=output_schema
        )

    elif mock_response:
        if not isinstance(mock_response, str):
            raise TypeError("mock_response must be a string.")
        logger.debug("Using mock response", mock_response=mock_response)
        config["mock_response"] = mock_response

    # Prioritize user config over default, then kwargs
    config = {**default_config, **config, **kwargs}

    # Add messages + config to params
    params = {"messages": messages, **config}
    logger.debug("Sending request with config (excluding messages):", config=config)
    return params


def complete(
    messages: list[dict[str, Any]],
    config_path: Optional[PathLike] = None,
    output_schema: Optional[type[ModelT]] = None,
    mock_response: Optional[str] = None,
    **kwargs,
) -> ResponseType:
    """Send a single completion request to the LLM API.

    Args:
        messages: List of messages to send to the LLM
        config_path: Path to configuration file
        output_schema: Pydantic model for validation
        mock_response: Mock response to use instead of LLM response
        **kwargs: Additional keyword arguments for build_input_params

    Returns:
        Processed response data (string, validated model, or raw dict)

    Raises:
        ValueError: If environment is not valid for the model
        Exception: Any error during API call or response processing
    """
    try:
        # Send request using litellm
        params = build_input_params(
            messages,
            config_path=config_path,
            output_schema=output_schema,
            mock_response=mock_response,
            **kwargs,
        )

        model = params.get("model")
        if not model:
            logger.exception("No model specified in params")
            raise ValueError("No model specified in params")

        if validate_environment(model=model):
            logger.debug("Environment is valid for model", model=model)
            response = completion(**params)
        else:
            logger.warning("Environment is NOT valid for model", model=model)
            raise ValueError(f"Environment is NOT valid for model {model}")

        return process_llm_response(response, output_schema)
    except Exception as e:
        logger.exception("Error calling LLM API")
        raise e


def batch_complete(
    messages: list[list[dict[str, Any]]],
    config_path: Optional[PathLike] = None,
    output_schema: Optional[type[ModelT]] = None,
    mock_response: Optional[str] = None,
    **kwargs,
) -> BatchResponseType:
    """Send a batch completion request to the LLM API.

    Args:
        messages: List of lists of messages to send to the LLM
        config_path: Path to configuration file
        output_schema: Pydantic model for validation
        mock_response: Mock response to use instead of LLM response
        **kwargs: Additional keyword arguments for build_input_params

    Returns:
        List of processed response data

    Raises:
        ValueError: If environment is not valid for the model
        Exception: Any error during batch API call or response processing
    """
    try:
        # Send request using litellm
        params = build_input_params(
            messages,
            config_path=config_path,
            output_schema=output_schema,
            mock_response=mock_response,
            **kwargs,
        )

        model = params.get("model")
        if not model:
            logger.warning("No model specified in params")
            model = "default model"

        if validate_environment(model=model):
            logger.debug("Environment is valid for model", model=model)
            responses = batch_completion(**params)
        else:
            logger.warning("Environment is NOT valid for model", model=model)
            raise ValueError(f"Environment is NOT valid for model {model}")

        return process_llm_responses(responses, output_schema)
    except Exception as e:
        logger.exception("Error calling batch LLM API")
        raise e


def process_llm_response(
    response: Any, output_schema: Optional[type[ModelT]] = None
) -> ResponseType:
    """Process the LLM response and validate against the schema if provided.

    Args:
        response: The raw response from the LLM
        output_schema: Pydantic model for validation

    Returns:
        Processed response (string, validated model, or raw dict)

    Raises:
        ValueError: If the response or content is None
        TypeError: If output_schema is not a subclass of pydantic.BaseModel
        Exception: Any error during response processing
    """
    if response is None:
        logger.exception("Response from LLM is None")
        raise ValueError("Response is None")

    if output_schema is not None and not issubclass(output_schema, BaseModel):
        logger.exception("output_schema is not a subclass of pydantic.BaseModel")
        raise TypeError("output_schema must be a subclass of pydantic.BaseModel")

    # Process response
    try:
        logger.debug(
            "Processing raw response",
            raw_response=str(response)[:100] + "..."
            if len(str(response)) > 100
            else str(response),
        )

        # Extract content from response
        if hasattr(response, "choices") and response.choices:
            if hasattr(response.choices[0], "message") and hasattr(
                response.choices[0].message, "content"
            ):
                content = response.choices[0].message.content
            else:
                logger.exception(
                    "Unexpected response structure - missing message.content",
                    raw_response=response,
                )
                raise ValueError("Unexpected response structure")
        else:
            logger.exception(
                "Unexpected response structure - missing choices", raw_response=response
            )
            raise ValueError("Unexpected response structure")

        logger.debug(
            "Content of the response",
            content=content[:100] + "..."
            if content and len(content) > 100
            else content,
        )

        if hasattr(response, "usage"):
            logger.debug("Usage of the response", usage=response.usage)

        if content is None:
            logger.exception("Response content is None")
            raise ValueError("Response content is None")

        if output_schema:
            return validate_content_against_schema(content, output_schema)

        return content

    except Exception as e:
        logger.exception("Error processing LLM response")
        raise e


def process_llm_responses(
    responses: Sequence[Any], output_schema: Optional[type[ModelT]] = None
) -> BatchResponseType:
    """Process multiple LLM responses and validate against the schema if provided.

    Args:
        responses: List of raw responses from the LLM
        output_schema: Pydantic model for validation

    Returns:
        List of processed responses (strings, validated models, or raw dicts)
    """
    if not responses:
        logger.warning("Empty responses list received")
        return []

    processed_responses = []
    for response in tqdm(responses, desc="Processing LLM responses"):
        try:
            processed_response = process_llm_response(response, output_schema)
            processed_responses.append(processed_response)
        except Exception as e:
            logger.exception("Error processing response", error=str(e))
            processed_responses.append(None)

    return processed_responses


def validate_content_against_schema(content: Any, schema: type[ModelT]) -> ModelT:
    """Validate the content against the schema.

    Args:
        content: The content to validate (string, dict, or model instance)
        schema: The schema to validate against

    Returns:
        The content parsed and validated against the schema

    Raises:
        ValueError: If the content is not a string or dictionary
        json.JSONDecodeError: If the content is not valid JSON
        ValidationError: If the content does not match the schema
    """
    if content is None:
        logger.exception("Content is None")
        raise ValueError("Content is None")
    try:
        # if content is already schema, return it
        if isinstance(content, schema):
            return content
        elif isinstance(content, dict):
            logger.debug("Validating dictionary against schema")
            return schema.model_validate(content)

        elif isinstance(content, (str, bytes, bytearray)):
            try:
                logger.debug("Validating JSON string against schema")
                return schema.model_validate_json(content)
            except ValidationError as e:
                # Try parsing JSON first then validate
                logger.warning(
                    "Direct validation failed, parsing JSON first", error=str(e)
                )
                parsed_json = json.loads(content)
                return schema.model_validate(parsed_json)

        # Fallback for other types
        else:
            logger.warning(f"Unexpected content type: {type(content)}")
            content_str = str(content)
            return schema.model_validate_json(content_str)

    except json.JSONDecodeError as json_error:
        content_str = str(content)
        snippet = content_str[:100] + "..." if len(content_str) > 100 else content_str
        logger.exception("JSON decode error", error=str(json_error), content=snippet)

        # Try to fix incomplete JSON
        fixed_content = _attempt_fix_incomplete_json(str(content))
        fixed_snippet = (
            fixed_content[:100] + "..." if len(fixed_content) > 100 else fixed_content
        )
        logger.debug("Attempting with fixed JSON", fixed_content=fixed_snippet)

        try:
            return schema.model_validate_json(fixed_content)
        except Exception as fix_error:
            logger.exception("Error validating fixed JSON", error=str(fix_error))
            raise fix_error

    except ValidationError as validation_error:
        logger.exception("Error validating content with schema")
        raise validation_error

    except Exception as e:
        logger.exception("Unexpected error validating content")
        raise e


def _attempt_fix_incomplete_json(content: str) -> str:
    """Attempt to fix incomplete JSON by closing open brackets and quotes.

    Args:
        content: The incomplete JSON string

    Returns:
        The fixed JSON string
    """
    # Count open brackets and quotes
    open_braces = content.count("{") - content.count("}")
    open_brackets = content.count("[") - content.count("]")
    open_quotes = content.count('"') % 2

    # Fix open braces and brackets
    fixed_content = content

    # Handle trailing commas in objects or arrays
    if fixed_content.rstrip().endswith(","):
        fixed_content = fixed_content.rstrip()[:-1]

    # First fix quotes if needed
    if open_quotes > 0:
        # Try to find the last property name or value
        last_quote = content.rfind('"')
        if last_quote > 0:
            # Check if we're in a property name or value
            prev_colon = content.rfind(":", 0, last_quote)
            prev_comma = content.rfind(",", 0, last_quote)

            if prev_colon > prev_comma:  # We're in a value
                fixed_content = content + '"'
            else:  # We're in a property name
                fixed_content = content + '":""'

    # Then close brackets and braces
    if open_brackets > 0:
        fixed_content += "]" * open_brackets

    if open_braces > 0:
        fixed_content += "}" * open_braces

    return fixed_content
