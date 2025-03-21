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

from src import const
from src.utils import check_file_path, read_config

if TYPE_CHECKING:
    from typing import Any, Optional

    from structlog.stdlib import BoundLogger

    from src.utils.aliases import PathLike

# Set up logging
logger: BoundLogger = structlog.getLogger(__name__)


def build_input_params(
    messages: list[dict[str, Any]] | list[list[dict[str, Any]]],
    config_path: Optional[PathLike] = None,
    default_config_path: Optional[PathLike] = const.CONF_DEFAULT,
    output_schema: Optional[type[BaseModel]] = None,
    mock_response: Optional[str] = None,
    **kwargs,
) -> dict[str, Any]:
    """Build input parameters for the LLM request.

    Args:
        messages (list of dict of str, Any): List of messages to send to the LLM
        config_path (PathLike, optional): Path to configuration file
        default_config_path (PathLike, optional): Path to default configuration file
        output_schema (subclass of BaseModel, optional): Pydantic model for validation. If set, the model will be used to validate the response.
            INCOMPATIBLE with mock_response
        mock_response (str, optional): Mock response to use instead of LLM response. If set, the model will not be called.
            INCOMPATIBLE with output_schema
        **kwargs: Additional keyword arguments for litellm.completion

    Returns:
        Dictionary of input parameters

    Raises:
        ValueError: If mock response is used with JSON schema output
    """
    # Load configuration if provided
    config = {}
    if config_path:
        config = read_config(config_path)

    if default_config_path:
        default_config = read_config(default_config_path)
    if default_config is None:
        default_config = {}

    # Check for mock response and output schema compatibility
    if mock_response and output_schema:
        raise ValueError("Cannot use mock_response and output_schema at the same time.")
    elif output_schema:
        if not issubclass(output_schema, BaseModel):
            raise TypeError("output_schema must be a subclass of pydantic.BaseModel")

        elif not supports_response_schema(model=config["model"]):
            raise ValueError(
                f"Model {config['model']} does not support JSON schema output."
            )

        config["response_format"] = output_schema
        logger.debug(f"Using JSON schema output: {output_schema}")

    elif mock_response:
        if not isinstance(mock_response, str):
            raise TypeError("mock_response must be a string.")
        logger.debug(f"Using mock response: {mock_response}")
        config["mock_response"] = mock_response

    # Prioritize user config over default, then kwargs
    config = {**default_config, **config, **kwargs}

    # Add messages + config to params
    params = {"messages": messages, **config}
    logger.debug(f"Sending request with config (excluding messages): {config}")
    return params


# TODO: Add logging observability with langfuse
def complete(
    messages: list[dict[str, Any]],
    config_path: Optional[PathLike] = None,
    output_schema: Optional[type[BaseModel]] = None,
    mock_response: Optional[str] = None,
) -> list[dict[str, Any]]:
    """Send a single completion request to the LLM API.

    Args:
        messages (list of dict of str, Any): List of messages to send to the LLM
        config_path (PathLike, optional): Path to configuration file
        output_schema (subclass of BaseModel, optional): Pydantic model for validation
        mock_response (str, optional): Mock response to use instead of LLM response

    Returns:
        List of processed response data
    """
    try:
        # Send request using litellm
        params = build_input_params(
            messages,
            config_path=config_path,
            output_schema=output_schema,
            mock_response=mock_response,
        )
        validate_environment(model=params["model"])
        response = completion(**params)

        return process_llm_response(response, output_schema)
    except Exception:
        logger.exception("Error calling LLM API", stack_info=True)


def batch_complete(
    messages: list[list[dict[str, Any]]],
    config_path: Optional[PathLike] = None,
    output_schema: Optional[type[BaseModel]] = None,
    mock_response: Optional[str] = None,
    **kwargs,
) -> list[dict[str, Any]]:
    """Send a batch completion request to the LLM API.

    Args:
        messages (list of list of dict of str, Any): List of messages to send to the LLM
        config_path (PathLike, optional): Path to configuration file
        output_schema (subclass of BaseModel, optional): Pydantic model for validation
        mock_response (str, optional): Mock response to use instead of LLM response
        **kwargs: Additional keyword arguments for build_input_params
    Returns:
        List of processed response data
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
        validate_environment(model=params["model"])
        response = batch_completion(**params)

        return process_llm_response(response, output_schema)
    except Exception as e:
        logger.exception("Error calling LLM API", stack_info=True)
        raise e


def process_llm_response(
    response: Any, output_schema: Optional[type[BaseModel]] = None
) -> list[dict[str, Any]] | dict[str, Any]:
    """Process the LLM response (OpenAI format) and validate against the schema.

    Args:
        response: The raw response from the LLM. To access the first response content, use `response.choices[0].message.content`
        output_schema (subclass of  BaseModel, optional): Pydantic model for validation
    Returns:
        List of processed response content, or a single processed response content
    """
    if output_schema is not None and not issubclass(output_schema, BaseModel):
        raise TypeError("output_schema must be a subclass of pydantic.BaseModel")

    # Process response
    try:
        processed_responses = []
        if isinstance(response, list):
            logger.debug(f"Processing batch response with {len(response)} items.")
            for res in response:
                content = res.choices[0].message.content
                if output_schema:
                    content = validate_content_against_schema(content, output_schema)
                processed_responses.append(content)
            return processed_responses
        else:
            logger.debug("Processing single response.")
            content = response.choices[0].message.content

            if output_schema:
                content = validate_content_against_schema(content, output_schema)
            return content
    except Exception as e:
        logger.exception("Error processing LLM response:", raw_respose=response)
        raise e


def validate_content_against_schema(
    content: Any, schema: type[BaseModel]
) -> dict[str, Any]:
    """Validate the content against the schema.

    Args:
        content (Any): The content to validate. Ideally a JSON string.
        schema (subclass of pydantic.BaseModel): The schema to validate against

    Returns:
        dict[str, Any]: The validated content
    """
    try:
        content = schema.model_validate_json(content)
    except ValidationError:
        logger.warning("Validation error. Attempting to fix incomplete JSON: {content}")
        content = _attempt_fix_incomplete_json(content)
        try:
            content = schema.model_validate_json(content)
        except ValidationError as validation_error:
            logger.exception(
                "Error validating content with schema", content=content, schema=schema
            )
            raise validation_error
    except Exception as e:
        logger.error(
            "Non-validation error while validating content",
            content=content,
            schema=schema,
        )
        raise e from e


def save_results(
    results: list[dict[str, Any]],
    output_path: PathLike,
    flatten: bool = True,
) -> None:
    """Save results to a JSON file.

    Args:
        results: List of result data
        output_path: Path to save the results
        flatten: Whether to flatten nested lists
    """
    output_path = check_file_path(
        output_path, new_ok=True, to_create=True, extensions=[".json"]
    )

    # Flatten if needed and requested
    final_results = []
    if flatten:
        for result in results:
            if isinstance(result, list):
                final_results.extend(result)
            else:
                final_results.append(result)
    else:
        final_results = results

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(final_results, f, ensure_ascii=False, indent=2)

    logger.info(f"Saved {len(final_results)} results to {output_path}")


def _attempt_fix_incomplete_json(content: str) -> str:
    """Attempt to fix incomplete JSON by closing open brackets and quotes.

    Args:
        content: The incomplete JSON string

    Returns:
        str: The fixed JSON string
    """
    # Count open brackets and quotes
    open_braces = content.count("{") - content.count("}")
    open_quotes = content.count('"') % 2

    # Simple case: just need to close braces
    if open_braces > 0 and open_quotes == 0:
        return content + "}" * open_braces

    # More complex case: need to close quotes and possibly fields
    if open_quotes > 0:
        # Try to find the last property name
        last_property = content.rfind('"')
        if last_property > 0:
            # Check if we're in a property name or value
            prev_colon = content.rfind(":", 0, last_property)
            prev_comma = content.rfind(",", 0, last_property)

            if prev_colon > prev_comma:  # We're in a value
                return content + '"' + "}" * open_braces
            else:  # We're in a property name
                return content + '":""' + "}" * open_braces

    # If all else fails, just return the original content
    return content
