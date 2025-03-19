"""Module for processing LLM requests and responses with litellm, endpoint: chat completion."""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING

from litellm import (
    batch_completion,
    completion,
    supports_response_schema,
    validate_environment,
)
from pydantic import BaseModel

from src import const
from src.utils import check_file_path, read_config

if TYPE_CHECKING:
    from typing import Any, Optional

    from src.utils.aliases import PathLike

# Set up logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    logger.addHandler(logging.StreamHandler())


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
        output_schema (subclass of BaseModel, optional): Pydantic model for validation
        mock_response (str, optional): Mock response to use instead of LLM response
        **kwargs: Additional keyword arguments for litellm.completion

    Returns:
        Dictionary of input parameters
    """
    # Load configuration if provided
    config = {}
    if config_path:
        config = read_config(config_path)

    if default_config_path:
        default_config = read_config(default_config_path)
    if default_config is None:
        default_config = {}

    # Prioritize user config over default, then kwargs
    config = {**default_config, **config, **kwargs}

    # JSON mode
    if output_schema:
        assert supports_response_schema(model=config["model"]), (
            f"Model {config['model']} does not support JSON schema output."
        )
        config["response_format"] = output_schema

    # Use mock response if requested
    if mock_response:
        config["mock_response"] = "A mock response"

    # Add messages + config to params
    params = {"messages": messages, **config}
    logger.info(f"Sending request with params: {params}")
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
    except Exception as e:
        logger.error(f"Error calling LLM API: {e}")
        raise e


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
        logger.error(f"Error calling LLM API: {e}")
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
                    content = output_schema.model_validate_json(content)
                processed_responses.append(content)
            return processed_responses
        else:
            logger.debug("Processing single response.")
            content = response.choices[0].message.content
            if output_schema:
                processed_response = output_schema.model_validate_json(content)
            return processed_response
    except Exception as e:
        logger.error(f"Error processing LLM response: {e}")
        logger.error(f"Raw response: {response}")
        raise e


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
