"""Module for processing LLM requests and responses with litellm, endpoint: chat completion."""

from __future__ import annotations

import json
from multiprocessing import process
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
    from typing import Any, Optional, Sequence

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
        messages (list of dict of [str, Any], or list of list of dict of [str, Any]): List of messages to send to the LLM
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
        logger.debug(
            "Using JSON schema output:", model=config["model"], schema=output_schema
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
        if validate_environment(model=params["model"]):
            logger.debug("Environment is valid for model", model=params["model"])
            response = completion(**params)
        else:
            logger.warning("Environment is NOT valid for model", model=params["model"])
            raise ValueError("Environment is NOT valid for model")

        return process_llm_response(response, output_schema)
    except Exception as e:
        logger.exception("Error calling LLM API")
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
        logger.exception("Error calling LLM API")
        raise e


def process_llm_response(
    response: Any, output_schema: Optional[type[BaseModel]] = None
) -> str | BaseModel | dict[str, Any]:
    """Process the LLM response (OpenAI format) and validate against the schema.

    Args:
        response (Any): The raw response from the LLM. To access the first response content, use `response.choices[0].message.content`
        output_schema (subclass of  BaseModel, optional): Pydantic model for validation
    Returns:
        content (str | type[BaseModel] | dict[str, Any]): A single processed response content, or a structured object if output_schema is provided

    Raises:
        ValueError: If the response is None
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
        logger.debug("Processing raw response", raw_response=response)
        content = response.choices[0].message.content
        logger.debug("Content of the response", content=content)
        logger.debug("Usage of the response", usage=response.usage)

        if content is None:
            logger.exception("response.choices[0].message.content is None")
            raise ValueError("response.choices[0].message.content is None")
        if output_schema:
            content = validate_content_against_schema(content, output_schema)
        return content

    except Exception as e:
        logger.exception("Error processing LLM response:", raw_respose=response)
        raise e


def process_llm_multiple_responses(
    responses: Sequence[Any], output_schema: Optional[type[BaseModel]] = None
) -> list[str] | list[type[BaseModel]] | list[dict[str, Any]]:
    """Process multiple LLM responses and validate against the schema.

    Args:
        responses (Sequence[Any]): List of raw responses from the LLM
        output_schema (subclass of BaseModel, optional): Pydantic model for validation

    Returns:
        processed_responses: List of processed response data
    """
    processed_responses = []
    for response in tqdm(responses):
        try:
            processed_response = process_llm_response(response, output_schema)
            processed_responses.append(processed_response)
        except Exception:
            logger.exception("Error processing LLM response:", raw_respose=response)
            processed_responses.append(None)
    return processed_responses


def validate_content_against_schema(
    content: Any, schema: type[BaseModel]
) -> BaseModel | dict[str, Any]:
    """Validate the content against the schema.

    Args:
        content (Any): The content to validate. Ideally a JSON string.
        schema (subclass of pydantic.BaseModel): The schema to validate against

    Returns:
        parsed (BaseModel or dict[str, Any]): The content parsed and validated against the schema

    Raises:
        ValueError: If the content is not a string or dictionary
        json.JSONDecodeError: If the content is not valid JSON
        pydantic.ValidationError: If the content does not match the schema
    """
    try:
        if isinstance(content, (dict, schema)):
            logger.debug(
                "Validating content against schema using model_validate method"
            )
            return schema.model_validate(content)
        elif isinstance(content, (str, bytearray, bytes)):
            try:
                # First try direct JSON validation
                logger.debug("Validating JSON string against schema")
                return schema.model_validate_json(content)
            except ValidationError as e:
                # If direct validation fails, try parsing JSON first then validate
                logger.warning(
                    "Direct validation failed, parsing JSON first", error=str(e)
                )
                parsed_json = json.loads(content)
                return schema.model_validate(parsed_json)
        else:
            logger.warning(f"Unexpected content type: {type(content)}")
            content_str = str(content)
            logger.debug("Validating content string against schema")
            return schema.model_validate_json(content_str)

    except json.JSONDecodeError:
        logger.exception("JSON decode error", content=content)
        # Try to fix incomplete JSON
        fixed_content = _attempt_fix_incomplete_json(str(content))
        logger.debug("Attempting with fixed JSON", content=fixed_content)
        try:
            return schema.model_validate_json(fixed_content)
        except Exception as fix_error:
            logger.exception(
                "Error validating fixed JSON",
                content=fixed_content,
                schema=schema,
            )
            raise fix_error

    except ValidationError as validation_error:
        logger.exception(
            "Error validating content with schema", content=content, schema=schema
        )
        raise validation_error
    except Exception as e:
        logger.error(
            "Unexpected error validating content with schema",
            content=content,
            schema=schema,
        )
        raise e


def _attempt_fix_incomplete_json(content: str) -> str:
    """Attempt to fix incomplete JSON by closing open brackets and quotes.

    Args:
        content: The incomplete JSON string

    Returns:
        str: The fixed JSON string
    """
    # Count open brackets and quotes
    open_braces = content.count("{") - content.count("}")
    open_brackets = content.count("[") - content.count("]")
    open_quotes = content.count('"') % 2

    # Fix open braces and brackets
    fixed_content = content

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


# from src._mnemonic_enhancer.mnemonic_schemas import ImprovedMnemonic

# json_data = "{\"improved_mnemonic\":\"Think of 'anachronism' as 'an-a-chronic-mistake.' Imagine a caveman using an iPhone; that's a funny and clear example of an anachronism—something mistakenly placed in the wrong time period. The prefix 'an-' means not, 'chron' relates to time, and '-ism' denotes a practice or condition, making it evident that this term refers to something out of its proper chronological context.\",\"linguistic_reasoning\":\"The mnemonic breaks down the word into components: 'an-' (not), 'chron' (time), and '-ism' (condition). It uses a vivid scenario to illustrate the definition.\"}"
# validated = validate_content_against_schema(json_data, ImprovedMnemonic)

# print(validated)  # works

# TODO: Test other functions with pytest and patching
# TODO: Then test with real LLM
