"""Tests for the client module."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from pydantic import BaseModel

from src.llms.client import (
    build_input_params,
    complete,
    process_llm_response,
    batch_complete,
)


class SimpleOutputSchema(BaseModel):
    """Simple schema for testing."""

    message: str


@pytest.fixture
def mock_config_path(tmp_path):
    """Create a mock config file for testing."""
    config = {
        "model": "test-model",
        "temperature": 0.7,
        "max_tokens": 100,
    }
    config_path = tmp_path / "test_config.json"
    config_path.write_text(json.dumps(config))
    return config_path


@pytest.fixture
def mock_default_config_path(tmp_path):
    """Create a mock default config file for testing."""
    config = {
        "model": "default-model",
        "temperature": 0.5,
        "max_tokens": 50,
    }
    config_path = tmp_path / "default_config.json"
    config_path.write_text(json.dumps(config))
    return config_path


@pytest.fixture
def mock_messages():
    """Create mock messages for testing."""
    return [{"role": "user", "content": "Hello, world!"}]


def test_build_input_params_with_config(mock_config_path, mock_messages):
    """Test build_input_params with a config file."""
    params = build_input_params(
        messages=mock_messages,
        config_path=mock_config_path,
    )

    assert params["messages"] == mock_messages
    assert params["model"] == "test-model"
    assert params["temperature"] == 0.7
    assert params["max_tokens"] == 100


def test_build_input_params_with_default_config(
    mock_default_config_path, mock_messages
):
    """Test build_input_params with a default config file."""
    params = build_input_params(
        messages=mock_messages,
        default_config_path=mock_default_config_path,
    )

    assert params["messages"] == mock_messages
    assert params["model"] == "default-model"
    assert params["temperature"] == 0.5
    assert params["max_tokens"] == 50


def test_build_input_params_override_priority(
    mock_config_path, mock_default_config_path, mock_messages
):
    """Test that config overrides default config, and kwargs override both."""
    params = build_input_params(
        messages=mock_messages,
        config_path=mock_config_path,
        default_config_path=mock_default_config_path,
        temperature=0.9,  # should override both configs
    )

    assert params["model"] == "test-model"  # from config
    assert params["temperature"] == 0.9  # from kwargs
    assert params["max_tokens"] == 100  # from config


def test_build_input_params_with_output_schema(mock_config_path, mock_messages):
    """Test build_input_params with an output schema."""
    with patch("src.llms.client.supports_response_schema", return_value=True):
        params = build_input_params(
            messages=mock_messages,
            config_path=mock_config_path,
            output_schema=SimpleOutputSchema,
        )

        # Check if response_format exists in params
        assert "response_format" in params


def test_build_input_params_output_schema_error(mock_config_path, mock_messages):
    """Test that an error is raised when the model doesn't support response schema."""
    with patch("src.llms.client.supports_response_schema", return_value=False):
        with pytest.raises(ValueError, match="does not support JSON schema output"):
            build_input_params(
                messages=mock_messages,
                config_path=mock_config_path,
                output_schema=SimpleOutputSchema,
            )


def test_build_input_params_with_mock_response(mock_config_path, mock_messages):
    """Test build_input_params with a mock response."""
    params = build_input_params(
        messages=mock_messages,
        config_path=mock_config_path,
        mock_response="Test response",
    )

    assert params["mock_response"] == "Test response"


@patch("src.llms.client.process_llm_response")
@patch("src.llms.client.completion")
@patch("src.llms.client.validate_environment")
@patch("src.llms.client.build_input_params")
def test_complete_success(
    mock_build_params, mock_validate, mock_completion, mock_process, mock_messages
):
    """Test successful completion with standard parameters."""
    # Setup
    mock_build_params.return_value = {"model": "test-model", "messages": mock_messages}
    mock_response = MagicMock()
    mock_process.return_value = "Hello, there!"
    mock_completion.return_value = mock_response

    # Execute
    result = complete(messages=mock_messages)

    # Verify
    mock_build_params.assert_called_once()
    mock_validate.assert_called_once_with(model="test-model")
    mock_completion.assert_called_once_with(**mock_build_params.return_value)
    mock_process.assert_called_once_with(mock_response, None)
    assert result == "Hello, there!"


@patch("src.llms.client.process_llm_response")
@patch("src.llms.client.completion")
@patch("src.llms.client.validate_environment")
@patch("src.llms.client.build_input_params")
def test_complete_with_output_schema(
    mock_build_params, mock_validate, mock_completion, mock_process, mock_messages
):
    """Test completion with output schema validation."""
    # Setup
    mock_build_params.return_value = {"model": "test-model", "messages": mock_messages}
    mock_response = MagicMock()
    schema_instance = SimpleOutputSchema(message="Hello, there!")
    mock_process.return_value = schema_instance
    mock_completion.return_value = mock_response

    # Execute
    result = complete(messages=mock_messages, output_schema=SimpleOutputSchema)

    # Verify
    mock_process.assert_called_once_with(mock_response, SimpleOutputSchema)
    assert isinstance(result, SimpleOutputSchema)
    assert result.message == "Hello, there!"


@patch("src.llms.client.completion")
def test_complete_error_handling(mock_completion, mock_messages):
    """Test error handling in complete function."""
    # Setup
    mock_completion.side_effect = Exception("Test error")

    # Execute and verify
    with pytest.raises(Exception, match="Test error"):
        complete(messages=mock_messages)


def test_process_llm_response_single():
    """Test processing a single LLM response."""
    # Setup
    mock_response = MagicMock()
    mock_response.choices[0].message.content = "Test response"

    # Execute
    result = process_llm_response(mock_response)

    # Verify
    assert result == "Test response"


def test_process_llm_response_single_with_schema():
    """Test processing a single LLM response with schema validation."""
    # Setup
    mock_response = MagicMock()
    mock_response.choices[0].message.content = '{"message": "Test response"}'

    # Execute
    result = process_llm_response(mock_response, SimpleOutputSchema)

    # Verify
    assert isinstance(result, SimpleOutputSchema)
    assert result.message == "Test response"


def test_process_llm_response_batch():
    """Test processing a batch of LLM responses."""
    # Setup
    mock_response1 = MagicMock()
    mock_response1.choices[0].message.content = "Response 1"
    mock_response2 = MagicMock()
    mock_response2.choices[0].message.content = "Response 2"
    mock_responses = [mock_response1, mock_response2]

    # Execute
    results = process_llm_response(mock_responses)

    # Verify
    assert len(results) == 2
    assert results[0] == "Response 1"
    assert results[1] == "Response 2"


def test_process_llm_response_batch_with_schema():
    """Test processing a batch of LLM responses with schema validation."""
    # Setup
    mock_response1 = MagicMock()
    mock_response1.choices[0].message.content = '{"message": "Response 1"}'
    mock_response2 = MagicMock()
    mock_response2.choices[0].message.content = '{"message": "Response 2"}'
    mock_responses = [mock_response1, mock_response2]

    # Execute
    results = process_llm_response(mock_responses, SimpleOutputSchema)

    # Verify
    assert len(results) == 2
    assert all(isinstance(result, SimpleOutputSchema) for result in results)
    assert [result.message for result in results] == ["Response 1", "Response 2"]


def test_process_llm_response_invalid_schema():
    """Test processing a response with invalid schema data."""
    # Setup
    mock_response = MagicMock()
    mock_response.choices[0].message.content = '{"invalid": "data"}'

    # Execute and verify
    with pytest.raises(Exception):
        process_llm_response(mock_response, SimpleOutputSchema)


def test_process_llm_response_invalid_json():
    """Test processing a response with invalid JSON."""
    # Setup
    mock_response = MagicMock()
    mock_response.choices[0].message.content = "Not valid JSON"

    # Execute and verify
    with pytest.raises(Exception):
        process_llm_response(mock_response, SimpleOutputSchema)
