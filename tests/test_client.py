"""Tests for the client module."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from pydantic import BaseModel, ValidationError

from src.llms.client import (
    _attempt_fix_incomplete_json,
    batch_complete,
    build_input_params,
    complete,
    process_llm_response,
    process_llm_responses,
    validate_content_against_schema,
)


class SimpleOutputSchema(BaseModel):
    """Simple schema for testing."""

    message: str


class ComplexOutputSchema(BaseModel):
    """Complex schema with nested fields for testing."""

    message: str
    details: dict
    items: list


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


@pytest.fixture
def mock_batch_messages():
    """Create mock batch messages for testing."""
    return [
        [{"role": "user", "content": "Hello, world!"}],
        [{"role": "user", "content": "How are you?"}],
    ]


@pytest.fixture
def mock_llm_response():
    """Create a mock LLM response for testing."""
    response = MagicMock()
    response.choices = [MagicMock()]
    response.choices[0].message = MagicMock()
    response.choices[0].message.content = "Test response"
    response.usage = {"total_tokens": 10}
    return response


@pytest.fixture
def mock_llm_batch_responses():
    """Create mock batch LLM responses for testing."""
    response1 = MagicMock()
    response1.choices = [MagicMock()]
    response1.choices[0].message = MagicMock()
    response1.choices[0].message.content = "Response 1"

    response2 = MagicMock()
    response2.choices = [MagicMock()]
    response2.choices[0].message = MagicMock()
    response2.choices[0].message.content = "Response 2"

    return [response1, response2]


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
        assert "response_format" in params
        assert params["response_format"] == SimpleOutputSchema


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


def test_build_input_params_incompatible_options(mock_config_path, mock_messages):
    """Test error when both mock_response and output_schema are provided."""
    with pytest.raises(
        ValueError, match="Cannot use mock_response and output_schema at the same time"
    ):
        build_input_params(
            messages=mock_messages,
            config_path=mock_config_path,
            mock_response="Test response",
            output_schema=SimpleOutputSchema,
        )


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
    mock_validate.return_value = True
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
    mock_validate.return_value = True
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
@patch("src.llms.client.validate_environment")
def test_complete_validation_failure(mock_validate, mock_completion, mock_messages):
    """Test error handling when environment validation fails."""
    # Setup
    mock_validate.return_value = False

    # Execute and verify
    with pytest.raises(ValueError, match="Environment is NOT valid for model"):
        complete(messages=mock_messages)

    # Completion shouldn't be called if validation fails
    mock_completion.assert_not_called()


@patch("src.llms.client.process_llm_responses")
@patch("src.llms.client.batch_completion")
@patch("src.llms.client.validate_environment")
@patch("src.llms.client.build_input_params")
def test_batch_complete_success(
    mock_build_params,
    mock_validate,
    mock_batch_completion,
    mock_process,
    mock_batch_messages,
):
    """Test successful batch completion."""
    # Setup
    mock_build_params.return_value = {
        "model": "test-model",
        "messages": mock_batch_messages,
    }
    mock_validate.return_value = True
    mock_responses = [MagicMock(), MagicMock()]
    mock_process.return_value = ["Response 1", "Response 2"]
    mock_batch_completion.return_value = mock_responses

    # Execute
    result = batch_complete(messages=mock_batch_messages)

    # Verify
    mock_build_params.assert_called_once()
    mock_validate.assert_called_once_with(model="test-model")
    mock_batch_completion.assert_called_once_with(**mock_build_params.return_value)
    mock_process.assert_called_once_with(mock_responses, None)
    assert result == ["Response 1", "Response 2"]


def test_process_llm_response(mock_llm_response):
    """Test processing a single LLM response."""
    # Execute
    result = process_llm_response(mock_llm_response)

    # Verify
    assert result == "Test response"


def test_process_llm_response_with_schema(mock_llm_response):
    """Test processing a single LLM response with schema validation."""
    # Setup - modify the mock response to return valid JSON
    mock_llm_response.choices[0].message.content = '{"message": "Test response"}'

    # Execute
    result = process_llm_response(mock_llm_response, SimpleOutputSchema)

    # Verify
    assert isinstance(result, SimpleOutputSchema)
    assert result.message == "Test response"


def test_process_llm_response_none():
    """Test error when response is None."""
    # Execute and verify
    with pytest.raises(ValueError, match="Response is None"):
        process_llm_response(None)


def test_process_llm_response_invalid_schema(mock_llm_response):
    """Test error when output_schema is not a BaseModel."""
    # Execute and verify
    with pytest.raises(
        TypeError, match="output_schema must be a subclass of pydantic.BaseModel"
    ):
        process_llm_response(mock_llm_response, dict)


def test_process_llm_response_missing_content(mock_llm_response):
    """Test error when response has no content."""
    mock_llm_response.choices[0].message.content = None

    with pytest.raises(ValueError, match="Response content is None"):
        process_llm_response(mock_llm_response)


def test_process_llm_responses(mock_llm_batch_responses):
    """Test processing multiple LLM responses."""
    results = process_llm_responses(mock_llm_batch_responses)

    assert len(results) == 2
    assert results[0] == "Response 1"
    assert results[1] == "Response 2"


def test_process_llm_responses_with_schema(mock_llm_batch_responses):
    """Test processing multiple LLM responses with schema validation."""
    # Setup - modify the mock responses to return valid JSON
    mock_llm_batch_responses[0].choices[0].message.content = '{"message": "Response 1"}'
    mock_llm_batch_responses[1].choices[0].message.content = '{"message": "Response 2"}'

    results = process_llm_responses(mock_llm_batch_responses, SimpleOutputSchema)

    assert len(results) == 2
    assert all(isinstance(result, SimpleOutputSchema) for result in results)
    assert [result.message for result in results] == ["Response 1", "Response 2"]


def test_process_llm_responses_empty_list():
    """Test processing an empty list of responses."""
    results = process_llm_responses([])

    assert results == []


def test_process_llm_responses_with_errors(mock_llm_batch_responses):
    """Test processing responses with some errors."""
    # Setup - make the second response raise an error
    mock_llm_batch_responses[1].choices[0].message.content = None

    results = process_llm_responses(mock_llm_batch_responses)

    # Verify - should include None for the failed response
    assert len(results) == 2
    assert results[0] == "Response 1"
    assert results[1] is None


def test_validate_content_against_schema_string():
    """Test validating a JSON string against a schema."""
    content = '{"message": "Test message"}'
    result = validate_content_against_schema(content, SimpleOutputSchema)

    assert isinstance(result, SimpleOutputSchema)
    assert result.message == "Test message"


def test_validate_content_against_schema_dict():
    """Test validating a dictionary against a schema."""
    content = {"message": "Test message"}
    result = validate_content_against_schema(content, SimpleOutputSchema)

    assert isinstance(result, SimpleOutputSchema)
    assert result.message == "Test message"


def test_validate_content_against_schema_model_instance():
    """Test validating a model instance against a schema."""

    content = SimpleOutputSchema(message="Test message")
    result = validate_content_against_schema(content, SimpleOutputSchema)

    assert result is content  # Should return the same instance


def test_validate_content_against_schema_invalid_json():
    """Test error when content is invalid JSON and cannot be fixed."""
    # Setup - make a very broken JSON that even the fixer can't handle
    content = '{"message" "Test }: message"'  # Malformed beyond repair

    # Mock _attempt_fix_incomplete_json to return still-broken JSON
    with patch(
        "src.llms.client._attempt_fix_incomplete_json",
        return_value='{"message" "Test }: message"}',
    ):
        # Execute
        with pytest.raises((ValidationError, json.JSONDecodeError)):
            validate_content_against_schema(content, SimpleOutputSchema)


def test_validate_content_against_schema_invalid_model():
    """Test error when content doesn't match schema."""
    content = '{"wrong_field": "Test message"}'

    with pytest.raises(ValueError):
        validate_content_against_schema(content, SimpleOutputSchema)


def test_validate_content_against_schema_fix_json():
    """Test that it attempts to fix incomplete JSON."""
    content = '{"message": "Test message"'  # Missing closing brace

    # Execute with patched _attempt_fix_incomplete_json
    with patch(
        "src.llms.client._attempt_fix_incomplete_json",
        return_value='{"message": "Test message"}',
    ):
        result = validate_content_against_schema(content, SimpleOutputSchema)

    assert isinstance(result, SimpleOutputSchema)
    assert result.message == "Test message"


def test_attempt_fix_incomplete_json_braces():
    """Test fixing JSON with missing closing braces."""
    incomplete = '{"key": "value", "nested": {"key2": "value2"'
    fixed = _attempt_fix_incomplete_json(incomplete)

    assert fixed.endswith("}")
    assert fixed.count("{") == fixed.count("}")

    # Should be valid JSON now
    parsed = json.loads(fixed)
    assert parsed["key"] == "value"
    assert parsed["nested"]["key2"] == "value2"


def test_attempt_fix_incomplete_json_quotes():
    """Test fixing JSON with unclosed quotes."""
    incomplete = '{"key": "value", "message": "hello world'
    fixed = _attempt_fix_incomplete_json(incomplete)

    assert '"message": "hello world"' in fixed
    assert fixed.endswith("}")

    # Should be valid JSON now
    parsed = json.loads(fixed)
    assert parsed["key"] == "value"
    assert parsed["message"] == "hello world"


def test_attempt_fix_incomplete_json_arrays():
    """Test fixing JSON with unclosed arrays."""
    incomplete = '{"items": [1, 2, 3'
    fixed = _attempt_fix_incomplete_json(incomplete)

    assert fixed.endswith("]}")

    # Should be valid JSON now
    parsed = json.loads(fixed)
    assert parsed["items"] == [1, 2, 3]


def test_attempt_fix_incomplete_json_trailing_comma():
    """Test fixing JSON with trailing commas."""
    incomplete = '{"key": "value", "items": [1, 2, 3],'
    fixed = _attempt_fix_incomplete_json(incomplete)

    assert not fixed.rstrip().endswith(",")

    # Should be valid JSON now
    try:
        parsed = json.loads(fixed)
        assert parsed["key"] == "value"
        assert parsed["items"] == [1, 2, 3]
    except json.JSONDecodeError:
        pytest.fail("Fixed JSON should be valid")
