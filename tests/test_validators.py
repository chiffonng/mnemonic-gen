"""Test the data validators."""

import json
from unittest.mock import patch

import pytest
from pydantic import ValidationError
from src.data_prep.data_validators import (
    _attempt_fix_incomplete_json,
    validate_content_against_schema,
)

from tests.conftest import SimpleOutputSchema


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
        "src.data_prep.data_validators._attempt_fix_incomplete_json",
        return_value='{"message" "Test }: message"}',
    ):
        # Execute
        with pytest.raises((ValidationError, json.JSONDecodeError)):
            validate_content_against_schema(content, SimpleOutputSchema)


def test_validate_content_against_schema_invalid_content():
    """Test error when content doesn't match schema."""
    content = '{"wrong_field": "Test message"}'

    with pytest.raises(ValidationError, match=r".*validation.*error"):
        validate_content_against_schema(content, SimpleOutputSchema)


def test_validate_content_against_schema_fix_json():
    """Test that it attempts to fix incomplete JSON."""
    content = '{"message": "Test message"'  # Missing closing brace

    # Execute with patched _attempt_fix_incomplete_json
    with patch(
        "src.data_prep.data_validators._attempt_fix_incomplete_json",
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
