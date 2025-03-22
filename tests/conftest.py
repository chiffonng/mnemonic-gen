"""Configuration for pytest fixtures."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest
from pydantic import BaseModel


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
