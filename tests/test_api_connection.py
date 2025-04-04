# scripts/test_api_connection.py
"""Test API connections to external providers like DeepSeek."""

import os

import pytest
from dotenv import load_dotenv
from openai import OpenAI


@pytest.fixture(scope="module")
def deepseek_client():
    """Create a DeepSeek client using the OpenAI client with custom base URL."""
    load_dotenv()

    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        pytest.skip("DEEPSEEK_API_KEY not found in environment variables")

    client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
    return client


def test_deepseek_reasoner(deepseek_client):
    """Test completion with DeepSeek API."""
    try:
        response = deepseek_client.chat.completions.create(
            model="deepseek-reasoner",
            messages=[
                {"role": "system", "content": "You are a helpful assistant"},
                {"role": "user", "content": "Say hello in one word"},
            ],
            max_tokens=10,
            stream=False,
        )

        # Check if we got a response
        assert response is not None
        assert hasattr(response, "choices")
        assert len(response.choices) > 0
        assert hasattr(response.choices[0], "message")
        assert hasattr(response.choices[0].message, "content")

        # Print the response content
        content = response.choices[0].message.content
        print(f"DeepSeek API response: '{content}'")
        assert len(content) > 0

    except Exception as e:
        pytest.fail(f"DeepSeek API chat completion test failed: {str(e)}")


def test_deepseek_list_models(deepseek_client):
    """Test listing models with DeepSeek API."""
    try:
        models = deepseek_client.models.list()

        # Check if we got a response
        assert models is not None
        assert hasattr(models, "data")
        assert len(models.data) > 0

        # Print available models
        print("Available DeepSeek models:")
        for model in models.data:
            print(f"- {model.id}")

    except Exception as e:
        pytest.fail(f"DeepSeek API list models test failed: {str(e)}")


if __name__ == "__main__":
    # Allow running as a script for quick testing
    pytest.main(["-xvs", __file__])
