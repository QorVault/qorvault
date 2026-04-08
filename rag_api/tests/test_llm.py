"""Tests for the Anthropic LLM client."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from rag_api.llm import LLMClient, LLMResponse


def test_missing_api_key_raises():
    with pytest.raises(ValueError, match="ANTHROPIC_API_KEY is required"):
        LLMClient(api_key="")


def test_missing_api_key_none_raises():
    with pytest.raises(ValueError, match="ANTHROPIC_API_KEY is required"):
        LLMClient(api_key="")


@patch("rag_api.llm.anthropic.Anthropic")
def test_generate_calls_anthropic_correctly(mock_anthropic_cls):
    mock_client = MagicMock()
    mock_anthropic_cls.return_value = mock_client

    # Build a mock response
    mock_text_block = MagicMock()
    mock_text_block.type = "text"
    mock_text_block.text = "The answer is 42."

    mock_response = MagicMock()
    mock_response.content = [mock_text_block]
    mock_response.model = "claude-opus-4-6"
    mock_response.usage.input_tokens = 100
    mock_response.usage.output_tokens = 20
    mock_response.stop_reason = "end_turn"

    mock_client.messages.create.return_value = mock_response

    client = LLMClient(api_key="sk-test-key", model="claude-opus-4-6")
    result = client.generate(
        system_prompt="You are helpful.",
        user_message="What is the meaning of life?",
        max_tokens=1024,
    )

    mock_client.messages.create.assert_called_once()
    call_kwargs = mock_client.messages.create.call_args[1]
    assert call_kwargs["model"] == "claude-opus-4-6"
    assert call_kwargs["max_tokens"] == 1024
    assert call_kwargs["system"] == "You are helpful."
    assert call_kwargs["messages"][0]["role"] == "user"


@patch("rag_api.llm.anthropic.Anthropic")
def test_generate_extracts_text_from_response(mock_anthropic_cls):
    mock_client = MagicMock()
    mock_anthropic_cls.return_value = mock_client

    mock_text_block = MagicMock()
    mock_text_block.type = "text"
    mock_text_block.text = "Hello world"

    mock_response = MagicMock()
    mock_response.content = [mock_text_block]
    mock_response.model = "claude-opus-4-6"
    mock_response.usage.input_tokens = 50
    mock_response.usage.output_tokens = 10
    mock_response.stop_reason = "end_turn"

    mock_client.messages.create.return_value = mock_response

    client = LLMClient(api_key="sk-test-key")
    result = client.generate("system", "user msg")

    assert isinstance(result, LLMResponse)
    assert result.content == "Hello world"
    assert result.input_tokens == 50
    assert result.output_tokens == 10
    assert result.model == "claude-opus-4-6"


@patch("rag_api.llm.anthropic.Anthropic")
def test_generate_tracks_latency(mock_anthropic_cls):
    mock_client = MagicMock()
    mock_anthropic_cls.return_value = mock_client

    mock_text_block = MagicMock()
    mock_text_block.type = "text"
    mock_text_block.text = "Response"

    mock_response = MagicMock()
    mock_response.content = [mock_text_block]
    mock_response.model = "claude-opus-4-6"
    mock_response.usage.input_tokens = 10
    mock_response.usage.output_tokens = 5
    mock_response.stop_reason = "end_turn"

    mock_client.messages.create.return_value = mock_response

    client = LLMClient(api_key="sk-test-key")
    result = client.generate("system", "user msg")

    assert result.latency_seconds >= 0
