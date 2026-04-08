"""Anthropic Claude API client for RAG responses."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import anthropic

logger = logging.getLogger(__name__)


@dataclass
class LLMResponse:
    """Response from the LLM."""

    content: str
    model: str
    input_tokens: int
    output_tokens: int
    latency_seconds: float
    stop_reason: str


class LLMClient:
    """Wrapper around Anthropic SDK for RAG queries."""

    def __init__(self, api_key: str, model: str = "claude-opus-4-6") -> None:
        if not api_key:
            raise ValueError(
                "ANTHROPIC_API_KEY is required. " "Set RAG_ANTHROPIC_API_KEY or ANTHROPIC_API_KEY environment variable."
            )
        self._client = anthropic.Anthropic(api_key=api_key)
        self._model = model

    def generate(
        self,
        system_prompt: str,
        user_message: str,
        max_tokens: int = 4096,
    ) -> LLMResponse:
        """Send a message to Claude and return the response."""
        t0 = time.time()

        response = self._client.messages.create(
            model=self._model,
            max_tokens=max_tokens,
            system=system_prompt,
            messages=[
                {"role": "user", "content": user_message},
            ],
        )

        elapsed = time.time() - t0

        content = ""
        for block in response.content:
            if block.type == "text":
                content += block.text

        result = LLMResponse(
            content=content,
            model=response.model,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            latency_seconds=round(elapsed, 2),
            stop_reason=response.stop_reason,
        )

        logger.info(
            "LLM response: %d input tokens, %d output tokens, %.1fs",
            result.input_tokens,
            result.output_tokens,
            result.latency_seconds,
        )
        return result
