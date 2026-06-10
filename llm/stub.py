"""Test doubles for the LLM layer."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence


@dataclass
class StubLLMClient:
    """In-memory stand-in for :class:`llm.client.AIHubMixClient`.

    The stub records calls and returns a predictable assistant response, which
    keeps unit tests independent from external API credentials.
    """

    response_text: str = "stubbed response"
    calls: list[dict[str, Any]] = field(default_factory=list)

    def chat_completion(
        self,
        messages: Sequence[Mapping[str, Any]],
        *,
        model: str | None = None,
        temperature: float | None = 0.2,
        max_tokens: int | None = None,
        response_format: Mapping[str, Any] | None = None,
        extra: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        self.calls.append(
            {
                "messages": [dict(message) for message in messages],
                "model": model,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "response_format": response_format,
                "extra": dict(extra) if extra else None,
            }
        )
        return {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": self.response_text,
                    }
                }
            ]
        }

    def chat_text(
        self,
        messages: Sequence[Mapping[str, Any]],
        *,
        model: str | None = None,
        temperature: float | None = 0.2,
        max_tokens: int | None = None,
        response_format: Mapping[str, Any] | None = None,
        extra: Mapping[str, Any] | None = None,
    ) -> str:
        self.chat_completion(
            messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format=response_format,
            extra=extra,
        )
        return self.response_text
