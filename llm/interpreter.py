"""High-level macro interpretation helpers backed by AIHubMix."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from llm.client import AIHubMixClient
from llm.prompts import (
    INDICATOR_SYSTEM_PROMPT,
    REGIME_SYSTEM_PROMPT,
    build_indicator_messages,
    build_regime_messages,
    build_stream_regime_messages,
)


ChatMessage = Mapping[str, Any]


@dataclass
class MacroInterpreter:
    """Translate macro inputs into LLM-generated analysis."""

    client: AIHubMixClient = field(default_factory=AIHubMixClient.from_env)

    def interpret_indicator(
        self,
        *,
        indicator_name: str,
        indicator_unit: str,
        current_value: float,
        mom_change: float | None,
        trend_3m: float | None,
        recent_history: list[dict],
        market_context: dict,
        model: str | None = None,
    ) -> str:
        """Interpret a single macro indicator using the AIHubMix provider."""

        messages = with_system_prompt(
            INDICATOR_SYSTEM_PROMPT,
            build_indicator_messages(
                indicator_name=indicator_name,
                indicator_unit=indicator_unit,
                current_value=current_value,
                mom_change=mom_change,
                trend_3m=trend_3m,
                recent_history=recent_history,
                market_context=market_context,
            ),
        )
        return self.client.chat_text(messages, model=model)

    def assess_regime(
        self,
        context_summary: dict,
        *,
        model: str | None = None,
    ) -> str:
        """Synthesize the full macro context into a regime assessment."""

        messages = with_system_prompt(
            REGIME_SYSTEM_PROMPT,
            build_regime_messages(context_summary),
        )
        return self.client.chat_text(messages, model=model)

    def write_regime_briefing(
        self,
        context_summary: dict,
        *,
        model: str | None = None,
    ) -> str:
        """Create a narrative macro briefing for streaming/UI-style display."""

        messages = with_system_prompt(
            REGIME_SYSTEM_PROMPT,
            build_stream_regime_messages(context_summary),
        )
        return self.client.chat_text(messages, model=model)


def with_system_prompt(
    system_prompt: str,
    messages: Sequence[ChatMessage],
) -> list[dict[str, Any]]:
    """Prepend a system prompt to a list of chat messages."""

    return [{"role": "system", "content": system_prompt}, *[dict(m) for m in messages]]
