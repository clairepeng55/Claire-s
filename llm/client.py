"""AIHubMix chat-completions client.

AIHubMix exposes an OpenAI-compatible API at https://aihubmix.com/v1.  This
module keeps that provider wiring in one place and reads secrets from the
environment so API keys are never committed to the repository.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from typing import Any, Mapping, MutableMapping, Sequence
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


DEFAULT_BASE_URL = "https://aihubmix.com/v1"
DEFAULT_MODEL = "gpt-4o-mini"


class AIHubMixError(RuntimeError):
    """Raised when the AIHubMix API cannot return a usable response."""


@dataclass(frozen=True)
class AIHubMixConfig:
    """Runtime configuration for AIHubMix API calls."""

    api_key: str
    base_url: str = DEFAULT_BASE_URL
    model: str = DEFAULT_MODEL
    timeout_seconds: float = 60.0

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> "AIHubMixConfig":
        """Build configuration from environment variables.

        Required:
            AIHUBMIX_API_KEY

        Optional:
            AIHUBMIX_BASE_URL (defaults to https://aihubmix.com/v1)
            AIHUBMIX_MODEL (defaults to gpt-4o-mini)
            AIHUBMIX_TIMEOUT_SECONDS (defaults to 60)
        """

        env = os.environ if environ is None else environ
        api_key = env.get("AIHUBMIX_API_KEY", "").strip()
        if not api_key:
            raise AIHubMixError(
                "AIHUBMIX_API_KEY is not set. Set it to the key from "
                "https://console.aihubmix.com/token before making LLM calls."
            )

        timeout_raw = env.get("AIHUBMIX_TIMEOUT_SECONDS", "60").strip()
        try:
            timeout_seconds = float(timeout_raw)
        except ValueError as exc:
            raise AIHubMixError(
                "AIHUBMIX_TIMEOUT_SECONDS must be a number."
            ) from exc

        return cls(
            api_key=api_key,
            base_url=env.get("AIHUBMIX_BASE_URL", DEFAULT_BASE_URL).strip()
            or DEFAULT_BASE_URL,
            model=env.get("AIHUBMIX_MODEL", DEFAULT_MODEL).strip()
            or DEFAULT_MODEL,
            timeout_seconds=timeout_seconds,
        )


class AIHubMixClient:
    """Small OpenAI-compatible client for AIHubMix chat completions."""

    def __init__(self, config: AIHubMixConfig | None = None) -> None:
        self.config = config or AIHubMixConfig.from_env()

    @classmethod
    def from_env(cls) -> "AIHubMixClient":
        """Create a client using AIHubMix environment variables."""

        return cls(AIHubMixConfig.from_env())

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
        """Call the AIHubMix chat-completions endpoint.

        Args:
            messages: OpenAI-compatible chat messages.
            model: Optional model override. Defaults to AIHUBMIX_MODEL.
            temperature: Sampling temperature. Pass None to omit it.
            max_tokens: Optional output token limit.
            response_format: Optional OpenAI-compatible response_format object.
            extra: Additional provider parameters to merge into the request.
        """

        if not messages:
            raise ValueError("messages must contain at least one chat message")

        payload: MutableMapping[str, Any] = {
            "model": model or self.config.model,
            "messages": list(messages),
        }
        if temperature is not None:
            payload["temperature"] = temperature
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if response_format is not None:
            payload["response_format"] = response_format
        if extra:
            payload.update(extra)

        return self._post_json("/chat/completions", payload)

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
        """Return the assistant message content from a chat completion."""

        response = self.chat_completion(
            messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format=response_format,
            extra=extra,
        )

        try:
            content = response["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise AIHubMixError(
                f"AIHubMix response did not include assistant content: {response!r}"
            ) from exc

        if isinstance(content, str):
            return content
        return json.dumps(content, ensure_ascii=False)

    def _post_json(self, path: str, payload: Mapping[str, Any]) -> dict[str, Any]:
        url = f"{self.config.base_url.rstrip('/')}{path}"
        body = json.dumps(payload).encode("utf-8")
        request = Request(
            url,
            data=body,
            headers={
                "Authorization": f"Bearer {self.config.api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
            method="POST",
        )

        try:
            with urlopen(request, timeout=self.config.timeout_seconds) as response:
                raw = response.read().decode("utf-8")
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise AIHubMixError(
                f"AIHubMix API returned HTTP {exc.code}: {detail}"
            ) from exc
        except URLError as exc:
            raise AIHubMixError(f"Could not reach AIHubMix API: {exc.reason}") from exc

        try:
            decoded = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise AIHubMixError(
                f"AIHubMix API returned invalid JSON: {raw[:500]}"
            ) from exc

        if not isinstance(decoded, dict):
            raise AIHubMixError(f"AIHubMix API returned unexpected JSON: {decoded!r}")
        return decoded
