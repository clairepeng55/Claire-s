from __future__ import annotations

import json
import unittest
from unittest.mock import patch

from llm.client import (
    AIHubMixClient,
    AIHubMixConfig,
    AIHubMixError,
    DEFAULT_BASE_URL,
    DEFAULT_MODEL,
)
from llm.interpreter import MacroInterpreter, with_system_prompt
from llm.stub import StubLLMClient


class FakeResponse:
    def __init__(self, payload: dict) -> None:
        self.payload = payload

    def __enter__(self) -> "FakeResponse":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        return None

    def read(self) -> bytes:
        return json.dumps(self.payload).encode("utf-8")


class AIHubMixConfigTests(unittest.TestCase):
    def test_from_env_uses_aihubmix_defaults(self) -> None:
        config = AIHubMixConfig.from_env({"AIHUBMIX_API_KEY": "test-key"})

        self.assertEqual(config.api_key, "test-key")
        self.assertEqual(config.base_url, DEFAULT_BASE_URL)
        self.assertEqual(config.model, DEFAULT_MODEL)
        self.assertEqual(config.timeout_seconds, 60.0)

    def test_from_env_requires_api_key(self) -> None:
        with self.assertRaises(AIHubMixError):
            AIHubMixConfig.from_env({})


class AIHubMixClientTests(unittest.TestCase):
    def test_chat_completion_posts_openai_compatible_request(self) -> None:
        captured = {}

        def fake_urlopen(request, timeout):
            captured["url"] = request.full_url
            captured["headers"] = dict(request.header_items())
            captured["payload"] = json.loads(request.data.decode("utf-8"))
            captured["timeout"] = timeout
            return FakeResponse(
                {
                    "choices": [
                        {
                            "message": {
                                "role": "assistant",
                                "content": "hello from aihubmix",
                            }
                        }
                    ]
                }
            )

        config = AIHubMixConfig(
            api_key="test-key",
            base_url="https://aihubmix.com/v1/",
            model="gpt-4o-mini",
            timeout_seconds=12,
        )

        with patch("llm.client.urlopen", fake_urlopen):
            text = AIHubMixClient(config).chat_text(
                [{"role": "user", "content": "hello"}],
                temperature=0.1,
                max_tokens=20,
            )

        self.assertEqual(text, "hello from aihubmix")
        self.assertEqual(captured["url"], "https://aihubmix.com/v1/chat/completions")
        self.assertEqual(captured["headers"]["Authorization"], "Bearer test-key")
        self.assertEqual(captured["headers"]["Content-type"], "application/json")
        self.assertEqual(captured["headers"]["Accept"], "application/json")
        self.assertEqual(captured["timeout"], 12)
        self.assertEqual(
            captured["payload"],
            {
                "model": "gpt-4o-mini",
                "messages": [{"role": "user", "content": "hello"}],
                "temperature": 0.1,
                "max_tokens": 20,
            },
        )


class MacroInterpreterTests(unittest.TestCase):
    def test_with_system_prompt_prepends_system_message(self) -> None:
        messages = with_system_prompt(
            "system text",
            [{"role": "user", "content": "user text"}],
        )

        self.assertEqual(messages[0], {"role": "system", "content": "system text"})
        self.assertEqual(messages[1], {"role": "user", "content": "user text"})

    def test_interpret_indicator_uses_stub_client(self) -> None:
        stub = StubLLMClient(response_text="indicator analysis")
        interpreter = MacroInterpreter(client=stub)

        result = interpreter.interpret_indicator(
            indicator_name="CPI",
            indicator_unit="%",
            current_value=3.1,
            mom_change=0.2,
            trend_3m=0.1,
            recent_history=[{"as_of": "2026-05-01", "value": 3.1}],
            market_context={"fed_funds": 5.25},
        )

        self.assertEqual(result, "indicator analysis")
        self.assertEqual(stub.calls[0]["messages"][0]["role"], "system")
        self.assertIn("CPI", stub.calls[0]["messages"][1]["content"])


if __name__ == "__main__":
    unittest.main()
