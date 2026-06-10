from __future__ import annotations

import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from api import app, get_context_payload, get_interpreter


class FakeInterpreter:
    def interpret_indicator(self, **kwargs):
        self.kwargs = kwargs
        return "fake CPI analysis"


class APIServerTests(unittest.TestCase):
    def setUp(self) -> None:
        app.dependency_overrides.clear()
        self.client = TestClient(app)

    def tearDown(self) -> None:
        app.dependency_overrides.clear()

    def test_docs_are_available(self) -> None:
        response = self.client.get("/docs")

        self.assertEqual(response.status_code, 200)
        self.assertIn("text/html", response.headers["content-type"])

    def test_context_endpoint_returns_market_payload(self) -> None:
        app.dependency_overrides[get_context_payload] = lambda: {
            "status": "ok",
            "source": "test",
            "market": {"quotes": [{"symbol": "SPY.US", "close": 500.0}]},
        }

        response = self.client.get("/context")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["market"]["quotes"][0]["symbol"], "SPY.US")

    def test_chat_endpoint_uses_chat_helper(self) -> None:
        with patch("api.chat", return_value="hello from server") as fake_chat:
            response = self.client.post(
                "/chat",
                json={"prompt": "Say hello", "model": "gpt-4o-mini"},
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"response": "hello from server"})
        fake_chat.assert_called_once_with("Say hello", model="gpt-4o-mini")

    def test_interpret_indicator_endpoint_uses_interpreter(self) -> None:
        fake_interpreter = FakeInterpreter()
        app.dependency_overrides[get_interpreter] = lambda: fake_interpreter

        response = self.client.post(
            "/interpret-indicator",
            json={
                "indicator_name": "CPI Inflation",
                "indicator_unit": "%",
                "current_value": 3.4,
                "mom_change": 0.2,
                "trend_3m": 0.1,
                "recent_history": [{"as_of": "2026-03", "value": 3.4}],
                "market_context": {"fed_funds_rate": 5.25},
            },
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"analysis": "fake CPI analysis"})
        self.assertEqual(fake_interpreter.kwargs["indicator_name"], "CPI Inflation")
        self.assertEqual(fake_interpreter.kwargs["recent_history"][0]["value"], 3.4)


if __name__ == "__main__":
    unittest.main()
