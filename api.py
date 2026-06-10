"""Command-line smoke test for the AIHubMix chat client."""

from __future__ import annotations

import argparse

from llm.client import AIHubMixClient


def chat(prompt: str, *, model: str | None = None) -> str:
    """Send a single user prompt to AIHubMix and return assistant text."""

    client = AIHubMixClient.from_env()
    return client.chat_text(
        [{"role": "user", "content": prompt}],
        model=model,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Send a prompt to AIHubMix.")
    parser.add_argument("prompt", help="Prompt text to send")
    parser.add_argument(
        "--model",
        default=None,
        help="Optional AIHubMix model override, e.g. gpt-4o-mini",
    )
    args = parser.parse_args()
    print(chat(args.prompt, model=args.model))


if __name__ == "__main__":
    main()
