"""FastAPI server and command-line smoke test for the macro platform."""

from __future__ import annotations

import argparse

from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel, Field

from data.fetcher import get_market_context
from llm.client import AIHubMixClient
from llm.interpreter import MacroInterpreter


app = FastAPI(
    title="Macro Platform",
    description=(
        "AIHubMix-powered macro analysis demo. Use /context for market "
        "context, /chat for a simple LLM call, and /interpret-indicator for "
        "macro indicator analysis."
    ),
    version="0.1.0",
)


class ChatRequest(BaseModel):
    prompt: str = Field(..., min_length=1, examples=["Say hello in one sentence"])
    model: str | None = Field(
        default=None,
        examples=["gpt-4o-mini"],
        description="Optional AIHubMix model override.",
    )


class ChatResponse(BaseModel):
    response: str


class IndicatorPoint(BaseModel):
    as_of: str = Field(..., examples=["2026-03"])
    value: float = Field(..., examples=[3.4])


class InterpretIndicatorRequest(BaseModel):
    indicator_name: str = Field(..., examples=["CPI Inflation"])
    indicator_unit: str = Field(..., examples=["%"])
    current_value: float = Field(..., examples=[3.4])
    mom_change: float | None = Field(default=None, examples=[0.2])
    trend_3m: float | None = Field(default=None, examples=[0.1])
    recent_history: list[IndicatorPoint] = Field(
        default_factory=list,
        examples=[
            [
                {"as_of": "2026-01", "value": 3.1},
                {"as_of": "2026-02", "value": 3.2},
                {"as_of": "2026-03", "value": 3.4},
            ]
        ],
    )
    market_context: dict = Field(
        default_factory=dict,
        examples=[
            {
                "fed_funds_rate": 5.25,
                "ten_year_yield": 4.4,
                "equity_market": "near highs",
            }
        ],
    )
    model: str | None = Field(default=None, examples=["gpt-4o-mini"])


class InterpretIndicatorResponse(BaseModel):
    analysis: str


def get_interpreter() -> MacroInterpreter:
    """FastAPI dependency for the macro interpreter."""

    return MacroInterpreter()


def get_context_payload() -> dict:
    """FastAPI dependency for market context."""

    return get_market_context()


def chat(prompt: str, *, model: str | None = None) -> str:
    """Send a single user prompt to AIHubMix and return assistant text."""

    client = AIHubMixClient.from_env()
    return client.chat_text(
        [{"role": "user", "content": prompt}],
        model=model,
    )


@app.get("/")
def root() -> dict:
    """Explain where to find the interactive demo."""

    return {
        "message": "Macro Platform API is running.",
        "docs": "/docs",
        "endpoints": ["/context", "/chat", "/interpret-indicator"],
    }


@app.get("/health")
def health() -> dict:
    """Simple health check for server demos."""

    return {"status": "ok"}


@app.get("/context")
def context(payload: dict = Depends(get_context_payload)) -> dict:
    """Return delayed public market data for macro context."""

    return payload


@app.post("/chat", response_model=ChatResponse)
def chat_endpoint(payload: ChatRequest) -> ChatResponse:
    """Send a prompt to AIHubMix and return the assistant response."""

    try:
        return ChatResponse(response=chat(payload.prompt, model=payload.model))
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@app.post("/interpret-indicator", response_model=InterpretIndicatorResponse)
def interpret_indicator_endpoint(
    payload: InterpretIndicatorRequest,
    interpreter: MacroInterpreter = Depends(get_interpreter),
) -> InterpretIndicatorResponse:
    """Interpret a macro indicator with current market context."""

    try:
        analysis = interpreter.interpret_indicator(
            indicator_name=payload.indicator_name,
            indicator_unit=payload.indicator_unit,
            current_value=payload.current_value,
            mom_change=payload.mom_change,
            trend_3m=payload.trend_3m,
            recent_history=[point.model_dump() for point in payload.recent_history],
            market_context=payload.market_context,
            model=payload.model,
        )
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    return InterpretIndicatorResponse(analysis=analysis)


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
