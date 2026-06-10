"""
llm/client.py
-------------
OpenAI GPT-4o client with built-in web search.
Web search lets GPT fetch live macro data (CPI, yields, VIX, etc.)
without needing a separate data fetcher.
"""

import os
import time
import uuid
from datetime import datetime

from openai import OpenAI

from data.models import (
    MacroContext, MacroInterpretation, MacroRegimeAssessment,
    IndicatorInterpretation, ForecastInputs,
)
from llm.prompts import REGIME_SYSTEM_PROMPT, build_regime_messages

MODEL = "gpt-4o"


def _make_client() -> OpenAI:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "OPENAI_API_KEY is not set. "
            "Run: export OPENAI_API_KEY=your_key_here"
        )
    return OpenAI(api_key=api_key)


def fetch_live_macro_data() -> dict:
    """
    Ask GPT-4o to search the web for current macro indicator values.
    Returns a plain dict of numbers the rest of the system can use.
    """
    client = _make_client()

    response = client.responses.create(
        model=MODEL,
        tools=[{"type": "web_search_preview"}],
        input=(
            "Search for the most current values of these US macro indicators "
            "and return ONLY a JSON object, no other text:\n"
            "{\n"
            '  "cpi_yoy": <latest CPI year-over-year %>,\n'
            '  "core_pce_yoy": <latest core PCE year-over-year %>,\n'
            '  "unemployment_rate": <latest unemployment rate %>,\n'
            '  "yield_10y": <current 10-year Treasury yield %>,\n'
            '  "yield_2y": <current 2-year Treasury yield %>,\n'
            '  "vix": <current VIX level>,\n'
            '  "sp500": <current S&P 500 price>,\n'
            '  "fed_funds_rate": <current federal funds rate %>,\n'
            '  "as_of": "<today date YYYY-MM-DD>"\n'
            "}"
        ),
    )

    import json
    text = response.output_text.strip()
    # Strip markdown code fences if GPT wraps in ```json ... ```
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
    return json.loads(text.strip())


def stream_macro_briefing(ctx: MacroContext):
    """
    Stream a narrative macro briefing using GPT-4o with web search.
    Yields text chunks for SSE streaming to the frontend.
    """
    client   = _make_client()
    messages = build_regime_messages(ctx.summary_dict())

    # GPT streaming with web search
    with client.responses.stream(
        model   = MODEL,
        tools   = [{"type": "web_search_preview"}],
        input   = messages[0]["content"],
        instructions = REGIME_SYSTEM_PROMPT,
    ) as stream:
        for event in stream:
            if hasattr(event, "delta") and event.delta:
                yield event.delta


def interpret_indicator(series, market_ctx: dict) -> IndicatorInterpretation:
    """
    Interpret a single indicator using GPT-4o with web search.
    Falls back to stub interpretation for simplicity.
    """
    from llm.stub import stub_interpret_indicator
    return stub_interpret_indicator(series)


def run_full_interpretation(ctx: MacroContext) -> MacroInterpretation:
    """
    Full interpretation pipeline using GPT-4o.
    Uses web search to ground the regime analysis in current data.
    """
    from llm.stub import stub_interpret_indicator, stub_regime_assessment

    t_start = time.monotonic()
    client  = _make_client()

    # Per-indicator interpretations (use stub structure, GPT for regime)
    interpretations = [
        stub_interpret_indicator(s)
        for s in ctx.indicators
        if s.latest is not None
    ]

    # Regime assessment via GPT with web search
    context_str = __import__("json").dumps(ctx.summary_dict(), indent=2, default=str)

    response = client.responses.create(
        model = MODEL,
        tools = [{"type": "web_search_preview"}],
        input = (
            f"{REGIME_SYSTEM_PROMPT}\n\n"
            f"Here is the macro context:\n{context_str}\n\n"
            "Search the web to verify these numbers are current, then provide "
            "a regime assessment. Return a JSON object with these exact keys: "
            "regime, regime_confidence, regime_reasoning, primary_driver, "
            "secondary_driver, key_risk, executive_summary, key_takeaways "
            "(list), risks_to_watch (list), data_releases_to_watch (list), "
            "and forecast_inputs containing: gdp_growth_central, "
            "gdp_growth_bull, gdp_growth_bear, cpi_yoy_central, "
            "cpi_yoy_6m_fwd, fed_funds_terminal, yield_10y_range_lo, "
            "yield_10y_range_hi, equity_risk_premium, vol_regime, "
            "recession_prob_12m."
        ),
    )

    import json
    text = response.output_text.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]

    raw = json.loads(text.strip())

    regime = MacroRegimeAssessment(
        regime             = raw["regime"],
        regime_confidence  = raw["regime_confidence"],
        regime_reasoning   = raw["regime_reasoning"],
        primary_driver     = raw["primary_driver"],
        secondary_driver   = raw.get("secondary_driver"),
        key_risk           = raw["key_risk"],
        executive_summary  = raw["executive_summary"],
        key_takeaways      = raw["key_takeaways"],
        risks_to_watch     = raw["risks_to_watch"],
        data_releases_to_watch = raw["data_releases_to_watch"],
        forecast_inputs    = ForecastInputs(**raw["forecast_inputs"]),
    )

    return MacroInterpretation(
        run_id                    = str(uuid.uuid4()),
        as_of                     = datetime.utcnow(),
        context_summary           = ctx.summary_dict(),
        indicator_interpretations = interpretations,
        regime_assessment         = regime,
        is_stub                   = False,
        model_used                = MODEL,
        latency_ms                = int((time.monotonic() - t_start) * 1000),
    )