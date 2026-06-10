"""
api.py
------
FastAPI application exposing the macro platform over HTTP.

Routes:
  GET  /health                    — liveness check
  GET  /context                   — fetch current MacroContext (market + indicators)
  POST /interpret                 — full LLM interpretation of current macro context
  GET  /interpret/indicator/{id}  — interpret a single indicator
  GET  /briefing/stream           — SSE stream of the narrative macro briefing
  GET  /indicators                — list available indicators and their latest values
  GET  /regime/history            — placeholder for stored regime history

Run locally:
  uvicorn api:app --reload --port 8000
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime
from typing import AsyncIterator

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from data.fetcher import build_macro_context
from data.models import IndicatorType, MacroContext, MacroInterpretation
from llm.interpreter import interpret, interpret_single_indicator, stream_briefing

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title       = "Macro Analytics Platform",
    description = "AI-powered macro indicator interpretation and regime analysis",
    version     = "0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["http://localhost:5173", "http://localhost:3000"],
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)

# ---------------------------------------------------------------------------
# Simple in-memory cache: avoids re-fetching Yahoo Finance on every request
# In production: replace with Redis with a 5-minute TTL
# ---------------------------------------------------------------------------

_context_cache: dict = {"ctx": None, "fetched_at": None}
CACHE_TTL_SECONDS = 300   # 5 minutes


def _get_cached_context() -> MacroContext:
    now = datetime.utcnow()
    cached = _context_cache
    if (
        cached["ctx"] is not None
        and cached["fetched_at"] is not None
        and (now - cached["fetched_at"]).total_seconds() < CACHE_TTL_SECONDS
    ):
        return cached["ctx"]

    logger.info("Fetching fresh MacroContext from Yahoo Finance")
    ctx = build_macro_context()
    _context_cache["ctx"]        = ctx
    _context_cache["fetched_at"] = now
    return ctx


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    return {"status": "ok", "timestamp": datetime.utcnow().isoformat()}


@app.get("/context")
def get_context():
    """
    Return the current MacroContext as a JSON summary.
    This is what the frontend uses to populate the data panel
    before the LLM interpretation arrives.
    """
    ctx = _get_cached_context()
    return ctx.summary_dict()


@app.get("/indicators")
def list_indicators():
    """
    Return all available indicators with their latest values and metadata.
    Used by the frontend to populate the indicator selector.
    """
    ctx = _get_cached_context()
    result = []
    for series in ctx.indicators:
        latest = series.latest
        result.append({
            "id":          series.indicator.value,
            "name":        series.name,
            "unit":        series.unit,
            "source":      series.source,
            "latest_value": latest.value  if latest else None,
            "latest_date":  str(latest.as_of) if latest else None,
            "mom_change":   series.mom_change,
            "trend_3m":     series.three_month_trend,
            "n_observations": len(series.points),
        })
    return {
        "indicators":    result,
        "market":        ctx.summary_dict()["market"],
        "data_warnings": ctx.data_warnings,
        "as_of":         ctx.as_of.isoformat(),
    }


@app.post("/interpret")
def run_interpretation(force_refresh: bool = Query(False)):
    """
    Run the full LLM interpretation of the current macro context.

    This is the main endpoint. Returns:
    - Per-indicator interpretations (trend, signal, implications)
    - Holistic regime assessment
    - Structured forecast inputs for downstream models
    - Executive summary and key takeaways

    Set force_refresh=true to bypass the data cache.
    """
    if force_refresh:
        _context_cache["ctx"] = None

    ctx = _get_cached_context()

    try:
        result: MacroInterpretation = interpret(ctx)
    except Exception as e:
        logger.error(f"Interpretation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    return result.model_dump(mode="json")


@app.get("/interpret/indicator/{indicator_id}")
def interpret_indicator_route(indicator_id: str):
    """
    Interpret a single indicator.
    indicator_id must be a valid IndicatorType value, e.g. 'cpi', 'unemployment'.

    Faster than the full /interpret endpoint -- useful for the frontend
    to refresh one panel without re-running the whole pipeline.
    """
    valid_ids = [t.value for t in IndicatorType]
    if indicator_id not in valid_ids:
        raise HTTPException(
            status_code=422,
            detail=f"Unknown indicator '{indicator_id}'. Valid: {valid_ids}"
        )

    ctx    = _get_cached_context()
    result = interpret_single_indicator(ctx, indicator_id)

    if "error" in result:
        raise HTTPException(status_code=404, detail=result["error"])

    return result


@app.get("/briefing/stream")
def stream_briefing_route():
    """
    SSE endpoint: streams a narrative macro briefing as it is generated.

    Frontend usage (EventSource):
        const es = new EventSource('/briefing/stream');
        es.onmessage = e => appendText(e.data);

    Each SSE event contains one text chunk.
    A final event with data='[DONE]' signals completion.
    """
    ctx = _get_cached_context()

    def event_generator():
        try:
            for chunk in stream_briefing(ctx):
                # SSE format: each message is "data: <content>\n\n"
                safe_chunk = chunk.replace("\n", " ")
                yield f"data: {safe_chunk}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as e:
            yield f"data: [ERROR] {str(e)}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type = "text/event-stream",
        headers    = {
            "Cache-Control":               "no-cache",
            "X-Accel-Buffering":           "no",   # disable nginx buffering
            "Access-Control-Allow-Origin": "*",
        },
    )


@app.get("/forecast-inputs")
def get_forecast_inputs():
    """
    Return only the ForecastInputs component of the latest regime assessment.
    This is the structured data consumed by downstream quant models --
    the LLM's output translated into model-ready parameters.
    """
    ctx = _get_cached_context()
    result: MacroInterpretation = interpret(ctx)
    fi = result.regime_assessment.forecast_inputs

    return {
        "regime":        result.regime_assessment.regime.value,
        "confidence":    result.regime_assessment.regime_confidence,
        "forecast_inputs": fi.model_dump(),
        "generated_at":  result.as_of.isoformat(),
        "is_stub":       result.is_stub,
    }


@app.get("/regime/history")
def regime_history():
    """
    Placeholder: returns a stub regime history.
    In production: query a PostgreSQL table of stored MacroInterpretation runs.
    """
    return {
        "message": "Regime history requires a database backend (PostgreSQL). "
                   "Wire up SQLAlchemy + Alembic and store MacroInterpretation "
                   "objects on each /interpret call.",
        "stub_history": [
            {"date": "2025-03-01", "regime": "disinflation",   "confidence": 0.71},
            {"date": "2025-04-01", "regime": "disinflation",   "confidence": 0.74},
            {"date": "2025-05-01", "regime": "disinflation",   "confidence": 0.69},
            {"date": "2025-06-01", "regime": "expansion",      "confidence": 0.55},
        ]
    }