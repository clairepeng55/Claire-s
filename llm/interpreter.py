"""
llm/interpreter.py
------------------
Single public entry point for the LLM layer.
 
The rest of the application (FastAPI routes, tests, CLI) only imports
from here -- never directly from client.py or stub.py.
 
Routing logic:
  1. If USE_STUB=true in env                       → stub
  2. If ANTHROPIC_API_KEY is not set               → stub (with warning)
  3. Otherwise                                     → live client
 
This means the application works out of the box without any config,
degrades gracefully when the key isn't present, and is trivially
switchable for testing.
"""
 
from __future__ import annotations
 
import logging
import os
from typing import Iterator
 
from data.models import MacroContext, MacroInterpretation
 
logger = logging.getLogger(__name__)
 
 
def _use_stub() -> bool:
    if os.environ.get("USE_STUB", "").lower() in ("1", "true", "yes"):
        return True
    if not os.environ.get("ANTHROPIC_API_KEY"):
        logger.warning(
            "ANTHROPIC_API_KEY not set — falling back to stub interpreter. "
            "Set the key and restart to use the live LLM."
        )
        return True
    return False
 
 
def interpret(ctx: MacroContext) -> MacroInterpretation:
    """
    Run the full macro interpretation pipeline.
 
    Automatically routes to the stub if no API key is configured.
    Returns a MacroInterpretation with is_stub=True/False accordingly.
    """
    if _use_stub():
        from llm.stub import run_stub_interpretation
        logger.info("Running stub interpretation")
        return run_stub_interpretation(ctx)
 
    from llm.client import run_full_interpretation
    logger.info("Running live LLM interpretation")
    return run_full_interpretation(ctx)
 
 
def stream_briefing(ctx: MacroContext) -> Iterator[str]:
    """
    Stream a narrative macro briefing as text chunks.
 
    Falls back to yielding the stub's executive_summary in one chunk
    if no API key is set (streaming doesn't make sense for stubs,
    but the interface stays consistent).
    """
    if _use_stub():
        from llm.stub import run_stub_interpretation
        result = run_stub_interpretation(ctx)
        # Yield the narrative in sentence-sized chunks to simulate streaming
        text = result.regime_assessment.executive_summary
        sentences = [s.strip() + " " for s in text.split(". ") if s.strip()]
        for sentence in sentences:
            yield sentence
        return
 
    from llm.client import stream_macro_briefing
    yield from stream_macro_briefing(ctx)
 
 
def interpret_single_indicator(
    ctx: MacroContext,
    indicator_value: str,
) -> dict:
    """
    Lightweight helper: interpret one indicator by name.
    Returns the IndicatorInterpretation as a dict, or None if not found.

    `indicator_value` is the IndicatorType enum value string,
    e.g. "cpi", "unemployment", "yield_10y".
    """
    from data.models import IndicatorType

    try:
        indicator = IndicatorType(indicator_value)
    except ValueError:
        return {"error": f"Unknown indicator '{indicator_value}'"}

    series = ctx.get_series(indicator)
    if series is None:
        return {"error": f"No data available for {indicator_value}"}

    if _use_stub():
        from llm.stub import stub_interpret_indicator
        result = stub_interpret_indicator(series)
    else:
        from llm.client import interpret_indicator
        market_ctx = ctx.summary_dict().get("market", {})
        result = interpret_indicator(series, market_ctx)

    return result.model_dump()


class MacroInterpreter:
    """
    Public class interface for macro interpretation.
    Wraps the module-level functions for compatibility with code
    expecting a class-based API.
    """

    @staticmethod
    def interpret(ctx: MacroContext) -> MacroInterpretation:
        """Run the full macro interpretation pipeline."""
        return interpret(ctx)

    @staticmethod
    def stream_briefing(ctx: MacroContext) -> Iterator[str]:
        """Stream a narrative macro briefing as text chunks."""
        return stream_briefing(ctx)

    @staticmethod
    def interpret_single_indicator(
        ctx: MacroContext,
        indicator_value: str,
    ) -> dict:
        """Interpret one indicator by name."""
        return interpret_single_indicator(ctx, indicator_value)
 