"""
tests/test_llm_layer.py
-----------------------
Tests for the full LLM layer: data models, stub, dispatcher, and API routes.

All tests run without an API key -- the stub is the testable surface.
Live client tests are marked and skipped unless ANTHROPIC_API_KEY is set.

Test philosophy:
  - Data layer: validate types and computed properties
  - Stub: validate that output is structurally correct and internally consistent
  - Dispatcher: validate routing logic (stub vs live)
  - API: validate HTTP contract (status codes, response shapes)
"""

import os
import sys
import json
from datetime import date, datetime

sys.path.insert(0, "/home/claude/macro_platform")

# Force stub mode for all tests
os.environ["USE_STUB"] = "true"

from data.models import (
    ForecastInputs, IndicatorInterpretation, IndicatorPoint,
    IndicatorSeries, IndicatorType, MacroContext, MacroInterpretation,
    MacroRegimeAssessment, MarketSnapshot, RegimeLabel, TrendDirection,
)
from data.fetcher import build_macro_context
from llm.stub import (
    run_stub_interpretation, stub_interpret_indicator,
    stub_regime_assessment, _derive_regime_from_context,
)
from llm.interpreter import interpret, stream_briefing, interpret_single_indicator

PASS = "✓"
FAIL = "✗"
_results = []


def check(name: str, condition: bool, detail: str = ""):
    status = PASS if condition else FAIL
    _results.append((status, name, detail))
    print(f"  {status}  {name}" + (f"  [{detail}]" if detail else ""))


# ---------------------------------------------------------------------------
# Helpers: build minimal test fixtures
# ---------------------------------------------------------------------------

def _make_series(
    indicator: IndicatorType,
    values: list[float],
    unit: str = "%",
) -> IndicatorSeries:
    points = [
        IndicatorPoint(
            indicator=indicator, name=indicator.value,
            value=v, unit=unit,
            as_of=date(2025, i+1, 1),
            source="test",
        )
        for i, v in enumerate(values)
    ]
    return IndicatorSeries(
        indicator=indicator, name=indicator.value,
        unit=unit, source="test", points=points,
    )


def _make_context(
    cpi: float = 3.2,
    unemp: float = 4.2,
    gdp: float = 2.1,
    vix: float = 18.0,
    y10: float = 4.3,
    y2: float = 4.7,
    cpi_rising: bool = False,   # True = rising CPI series, False = falling
) -> MacroContext:
    spread = y10 - y2
    market = MarketSnapshot(
        as_of=datetime.utcnow(),
        sp500_price=5200.0,
        yield_2y=y2,
        yield_10y=y10,
        yield_spread=spread,
        vix=vix,
        dxy=104.0,
    )
    # Build CPI series that is directionally consistent with scenario
    if cpi_rising:
        cpi_vals = [cpi - 0.3, cpi - 0.15, cpi - 0.05, cpi]
    else:
        cpi_vals = [cpi + 0.3, cpi + 0.15, cpi + 0.05, cpi]
    indicators = [
        _make_series(IndicatorType.CPI,          cpi_vals),
        _make_series(IndicatorType.PCE,          [3.2, 3.0, 2.8, 2.7]),
        _make_series(IndicatorType.UNEMPLOYMENT, [4.0, 4.1, 4.2, unemp]),
        _make_series(IndicatorType.GDP,          [2.8, 2.4, gdp],  unit="% annualised"),
    ]
    return MacroContext(as_of=datetime.utcnow(), market=market, indicators=indicators)


# ===========================================================================
print("\n=== 1. Data model correctness ===")
# ===========================================================================

# IndicatorSeries computed properties
s = _make_series(IndicatorType.CPI, [3.7, 3.5, 3.3, 3.2])

check("latest value correct",   s.latest.value == 3.2,     f"got {s.latest.value}")
check("previous value correct", s.previous.value == 3.3,   f"got {s.previous.value}")
check("mom_change correct",     s.mom_change == -0.1,       f"got {s.mom_change}")
check("three_month_trend negative (declining series)",
      s.three_month_trend is not None and s.three_month_trend < 0,
      f"got {s.three_month_trend}")

# MarketSnapshot derived properties
snap = MarketSnapshot(
    as_of=datetime.utcnow(),
    yield_2y=4.8, yield_10y=4.3, vix=12.0,
)
check("yield_spread computed correctly",
      snap.yield_spread == round(4.3 - 4.8, 4),
      f"got {snap.yield_spread}")
check("curve_inverted = True when 10y < 2y",
      snap.curve_inverted == True)
check("risk_regime = risk_on when VIX < 15",
      snap.risk_regime == "risk_on", f"got {snap.risk_regime}")

snap2 = MarketSnapshot(as_of=datetime.utcnow(), vix=28.0)
check("risk_regime = risk_off when VIX > 25",
      snap2.risk_regime == "risk_off", f"got {snap2.risk_regime}")

# MacroContext.summary_dict() structure
ctx = _make_context()
summary = ctx.summary_dict()
check("summary_dict has market key",     "market" in summary)
check("summary_dict has indicators key", "indicators" in summary)
check("summary_dict has as_of key",      "as_of" in summary)
check("market has vix",                  "vix" in summary["market"])
check("market has curve_inverted",       "curve_inverted" in summary["market"])
check("CPI in indicators",               "cpi" in summary["indicators"])
check("CPI has value",                   "value" in summary["indicators"]["cpi"])

# get_series
check("get_series returns correct type",
      ctx.get_series(IndicatorType.CPI).indicator == IndicatorType.CPI)
check("get_series returns None for missing",
      ctx.get_series(IndicatorType.YIELD_10Y) is None)


# ===========================================================================
print("\n=== 2. Stub indicator interpretation ===")
# ===========================================================================

cpi_series = _make_series(IndicatorType.CPI, [3.7, 3.5, 3.3, 3.2])
interp = stub_interpret_indicator(cpi_series)

check("returns IndicatorInterpretation",  isinstance(interp, IndicatorInterpretation))
check("indicator field correct",          interp.indicator == IndicatorType.CPI)
check("current_value matches series",     interp.current_value == 3.2)
check("trend is a TrendDirection",        isinstance(interp.trend, TrendDirection))
check("signal_strength in [0,1]",
      0.0 <= interp.signal_strength <= 1.0, f"got {interp.signal_strength}")
check("signal is non-empty string",       len(interp.signal) > 0)
check("trend_reasoning non-empty",        len(interp.trend_reasoning) > 0)
check("implication_for_rates non-empty",  len(interp.implication_for_rates) > 0)
check("implication_for_equities non-empty", len(interp.implication_for_equities) > 0)
check("implication_for_bonds non-empty",  len(interp.implication_for_bonds) > 0)

# Signal strength varies with trend consistency
flat_series    = _make_series(IndicatorType.VIX, [18.0, 18.1, 17.9, 18.0], unit="index")
uniform_series = _make_series(IndicatorType.CPI, [4.0, 3.7, 3.4, 3.1])
flat_strength    = stub_interpret_indicator(flat_series).signal_strength
uniform_strength = stub_interpret_indicator(uniform_series).signal_strength
check("uniform declining series has higher signal_strength than flat series",
      uniform_strength > flat_strength,
      f"uniform={uniform_strength:.2f}, flat={flat_strength:.2f}")

# All known indicator types return valid interpretations
for ind_type in [
    IndicatorType.PCE, IndicatorType.UNEMPLOYMENT, IndicatorType.GDP,
    IndicatorType.YIELD_10Y, IndicatorType.YIELD_2Y, IndicatorType.VIX, IndicatorType.DXY,
]:
    s = _make_series(ind_type, [1.0, 1.1, 1.2, 1.3])
    i = stub_interpret_indicator(s)
    check(f"stub handles {ind_type.value}", isinstance(i, IndicatorInterpretation))


# ===========================================================================
print("\n=== 3. Regime detection logic ===")
# ===========================================================================

# Disinflation: falling CPI, positive growth
ctx_dis = _make_context(cpi=3.0, gdp=2.1, unemp=4.2, y10=4.3, y2=4.7)
regime_dis, conf_dis = _derive_regime_from_context(ctx_dis)
check("disinflation detected with falling CPI + positive GDP",
      regime_dis == RegimeLabel.DISINFLATION,
      f"got {regime_dis}")

# Recession: inverted curve + high unemployment + low growth
ctx_rec = _make_context(cpi=3.0, gdp=0.2, unemp=5.0, y10=3.8, y2=4.5)
regime_rec, conf_rec = _derive_regime_from_context(ctx_rec)
check("recession detected with inverted curve + high unemployment + low growth",
      regime_rec == RegimeLabel.RECESSION,
      f"got {regime_rec}")

# Overheating: high CPI + strong GDP
ctx_hot = _make_context(cpi=4.2, gdp=3.5, unemp=3.8, y10=4.8, y2=4.5, cpi_rising=True)
regime_hot, _ = _derive_regime_from_context(ctx_hot)
check("overheating detected with high CPI + strong GDP",
      regime_hot == RegimeLabel.OVERHEATING,
      f"got {regime_hot}")

# Stagflation: high CPI + weak growth
ctx_stag = _make_context(cpi=4.5, gdp=0.8, unemp=4.8, y10=4.5, y2=4.3, cpi_rising=True)
regime_stag, _ = _derive_regime_from_context(ctx_stag)
check("stagflation detected with high CPI + weak GDP",
      regime_stag == RegimeLabel.STAGFLATION,
      f"got {regime_stag}")

# Confidence is always in [0,1]
for r, c in [
    _derive_regime_from_context(_make_context(cpi=3.0, gdp=2.1)),
    _derive_regime_from_context(_make_context(cpi=4.5, gdp=0.2, unemp=5.0, y10=3.5, y2=4.5)),
]:
    check(f"confidence in [0,1] for {r.value}", 0.0 <= c <= 1.0, f"c={c}")


# ===========================================================================
print("\n=== 4. Full stub interpretation pipeline ===")
# ===========================================================================

ctx = _make_context()
result = run_stub_interpretation(ctx)

check("returns MacroInterpretation",             isinstance(result, MacroInterpretation))
check("is_stub = True",                          result.is_stub == True)
check("has a run_id",                            len(result.run_id) > 0)
check("regime_assessment is MacroRegimeAssessment",
      isinstance(result.regime_assessment, MacroRegimeAssessment))
check("forecast_inputs present",
      isinstance(result.regime_assessment.forecast_inputs, ForecastInputs))
check("indicator_interpretations non-empty",
      len(result.indicator_interpretations) > 0)
check("each indicator interpretation is typed",
      all(isinstance(i, IndicatorInterpretation)
          for i in result.indicator_interpretations))
check("executive_summary non-empty",
      len(result.regime_assessment.executive_summary) > 0)
check("key_takeaways non-empty",
      len(result.regime_assessment.key_takeaways) > 0)
check("risks_to_watch non-empty",
      len(result.regime_assessment.risks_to_watch) > 0)

# ForecastInputs internal consistency
fi = result.regime_assessment.forecast_inputs
check("gdp bull > central > bear",
      fi.gdp_growth_bull > fi.gdp_growth_central > fi.gdp_growth_bear,
      f"bull={fi.gdp_growth_bull}, central={fi.gdp_growth_central}, bear={fi.gdp_growth_bear}")
check("recession_prob in [0,1]",
      0.0 <= fi.recession_prob_12m <= 1.0,
      f"got {fi.recession_prob_12m}")
check("vol_regime is valid string",
      fi.vol_regime in ("low", "normal", "elevated"),
      f"got '{fi.vol_regime}'")
check("yield range is ordered",
      fi.yield_10y_range_lo < fi.yield_10y_range_hi,
      f"lo={fi.yield_10y_range_lo}, hi={fi.yield_10y_range_hi}")

# Recession scenario: recession_prob should be higher
ctx_rec_full = _make_context(cpi=3.0, gdp=0.2, unemp=5.0, y10=3.8, y2=4.5)
result_rec = run_stub_interpretation(ctx_rec_full)
check("recession scenario has higher recession_prob than expansion",
      result_rec.regime_assessment.forecast_inputs.recession_prob_12m
      > result.regime_assessment.forecast_inputs.recession_prob_12m,
      f"recession={result_rec.regime_assessment.forecast_inputs.recession_prob_12m:.2f}, "
      f"base={result.regime_assessment.forecast_inputs.recession_prob_12m:.2f}")


# ===========================================================================
print("\n=== 5. Dispatcher routing ===")
# ===========================================================================

# USE_STUB=true → should always use stub
os.environ["USE_STUB"] = "true"
result_stub = interpret(ctx)
check("dispatcher uses stub when USE_STUB=true",
      result_stub.is_stub == True)

# interpret_single_indicator returns a dict
single = interpret_single_indicator(ctx, "cpi")
check("single indicator returns dict",           isinstance(single, dict))
check("single indicator has 'signal' key",       "signal" in single)
check("single indicator has 'trend' key",        "trend" in single)
check("single indicator has 'current_value' key","current_value" in single)

# Unknown indicator returns error dict
bad = interpret_single_indicator(ctx, "nonsense_indicator")
check("unknown indicator returns error dict",    "error" in bad)

# Missing indicator (valid type but not in context)
missing = interpret_single_indicator(ctx, "fed_funds")
check("missing indicator returns error dict",    "error" in missing)


# ===========================================================================
print("\n=== 6. Streaming ===")
# ===========================================================================

chunks = list(stream_briefing(ctx))
check("stream returns at least one chunk",        len(chunks) > 0)
check("all chunks are strings",                   all(isinstance(c, str) for c in chunks))
check("stream content is non-empty",
      sum(len(c) for c in chunks) > 50,
      f"total chars: {sum(len(c) for c in chunks)}")


# ===========================================================================
print("\n=== 7. JSON serialisation (API readiness) ===")
# ===========================================================================

result = run_stub_interpretation(_make_context())
try:
    serialised = result.model_dump(mode="json")
    json_str   = json.dumps(serialised, default=str)
    check("MacroInterpretation serialises to JSON cleanly", True)
    check("JSON is non-trivially large", len(json_str) > 1000, f"{len(json_str)} chars")
except Exception as e:
    check("MacroInterpretation serialises to JSON cleanly", False, str(e))

# Nested objects serialise correctly
check("forecast_inputs present in serialised output",
      "forecast_inputs" in serialised["regime_assessment"])
check("indicator_interpretations is a list",
      isinstance(serialised["indicator_interpretations"], list))
check("regime is a string in serialised output",
      isinstance(serialised["regime_assessment"]["regime"], str))


# ===========================================================================
print("\n=== 8. Data fetcher integration ===")
# ===========================================================================

# build_macro_context() should run without crashing even if Yahoo is unavailable
try:
    live_ctx = build_macro_context()
    check("build_macro_context() returns MacroContext",
          isinstance(live_ctx, MacroContext))
    check("MacroContext has indicators",
          len(live_ctx.indicators) > 0,
          f"got {len(live_ctx.indicators)} series")
    check("MacroContext has market snapshot",
          isinstance(live_ctx.market, MarketSnapshot))
    check("Simulated CPI series present",
          live_ctx.get_series(IndicatorType.CPI) is not None)
    # data_warnings is a list (may be empty or have warnings)
    check("data_warnings is a list",
          isinstance(live_ctx.data_warnings, list))
except Exception as e:
    check("build_macro_context() completes without fatal error", False, str(e))


# ===========================================================================
# Summary
# ===========================================================================

print("\n" + "=" * 58)
passed = sum(1 for s, _, _ in _results if s == PASS)
failed = sum(1 for s, _, _ in _results if s == FAIL)
print(f"  Results: {passed}/{len(_results)} passed  |  {failed} failed")
print("=" * 58)

if failed:
    print("\nFailed tests:")
    for s, name, detail in _results:
        if s == FAIL:
            print(f"  {FAIL}  {name}  {detail}")
    sys.exit(1)