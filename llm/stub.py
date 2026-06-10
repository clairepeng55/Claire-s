"""
llm/stub.py
-----------
Stub LLM interpreter that returns realistic, hardcoded responses.

Why a stub instead of just mocking?
  - A mock returns empty or trivially wrong data.
  - This stub returns data that is *structurally correct and economically
    plausible* -- same Pydantic types the live client returns, populated
    with values that reflect a real late-cycle disinflation environment.

This means:
  1. You can build and test the full API + frontend before you have an API key.
  2. You can run CI without burning tokens.
  3. You can demo the product with coherent, believable output.

The stub reads the MacroContext it receives and adjusts its output
slightly based on the actual data (e.g. VIX level, yield spread sign)
so it doesn't feel completely static. It is NOT a real LLM -- it is
a deterministic function that produces plausible output.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Optional

from data.models import (
    ForecastInputs,
    IndicatorInterpretation,
    IndicatorSeries,
    IndicatorType,
    MacroContext,
    MacroInterpretation,
    MacroRegimeAssessment,
    RegimeLabel,
    TrendDirection,
)


# ---------------------------------------------------------------------------
# Per-indicator stub responses
# Each entry covers the narrative fields; numeric fields are filled
# dynamically from the actual series data.
# ---------------------------------------------------------------------------

_INDICATOR_STUBS: dict[IndicatorType, dict] = {

    IndicatorType.CPI: {
        "trend": TrendDirection.DECELERATING,
        "trend_reasoning": (
            "Headline CPI has printed below its prior 12-month average for three "
            "consecutive months, driven primarily by declining goods prices and "
            "energy base effects. Services inflation remains the residual stickiness."
        ),
        "signal": "disinflation in progress, services inflation lagging",
        "implication_for_rates": (
            "With headline CPI decelerating toward 3%, the Fed has scope to begin "
            "cutting, but the pace is constrained by services stickiness. "
            "Markets are pricing roughly two 25bp cuts in the next 12 months."
        ),
        "implication_for_equities": (
            "Falling inflation is broadly constructive for equity multiples as it "
            "reduces the discount rate. The risk is that disinflation is accompanied "
            "by demand softness, compressing earnings."
        ),
        "implication_for_bonds": (
            "Decelerating CPI supports duration. The front end should rally first "
            "as the market prices a more dovish Fed path; the long end will follow "
            "once terminal rate expectations fall."
        ),
    },

    IndicatorType.PCE: {
        "trend": TrendDirection.DECELERATING,
        "trend_reasoning": (
            "Core PCE -- the Fed's preferred inflation gauge -- has trended lower "
            "over the past two quarters but remains above the 2% target. "
            "The 3-month annualised pace is running closer to 2.5%, suggesting "
            "the last mile of disinflation is the hardest."
        ),
        "signal": "approaching Fed target but not yet there",
        "implication_for_rates": (
            "The Fed will want to see core PCE sustain at or below 2.5% annualised "
            "before cutting aggressively. Current readings support a patient, "
            "data-dependent pause rather than a sharp easing cycle."
        ),
        "implication_for_equities": (
            "Core PCE near-target is equity-friendly but not a catalyst by itself. "
            "The equity market is already pricing a soft-landing; the upside from "
            "further disinflation is likely limited unless growth also holds up."
        ),
        "implication_for_bonds": (
            "Core PCE approaching target is the clearest buy signal for intermediate "
            "duration. The 5-year point likely offers the best risk-adjusted carry "
            "in this environment."
        ),
    },

    IndicatorType.UNEMPLOYMENT: {
        "trend": TrendDirection.ACCELERATING,
        "trend_reasoning": (
            "The unemployment rate has ticked up gradually from its cycle low, "
            "consistent with the Sahm Rule threshold approaching but not yet breached. "
            "The rise is driven by labour supply normalisation as much as demand weakness."
        ),
        "signal": "labour market softening from cycle highs",
        "implication_for_rates": (
            "Rising unemployment strengthens the Fed's employment mandate argument "
            "for rate cuts. If unemployment reaches 4.5%+, expect the Fed to shift "
            "from 'patient' to 'proactive' easing language."
        ),
        "implication_for_equities": (
            "A softening labour market is a leading indicator of earnings pressure. "
            "Cyclical sectors (discretionary, industrials) are most exposed; "
            "defensives and quality factor outperform in this environment."
        ),
        "implication_for_bonds": (
            "Weakening employment is the most bond-bullish macro signal. "
            "The Fed will prioritise full employment over inflation once the "
            "unemployment rate rises clearly, accelerating the rate-cutting cycle."
        ),
    },

    IndicatorType.GDP: {
        "trend": TrendDirection.DECELERATING,
        "trend_reasoning": (
            "Real GDP growth has stepped down from its post-reopening peak but "
            "remains positive. The deceleration is consistent with the lagged "
            "effect of 500bp of rate hikes working through credit conditions "
            "and business investment."
        ),
        "signal": "below-trend growth, soft landing scenario intact",
        "implication_for_rates": (
            "Sub-trend growth reduces the risk of re-acceleration in demand-driven "
            "inflation, giving the Fed more flexibility to cut. However, positive "
            "growth means there is no urgency to cut aggressively."
        ),
        "implication_for_equities": (
            "Below-trend growth implies modest earnings growth rather than contraction. "
            "This supports a flat-to-up market with multiple expansion (from lower rates) "
            "offsetting modest earnings revisions downward."
        ),
        "implication_for_bonds": (
            "Slowing growth is supportive of bonds across the curve. "
            "The real yield component should compress as growth expectations fall, "
            "adding to returns from nominal duration."
        ),
    },

    IndicatorType.YIELD_10Y: {
        "trend": TrendDirection.STABLE,
        "trend_reasoning": (
            "The 10-year yield has traded in a relatively narrow range, "
            "reflecting balanced forces: disinflation pulling it lower, "
            "fiscal supply and term premium pushing it higher."
        ),
        "signal": "rate market in equilibrium, watching for catalyst",
        "implication_for_rates": (
            "The 10-year anchoring near current levels suggests the market has "
            "priced a moderate easing cycle but not a recession. "
            "A break above prior highs would signal inflation re-acceleration fears; "
            "a break below would signal growth deterioration."
        ),
        "implication_for_equities": (
            "Stable long rates are a neutral-to-positive backdrop for equities. "
            "The equity risk premium remains compressed at these yield levels, "
            "limiting the multiple expansion that falling rates would otherwise provide."
        ),
        "implication_for_bonds": (
            "Range-bound 10-year yields suggest carry rather than capital gains "
            "is the primary source of fixed income returns in the near term. "
            "Duration positioning should be close to benchmark."
        ),
    },

    IndicatorType.YIELD_2Y: {
        "trend": TrendDirection.DECELERATING,
        "trend_reasoning": (
            "The 2-year yield has begun to fall as the market prices in more Fed cuts, "
            "consistent with disinflation progress and softening labour market data. "
            "The decline has been gradual, reflecting uncertainty about the timing of the first cut."
        ),
        "signal": "market pricing easing cycle, front-end rallying",
        "implication_for_rates": (
            "A falling 2-year yield is a leading indicator of Fed policy easing. "
            "The market is telling you it believes the next move is a cut; "
            "the speed of the decline reflects how quickly that cut is expected."
        ),
        "implication_for_equities": (
            "Front-end rate declines reduce financing costs and support risk assets, "
            "particularly rate-sensitive sectors like utilities, real estate, and "
            "growth equities with long-duration earnings."
        ),
        "implication_for_bonds": (
            "The 2-year is the highest-conviction trade: if the Fed cuts, "
            "you get both carry and capital gains. Risk is a re-acceleration in "
            "inflation that pushes cuts off the table."
        ),
    },

    IndicatorType.VIX: {
        "trend": TrendDirection.STABLE,
        "trend_reasoning": (
            "VIX has been range-bound, consistent with a market that is neither "
            "complacent nor particularly fearful. Realised vol has been low, "
            "supporting the implied vol compression."
        ),
        "signal": "low volatility regime, risk appetite healthy",
        "implication_for_rates": (
            "Low VIX suggests the bond market is not pricing a tail risk scenario. "
            "Flight-to-safety buying is minimal, which is consistent with the "
            "soft-landing narrative."
        ),
        "implication_for_equities": (
            "Low VIX is supportive of carry and short-vol strategies but is a "
            "contrarian warning sign -- complacency can precede sharp vol spikes. "
            "Equity protection is cheap relative to historical norms."
        ),
        "implication_for_bonds": (
            "Low VIX reduces safe-haven demand for Treasuries, which is a mild "
            "headwind to bond prices. A vol spike would reverse this quickly."
        ),
    },

    IndicatorType.DXY: {
        "trend": TrendDirection.STABLE,
        "trend_reasoning": (
            "The dollar index has been range-bound, with dollar strength from "
            "higher US rates offset by improving growth outlooks elsewhere. "
            "The net effect is a relatively stable DXY."
        ),
        "signal": "dollar neutral, no strong directional impulse",
        "implication_for_rates": (
            "A stable dollar reduces imported inflation pressures, which is "
            "marginally supportive of the Fed's ability to cut rates. "
            "A sharp dollar rally would complicate emerging market dynamics."
        ),
        "implication_for_equities": (
            "Dollar stability is neutral for US multinationals. "
            "A weaker dollar would be a tailwind for S&P 500 earnings "
            "translation from overseas operations."
        ),
        "implication_for_bonds": (
            "Dollar stability reduces currency hedging costs for foreign "
            "buyers of US Treasuries, supporting demand at the margin."
        ),
    },
}

# Fallback for any indicator not in the stub map
_FALLBACK_STUB: dict = {
    "trend": TrendDirection.STABLE,
    "trend_reasoning": "Insufficient historical data to identify a clear trend direction.",
    "signal": "data available but trend inconclusive",
    "implication_for_rates":    "Implications for rates policy are unclear from this indicator alone.",
    "implication_for_equities": "Equity implications are uncertain without a clearer trend signal.",
    "implication_for_bonds":    "Bond market implications require corroboration from other indicators.",
}


# ---------------------------------------------------------------------------
# Stub signal strength: derive from actual data so it varies realistically
# ---------------------------------------------------------------------------

def _signal_strength(series: IndicatorSeries) -> float:
    """
    Compute a plausible signal strength from the actual series data.
    Uses trend consistency: if the last 3 moves are in the same direction,
    strength is high. Mixed direction = low strength.
    """
    if len(series.points) < 4:
        return 0.5

    recent = [p.value for p in series.points[-4:]]
    changes = [recent[i+1] - recent[i] for i in range(3)]
    signs = [1 if c > 0 else (-1 if c < 0 else 0) for c in changes]

    if all(s == signs[0] and s != 0 for s in signs):
        return 0.82   # unanimous direction
    elif signs.count(signs[0]) >= 2:
        return 0.60   # majority direction
    else:
        return 0.38   # mixed signals


# ---------------------------------------------------------------------------
# Stub indicator interpreter
# ---------------------------------------------------------------------------

def stub_interpret_indicator(series: IndicatorSeries) -> IndicatorInterpretation:
    """
    Produce a stub IndicatorInterpretation for one series.
    Uses hardcoded narrative + actual values from the series.
    """
    cfg = _INDICATOR_STUBS.get(series.indicator, _FALLBACK_STUB)
    latest = series.latest
    value  = latest.value if latest else 0.0

    return IndicatorInterpretation(
        indicator         = series.indicator,
        name              = series.name,
        current_value     = value,
        unit              = series.unit,
        trend             = cfg["trend"],
        trend_reasoning   = cfg["trend_reasoning"],
        signal            = cfg["signal"],
        signal_strength   = _signal_strength(series),
        signal_reasoning  = (
            f"{series.name} is currently at {value} {series.unit}. "
            f"{cfg['trend_reasoning']}"
        ),
        implication_for_rates    = cfg["implication_for_rates"],
        implication_for_equities = cfg["implication_for_equities"],
        implication_for_bonds    = cfg["implication_for_bonds"],
    )


# ---------------------------------------------------------------------------
# Stub regime assessor: reads actual context to adjust output
# ---------------------------------------------------------------------------

def _derive_regime_from_context(ctx: MacroContext) -> tuple[RegimeLabel, float]:
    """
    Simple rules-based regime detection for the stub.
    Not a real model -- just enough logic to make the stub non-trivially static.
    """
    cpi_s   = ctx.get_series(IndicatorType.CPI)
    unemp_s = ctx.get_series(IndicatorType.UNEMPLOYMENT)
    gdp_s   = ctx.get_series(IndicatorType.GDP)
    vix     = ctx.market.vix or 18.0
    spread  = ctx.market.yield_spread  # 10y - 2y

    cpi_val   = cpi_s.latest.value   if cpi_s   and cpi_s.latest   else 3.2
    unemp_val = unemp_s.latest.value if unemp_s and unemp_s.latest else 4.2
    gdp_val   = gdp_s.latest.value   if gdp_s   and gdp_s.latest   else 2.1

    cpi_trend   = cpi_s.three_month_trend   if cpi_s   else -0.1
    unemp_trend = unemp_s.three_month_trend if unemp_s else 0.05

    # Recession: inverted curve + rising unemployment + low/negative GDP
    if (spread is not None and spread < -0.3
            and gdp_val < 0.5
            and unemp_val > 4.5):
        return RegimeLabel.RECESSION, 0.68

    # Stagflation: high inflation + slowing/weak growth (check before disinflation)
    if cpi_val > 4.0 and gdp_val < 1.5:
        return RegimeLabel.STAGFLATION, 0.62

    # Overheating: strong growth + elevated inflation (check before disinflation)
    if cpi_val > 3.5 and gdp_val > 2.5:
        return RegimeLabel.OVERHEATING, 0.70

    # Disinflation: CPI falling, growth ok
    if cpi_val < 3.5 and (cpi_trend or 0) < 0 and gdp_val > 1.0:
        return RegimeLabel.DISINFLATION, 0.72

    # Early recovery: unemployment falling, growth picking up
    if gdp_val > 2.0 and (unemp_trend or 0) < 0:
        return RegimeLabel.EARLY_RECOVERY, 0.58

    # Default: expansion with moderate confidence
    return RegimeLabel.EXPANSION, 0.55


def stub_regime_assessment(ctx: MacroContext) -> MacroRegimeAssessment:
    """
    Produce a stub MacroRegimeAssessment using the actual MacroContext.
    """
    regime, confidence = _derive_regime_from_context(ctx)

    cpi_val   = 3.2
    unemp_val = 4.2
    gdp_val   = 2.1
    vix       = ctx.market.vix or 18.0
    y10       = ctx.market.yield_10y or 4.3
    y2        = ctx.market.yield_2y  or 4.7
    spread    = ctx.market.yield_spread or (y10 - y2)

    # Adjust forecast inputs to be consistent with detected regime
    recession_prob = {
        RegimeLabel.RECESSION:      0.72,
        RegimeLabel.STAGFLATION:    0.42,
        RegimeLabel.OVERHEATING:    0.22,
        RegimeLabel.DISINFLATION:   0.14,
        RegimeLabel.EARLY_RECOVERY: 0.12,
        RegimeLabel.EXPANSION:      0.13,
        RegimeLabel.UNCERTAINTY:    0.28,
    }.get(regime, 0.18)

    gdp_central = {
        RegimeLabel.RECESSION:      -0.5,
        RegimeLabel.STAGFLATION:     0.8,
        RegimeLabel.OVERHEATING:     3.2,
        RegimeLabel.DISINFLATION:    2.0,
        RegimeLabel.EARLY_RECOVERY:  2.4,
        RegimeLabel.EXPANSION:       2.6,
        RegimeLabel.UNCERTAINTY:     1.5,
    }.get(regime, 2.0)

    regime_narratives = {
        RegimeLabel.DISINFLATION: (
            "The macro environment is best characterised as a disinflation regime. "
            "Headline CPI has decelerated meaningfully from its cycle peak, "
            "core PCE is approaching the Fed's 2% target, and growth -- while "
            "below its post-pandemic trend -- remains solidly positive. "
            "Labour market conditions are softening at the margin but not deteriorating. "
            "The Fed is on hold, data-dependent, and likely to begin a shallow "
            "cutting cycle within the next two quarters."
        ),
        RegimeLabel.EXPANSION: (
            "The economy remains in an expansion regime, with above-trend growth "
            "supported by resilient consumer spending and a still-tight labour market. "
            "Inflation has moderated but has not yet returned to target. "
            "The Fed is unlikely to cut aggressively in this environment."
        ),
        RegimeLabel.OVERHEATING: (
            "The economy is showing clear overheating characteristics: growth is "
            "running above trend and inflation is re-accelerating. "
            "The Fed faces a challenging environment where further tightening "
            "may be warranted, increasing the risk of a policy-induced downturn."
        ),
        RegimeLabel.RECESSION: (
            "Leading indicators are pointing toward a recessionary environment. "
            "The yield curve inversion, rising unemployment, and slowing growth "
            "are consistent with a moderate contraction. "
            "The Fed will likely pivot to aggressive easing."
        ),
        RegimeLabel.STAGFLATION: (
            "A stagflationary dynamic is emerging: inflation remains elevated "
            "while growth is slowing. This is the most challenging environment "
            "for both equities and bonds, and for Fed policy."
        ),
    }

    narrative = regime_narratives.get(
        regime,
        "Mixed macro signals make a definitive regime call difficult. "
        "The dominant dynamic appears to be a gradual transition, "
        "with key data releases in coming weeks likely to clarify the picture."
    )

    return MacroRegimeAssessment(
        regime            = regime,
        regime_confidence = confidence,
        regime_reasoning  = narrative,
        primary_driver    = "CPI trajectory and Fed policy expectations",
        secondary_driver  = "Labour market softening (unemployment trending higher)",
        key_risk          = (
            "Core services inflation re-acceleration -- particularly shelter costs "
            "re-rating upward -- could delay the Fed's easing cycle and reprice "
            "rate-sensitive assets sharply."
        ),
        forecast_inputs = ForecastInputs(
            gdp_growth_central  = gdp_central,
            gdp_growth_bull     = gdp_central + 1.2,
            gdp_growth_bear     = gdp_central - 1.8,
            cpi_yoy_central     = 3.0,
            cpi_yoy_6m_fwd      = 2.6,
            fed_funds_terminal  = 4.0,
            yield_10y_range_lo  = max(y10 - 0.5, 3.0),
            yield_10y_range_hi  = y10 + 0.4,
            equity_risk_premium = 4.2,
            vol_regime          = (
                "low" if vix < 15 else "elevated" if vix > 25 else "normal"
            ),
            recession_prob_12m  = recession_prob,
        ),
        executive_summary = narrative,
        key_takeaways = [
            f"Regime: {regime.value.replace('_', ' ').title()} "
            f"(confidence {confidence:.0%})",
            "CPI decelerating toward target; core services the remaining stickiness",
            "Labour market softening gradually -- Sahm Rule not yet triggered",
            f"Yield curve {'inverted' if spread < 0 else 'normalising'} "
            f"({spread:+.2f}pp spread)",
            f"VIX at {vix:.1f} -- "
            f"{'risk-on' if vix < 15 else 'neutral' if vix < 25 else 'risk-off'} environment",
            f"Recession probability (12m): {recession_prob:.0%}",
        ],
        risks_to_watch = [
            "Core services inflation re-acceleration driven by shelter re-pricing",
            "Labour market deterioration faster than expected (Sahm Rule breach)",
            "Fiscal expansion re-stoking demand-side inflation",
            "Geopolitical shock driving energy price spike",
        ],
        data_releases_to_watch = [
            "Next CPI release (core services and shelter subcomponents)",
            "Non-Farm Payrolls and unemployment rate",
            "FOMC meeting minutes and dot plot revision",
            "Q3 GDP advance estimate",
        ],
    )


# ---------------------------------------------------------------------------
# Main stub entry point
# ---------------------------------------------------------------------------

def run_stub_interpretation(ctx: MacroContext) -> MacroInterpretation:
    """
    Run the full stub interpretation pipeline.
    Returns a MacroInterpretation identical in structure to what the
    live client returns, but generated deterministically without an API call.
    """
    # Interpret each indicator series
    interpretations = [
        stub_interpret_indicator(s)
        for s in ctx.indicators
        if s.latest is not None
    ]

    # Holistic regime assessment
    regime = stub_regime_assessment(ctx)

    return MacroInterpretation(
        run_id                     = str(uuid.uuid4()),
        as_of                      = datetime.utcnow(),
        context_summary            = ctx.summary_dict(),
        indicator_interpretations  = interpretations,
        regime_assessment          = regime,
        is_stub                    = True,
        model_used                 = "stub (no API key configured)",
        latency_ms                 = 0,
    )