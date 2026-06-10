"""
data/models.py
--------------
Pydantic contracts for every piece of data that flows through the platform.

Design rule: nothing raw ever leaves the data layer. Yahoo Finance returns
messy dicts and DataFrames with inconsistent dtypes -- these models are the
sanitisation checkpoint. Anything that reaches the LLM layer or the API is
typed, validated, and documented.
"""

from __future__ import annotations

from datetime import date, datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class IndicatorType(str, Enum):
    """The macro indicators this platform understands."""
    CPI          = "cpi"           # Consumer Price Index
    PCE          = "pce"           # Personal Consumption Expenditures
    UNEMPLOYMENT = "unemployment"  # Unemployment rate
    GDP          = "gdp"           # GDP growth
    FED_FUNDS    = "fed_funds"     # Federal Funds Rate
    YIELD_2Y     = "yield_2y"      # 2-year Treasury yield
    YIELD_10Y    = "yield_10y"     # 10-year Treasury yield
    YIELD_SPREAD = "yield_spread"  # 10y - 2y spread (recession indicator)
    VIX          = "vix"           # Volatility index
    DXY          = "dxy"           # Dollar index
    CUSTOM       = "custom"        # User-defined


class RegimeLabel(str, Enum):
    """
    Macro regime labels the LLM assigns based on indicator patterns.
    These map to distinct forecasting and risk model configurations.
    """
    EXPANSION          = "expansion"          # growth above trend, low unemployment
    OVERHEATING        = "overheating"        # growth strong, inflation rising
    STAGFLATION        = "stagflation"        # high inflation + slowing growth
    RECESSION          = "recession"          # contracting output
    EARLY_RECOVERY     = "early_recovery"     # growth turning up from trough
    DISINFLATION       = "disinflation"       # inflation falling, growth ok
    UNCERTAINTY        = "uncertainty"        # mixed signals, no clear regime


class TrendDirection(str, Enum):
    ACCELERATING = "accelerating"
    STABLE       = "stable"
    DECELERATING = "decelerating"
    REVERSING    = "reversing"


# ---------------------------------------------------------------------------
# A single indicator observation
# ---------------------------------------------------------------------------

class IndicatorPoint(BaseModel):
    """One time-stamped reading of a macro indicator."""
    indicator:   IndicatorType
    name:        str             # human-readable: "US CPI YoY"
    value:       float
    unit:        str             # "% YoY", "% annualised", "index level", etc.
    as_of:       date
    source:      str             # "Yahoo Finance", "FRED", "BLS", etc.
    ticker_used: Optional[str] = None   # underlying Yahoo ticker if applicable


class IndicatorSeries(BaseModel):
    """A time series of readings for one indicator."""
    indicator:   IndicatorType
    name:        str
    unit:        str
    source:      str
    points:      list[IndicatorPoint]   # sorted ascending by as_of

    @property
    def latest(self) -> Optional[IndicatorPoint]:
        return self.points[-1] if self.points else None

    @property
    def previous(self) -> Optional[IndicatorPoint]:
        return self.points[-2] if len(self.points) >= 2 else None

    @property
    def mom_change(self) -> Optional[float]:
        """Most recent period-over-period change."""
        if self.latest and self.previous:
            return round(self.latest.value - self.previous.value, 4)
        return None

    @property
    def three_month_trend(self) -> Optional[float]:
        """Average change over last 3 periods."""
        if len(self.points) >= 4:
            recent = [p.value for p in self.points[-4:]]
            changes = [recent[i+1] - recent[i] for i in range(3)]
            return round(sum(changes) / 3, 4)
        return None


# ---------------------------------------------------------------------------
# Market data snapshot (from Yahoo Finance)
# ---------------------------------------------------------------------------

class MarketSnapshot(BaseModel):
    """
    Current market prices and derived indicators fetched from Yahoo Finance.
    This is the 'market context' fed to the LLM alongside macro indicators.
    """
    as_of:             datetime

    # Equity market
    sp500_price:       Optional[float] = None
    sp500_1d_chg_pct:  Optional[float] = None
    sp500_52w_high:    Optional[float] = None
    sp500_52w_low:     Optional[float] = None

    # Rates
    yield_2y:          Optional[float] = None   # % e.g. 4.85
    yield_10y:         Optional[float] = None
    yield_spread:      Optional[float] = None   # 10y - 2y

    # Vol / risk sentiment
    vix:               Optional[float] = None

    # Dollar
    dxy:               Optional[float] = None
    dxy_1d_chg_pct:    Optional[float] = None

    # Commodities (inflation proxies)
    crude_oil_price:   Optional[float] = None
    gold_price:        Optional[float] = None

    from pydantic import model_validator

    @model_validator(mode="after")
    def compute_spread(self) -> "MarketSnapshot":
        if self.yield_spread is None and self.yield_2y is not None and self.yield_10y is not None:
            object.__setattr__(self, "yield_spread", round(self.yield_10y - self.yield_2y, 4))
        return self

    @property
    def curve_inverted(self) -> Optional[bool]:
        if self.yield_spread is not None:
            return self.yield_spread < 0
        return None

    @property
    def risk_regime(self) -> str:
        """Rough risk-on / risk-off signal from VIX."""
        if self.vix is None:
            return "unknown"
        if self.vix < 15:
            return "risk_on"
        if self.vix < 25:
            return "neutral"
        if self.vix < 35:
            return "risk_off"
        return "stress"


# ---------------------------------------------------------------------------
# The full macro context bundle passed to the LLM
# ---------------------------------------------------------------------------

class MacroContext(BaseModel):
    """
    Everything the LLM needs to interpret the current macro environment.
    Built by the data layer, consumed by the LLM layer.
    """
    as_of:          datetime
    market:         MarketSnapshot
    indicators:     list[IndicatorSeries]   # all fetched series
    data_warnings:  list[str] = Field(default_factory=list)  # stale data, missing tickers, etc.

    def get_series(self, indicator: IndicatorType) -> Optional[IndicatorSeries]:
        for s in self.indicators:
            if s.indicator == indicator:
                return s
        return None

    def summary_dict(self) -> dict:
        """
        Compact dict representation for injecting into LLM prompts.
        Avoids dumping the full Pydantic model (too verbose).
        """
        out = {
            "as_of": self.as_of.isoformat(),
            "market": {
                "sp500":         self.market.sp500_price,
                "sp500_1d_chg":  self.market.sp500_1d_chg_pct,
                "yield_2y":      self.market.yield_2y,
                "yield_10y":     self.market.yield_10y,
                "yield_spread":  self.market.yield_spread,
                "curve_inverted":self.market.curve_inverted,
                "vix":           self.market.vix,
                "risk_regime":   self.market.risk_regime,
                "dxy":           self.market.dxy,
                "crude_oil":     self.market.crude_oil_price,
                "gold":          self.market.gold_price,
            },
            "indicators": {}
        }
        for series in self.indicators:
            if series.latest:
                out["indicators"][series.indicator.value] = {
                    "name":       series.name,
                    "value":      series.latest.value,
                    "unit":       series.unit,
                    "as_of":      series.latest.as_of.isoformat(),
                    "mom_change": series.mom_change,
                    "trend_3m":   series.three_month_trend,
                }
        if self.data_warnings:
            out["warnings"] = self.data_warnings
        return out


# ---------------------------------------------------------------------------
# LLM output models
# ---------------------------------------------------------------------------

class IndicatorInterpretation(BaseModel):
    """
    LLM's structured interpretation of a single indicator.
    One of these is produced per indicator in the macro context.
    """
    indicator:    IndicatorType
    name:         str
    current_value: float
    unit:         str

    # What the number means
    trend:        TrendDirection
    trend_reasoning: str    # 1-2 sentences explaining why this trend label was chosen

    # Signal interpretation
    signal:       str       # e.g. "inflationary pressure", "labour market cooling"
    signal_strength: float  # 0-1, how strong/clear the signal is
    signal_reasoning: str   # 2-3 sentences

    # Implications
    implication_for_rates:   str   # what this implies for Fed policy
    implication_for_equities: str  # what this implies for equity markets
    implication_for_bonds:   str   # what this implies for fixed income


class MacroRegimeAssessment(BaseModel):
    """
    LLM's overall regime call, drawing on all indicators together.
    This is the primary structured output consumed by downstream
    forecasting and risk models.
    """
    regime:             RegimeLabel
    regime_confidence:  float = Field(..., ge=0, le=1)
    regime_reasoning:   str   # paragraph explaining the call

    # Dominant forces
    primary_driver:     str   # the one indicator most driving the call
    secondary_driver:   Optional[str] = None
    key_risk:           str   # the single biggest thing that could change the regime

    # Forecasting model inputs
    # These are the structured parameters the downstream model will consume.
    # The LLM generates them from its regime assessment.
    forecast_inputs: "ForecastInputs"

    # Narrative
    executive_summary:  str   # 3-5 sentence plain-English summary
    key_takeaways:      list[str]   # 4-6 bullets for the dashboard
    risks_to_watch:     list[str]   # 3-5 items
    data_releases_to_watch: list[str]  # upcoming catalysts


class ForecastInputs(BaseModel):
    """
    Structured parameters generated by the LLM for use in downstream
    quantitative forecasting and risk models.

    These are not market predictions -- they are scenario parameters
    that describe the central case implied by the current regime.
    The quant model uses them as inputs, not as outputs.
    """
    # Growth
    gdp_growth_central:  float  # annualised %, e.g. 2.1
    gdp_growth_bull:     float
    gdp_growth_bear:     float

    # Inflation
    cpi_yoy_central:     float  # %, e.g. 3.2
    cpi_yoy_6m_fwd:      float  # where LLM thinks it's heading

    # Rates
    fed_funds_terminal:  float  # expected peak/trough rate
    yield_10y_range_lo:  float
    yield_10y_range_hi:  float

    # Risk parameters
    equity_risk_premium: float  # % above risk-free rate
    vol_regime:          str    # "low" (<15 VIX), "normal" (15-25), "elevated" (>25)
    recession_prob_12m:  float  # 0-1


# Resolve forward reference
MacroRegimeAssessment.model_rebuild()


# ---------------------------------------------------------------------------
# Full interpretation bundle
# ---------------------------------------------------------------------------

class MacroInterpretation(BaseModel):
    """
    Complete output of one LLM interpretation run.
    Stored, returned to the API, and displayed on the frontend.
    """
    run_id:               str
    as_of:                datetime
    context_summary:      dict              # the MacroContext.summary_dict() that was used
    indicator_interpretations: list[IndicatorInterpretation]
    regime_assessment:    MacroRegimeAssessment
    is_stub:              bool = False      # True when generated by the stub (no API key)
    model_used:           str = "claude-sonnet-4-20250514"
    latency_ms:           Optional[int] = None