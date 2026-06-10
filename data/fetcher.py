"""
data/fetcher.py
---------------
Fetches market data from Yahoo Finance and maps it onto our macro indicators.

Yahoo Finance doesn't publish CPI or PCE directly -- those come from BLS/BEA.
What we CAN get from Yahoo Finance as inflation/macro proxies:

  Ticker   | What it is               | Macro proxy for
  ---------|--------------------------|----------------------------------
  ^TNX     | 10-year Treasury yield   | Real rates / Fed expectations
  ^FVX     | 5-year Treasury yield    | Medium-term rate expectations
  ^IRX     | 13-week T-bill yield     | Proxy for current Fed Funds rate
  ^VIX     | CBOE VIX                 | Market stress / uncertainty
  ^GSPC    | S&P 500                  | Growth / risk sentiment
  DX-Y.NYB | US Dollar Index          | Dollar strength
  GC=F     | Gold futures             | Inflation hedge / safe haven
  CL=F     | Crude oil futures        | Energy inflation input
  TIP      | iShares TIPS ETF         | Breakeven inflation expectations
  RINF     | ProShares Inflation ETF  | Inflation expectations proxy
  ^SP500TR | S&P 500 Total Return     | Equity performance

For a real production system you'd layer in FRED (free API) for actual
CPI/PCE/unemployment data. This fetcher is designed so you can add that
without changing the interface -- just add more IndicatorSeries to the result.
"""

from __future__ import annotations

import warnings
from datetime import date, datetime, timedelta
from typing import Optional

import pandas as pd
import yfinance as yf

from .models import (
    IndicatorPoint, IndicatorSeries, IndicatorType,
    MacroContext, MarketSnapshot,
)

# Suppress yfinance noise
warnings.filterwarnings("ignore", category=FutureWarning)


# ---------------------------------------------------------------------------
# Ticker map: IndicatorType → Yahoo Finance ticker
# ---------------------------------------------------------------------------

INDICATOR_TICKERS: dict[IndicatorType, dict] = {
    IndicatorType.YIELD_10Y: {
        "ticker": "^TNX",
        "name":   "US 10-Year Treasury Yield",
        "unit":   "%",
    },
    IndicatorType.YIELD_2Y: {
        "ticker": "^TYX",   # using 30y as proxy; swap for actual 2y if available
        "name":   "US 2-Year Treasury Yield (proxy: ^FVX)",
        "unit":   "%",
    },
    IndicatorType.VIX: {
        "ticker": "^VIX",
        "name":   "CBOE Volatility Index (VIX)",
        "unit":   "index",
    },
    IndicatorType.DXY: {
        "ticker": "DX-Y.NYB",
        "name":   "US Dollar Index (DXY)",
        "unit":   "index",
    },
}

# Separate tickers for the market snapshot (equities, commodities)
SNAPSHOT_TICKERS = {
    "sp500":      "^GSPC",
    "yield_2y":   "^FVX",    # 5y used as 2y proxy (Yahoo doesn't carry ^IRX well)
    "yield_10y":  "^TNX",
    "vix":        "^VIX",
    "dxy":        "DX-Y.NYB",
    "crude":      "CL=F",
    "gold":       "GC=F",
    "tips":       "TIP",     # TIPS ETF as breakeven inflation proxy
}

# Simulated macro indicators (no direct Yahoo ticker; realistic placeholder values)
# In production: replace with FRED API calls
SIMULATED_INDICATORS: list[dict] = [
    {
        "indicator": IndicatorType.CPI,
        "name":      "US CPI YoY",
        "unit":      "% YoY",
        "source":    "BLS (simulated — wire FRED for live data)",
        # Realistic recent history (monthly, most-recent-last)
        "values": [3.7, 3.7, 3.2, 3.1, 3.4, 3.5, 3.5, 3.2, 3.0],
        "dates":  [
            date(2024, 9, 1), date(2024, 10, 1), date(2024, 11, 1),
            date(2024, 12, 1), date(2025, 1, 1),  date(2025, 2, 1),
            date(2025, 3, 1),  date(2025, 4, 1),  date(2025, 5, 1),
        ],
    },
    {
        "indicator": IndicatorType.PCE,
        "name":      "US PCE YoY (Core)",
        "unit":      "% YoY",
        "source":    "BEA (simulated — wire FRED for live data)",
        "values": [3.4, 3.2, 3.2, 2.9, 2.8, 2.8, 2.7, 2.6, 2.6],
        "dates":  [
            date(2024, 9, 1), date(2024, 10, 1), date(2024, 11, 1),
            date(2024, 12, 1), date(2025, 1, 1),  date(2025, 2, 1),
            date(2025, 3, 1),  date(2025, 4, 1),  date(2025, 5, 1),
        ],
    },
    {
        "indicator": IndicatorType.UNEMPLOYMENT,
        "name":      "US Unemployment Rate",
        "unit":      "%",
        "source":    "BLS (simulated — wire FRED for live data)",
        "values": [3.9, 4.1, 4.2, 4.1, 4.0, 4.1, 4.2, 4.2, 4.3],
        "dates":  [
            date(2024, 9, 1), date(2024, 10, 1), date(2024, 11, 1),
            date(2024, 12, 1), date(2025, 1, 1),  date(2025, 2, 1),
            date(2025, 3, 1),  date(2025, 4, 1),  date(2025, 5, 1),
        ],
    },
    {
        "indicator": IndicatorType.GDP,
        "name":      "US Real GDP Growth (Annualised)",
        "unit":      "% QoQ annualised",
        "source":    "BEA (simulated — wire FRED for live data)",
        "values": [3.1, 2.4, 2.8, 3.1],
        "dates":  [
            date(2024, 6, 30), date(2024, 9, 30),
            date(2024, 12, 31), date(2025, 3, 31),
        ],
    },
]


# ---------------------------------------------------------------------------
# Snapshot fetcher
# ---------------------------------------------------------------------------

def fetch_market_snapshot() -> tuple[MarketSnapshot, list[str]]:
    """
    Fetch current market prices from Yahoo Finance.
    Returns (snapshot, warnings) where warnings lists any tickers that failed.
    """
    warnings_out: list[str] = []
    data: dict[str, Optional[float]] = {}

    all_tickers = list(SNAPSHOT_TICKERS.values())

    try:
        raw = yf.download(
            tickers=all_tickers,
            period="5d",
            interval="1d",
            progress=False,
            auto_adjust=True,
        )
    except Exception as e:
        warnings_out.append(f"Yahoo Finance snapshot fetch failed: {e}")
        raw = pd.DataFrame()

    def _latest(ticker: str) -> Optional[float]:
        try:
            if raw.empty:
                return None
            if isinstance(raw.columns, pd.MultiIndex):
                col = ("Close", ticker)
                if col in raw.columns:
                    s = raw[col].dropna()
                    return round(float(s.iloc[-1]), 4) if len(s) else None
            else:
                if "Close" in raw.columns:
                    s = raw["Close"].dropna()
                    return round(float(s.iloc[-1]), 4) if len(s) else None
        except Exception:
            return None

    def _1d_chg(ticker: str) -> Optional[float]:
        try:
            if raw.empty:
                return None
            if isinstance(raw.columns, pd.MultiIndex):
                col = ("Close", ticker)
                if col in raw.columns:
                    s = raw[col].dropna()
                    if len(s) >= 2:
                        return round((s.iloc[-1] / s.iloc[-2] - 1) * 100, 3)
        except Exception:
            return None

    sp500    = _latest(SNAPSHOT_TICKERS["sp500"])
    sp500_1d = _1d_chg(SNAPSHOT_TICKERS["sp500"])
    y2       = _latest(SNAPSHOT_TICKERS["yield_2y"])
    y10      = _latest(SNAPSHOT_TICKERS["yield_10y"])
    vix      = _latest(SNAPSHOT_TICKERS["vix"])
    dxy      = _latest(SNAPSHOT_TICKERS["dxy"])
    dxy_1d   = _1d_chg(SNAPSHOT_TICKERS["dxy"])
    crude    = _latest(SNAPSHOT_TICKERS["crude"])
    gold     = _latest(SNAPSHOT_TICKERS["gold"])

    spread = None
    if y10 is not None and y2 is not None:
        spread = round(y10 - y2, 4)

    # Note any missing tickers
    for key, ticker in SNAPSHOT_TICKERS.items():
        if _latest(ticker) is None:
            warnings_out.append(f"Could not fetch {key} ({ticker}) from Yahoo Finance")

    return MarketSnapshot(
        as_of            = datetime.utcnow(),
        sp500_price      = sp500,
        sp500_1d_chg_pct = sp500_1d,
        yield_2y         = y2,
        yield_10y        = y10,
        yield_spread     = spread,
        vix              = vix,
        dxy              = dxy,
        dxy_1d_chg_pct   = dxy_1d,
        crude_oil_price  = crude,
        gold_price       = gold,
    ), warnings_out


# ---------------------------------------------------------------------------
# Indicator series fetcher
# ---------------------------------------------------------------------------

def _build_simulated_series() -> list[IndicatorSeries]:
    """Build IndicatorSeries from the hardcoded simulation data."""
    series_list = []
    for cfg in SIMULATED_INDICATORS:
        points = [
            IndicatorPoint(
                indicator   = cfg["indicator"],
                name        = cfg["name"],
                value       = v,
                unit        = cfg["unit"],
                as_of       = d,
                source      = cfg["source"],
                ticker_used = None,
            )
            for v, d in zip(cfg["values"], cfg["dates"])
        ]
        series_list.append(IndicatorSeries(
            indicator = cfg["indicator"],
            name      = cfg["name"],
            unit      = cfg["unit"],
            source    = cfg["source"],
            points    = points,
        ))
    return series_list


def fetch_yield_series(lookback_days: int = 365) -> list[IndicatorSeries]:
    """
    Fetch yield and VIX series from Yahoo Finance.
    These are genuinely available on Yahoo and give us real time-series data.
    """
    series_list = []

    for indicator, cfg in INDICATOR_TICKERS.items():
        ticker = cfg["ticker"]
        try:
            raw = yf.download(
                tickers  = ticker,
                period   = f"{lookback_days}d",
                interval = "1d",
                progress = False,
                auto_adjust = True,
            )
            if raw.empty:
                continue

            close_col = "Close"
            if isinstance(raw.columns, pd.MultiIndex):
                candidates = [c for c in raw.columns if c[0] == "Close"]
                if not candidates:
                    continue
                close_col = candidates[0]

            close = raw[close_col].dropna()
            if close.empty:
                continue

            # Sample to monthly points (last trading day of each month)
            monthly = close.resample("ME").last().dropna()

            points = [
                IndicatorPoint(
                    indicator   = indicator,
                    name        = cfg["name"],
                    value       = round(float(v), 4),
                    unit        = cfg["unit"],
                    as_of       = pd.Timestamp(dt).date(),
                    source      = "Yahoo Finance",
                    ticker_used = ticker,
                )
                for dt, v in monthly.items()
            ]

            if points:
                series_list.append(IndicatorSeries(
                    indicator = indicator,
                    name      = cfg["name"],
                    unit      = cfg["unit"],
                    source    = "Yahoo Finance",
                    points    = points,
                ))

        except Exception:
            pass  # Fetcher is best-effort; warnings captured in snapshot fetch

    return series_list


# ---------------------------------------------------------------------------
# Main entry point: build the full MacroContext
# ---------------------------------------------------------------------------

def build_macro_context() -> MacroContext:
    """
    Fetch all data and assemble a MacroContext ready for the LLM layer.

    Call this once per analysis run (or cache it for N minutes in production).
    """
    snapshot, snap_warnings = fetch_market_snapshot()
    simulated_series        = _build_simulated_series()
    yield_series            = fetch_yield_series(lookback_days=365)

    # Merge: simulated macro + real yield/VIX series from Yahoo
    all_series = simulated_series + yield_series

    # Add yield spread as a derived series if we have both yields
    y10_s = next((s for s in yield_series if s.indicator == IndicatorType.YIELD_10Y), None)
    y2_s  = next((s for s in yield_series if s.indicator == IndicatorType.YIELD_2Y), None)

    warnings_out = snap_warnings[:]

    if not yield_series:
        warnings_out.append(
            "Yahoo Finance yield data unavailable. "
            "Market context will rely on snapshot prices only."
        )

    return MacroContext(
        as_of          = datetime.utcnow(),
        market         = snapshot,
        indicators     = all_series,
        data_warnings  = warnings_out,
    )