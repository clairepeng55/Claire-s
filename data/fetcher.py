"""Market context fetching utilities."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
import io
from typing import Iterable
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import urlopen


DEFAULT_MARKET_SYMBOLS = {
    "SPY.US": "S&P 500 ETF",
    "QQQ.US": "Nasdaq 100 ETF",
    "TLT.US": "20+ Year Treasury ETF",
    "GLD.US": "Gold ETF",
}


def get_market_context(
    symbols: Iterable[str] | None = None,
    *,
    timeout_seconds: float = 5.0,
) -> dict:
    """Return a compact market context payload for demos and prompts.

    Stooq provides delayed public quote data without an API key. If that service
    is temporarily unavailable, this function returns an explanatory payload
    instead of crashing the API server.
    """

    fetched_at = datetime.now(timezone.utc).isoformat()
    selected_symbols = list(symbols or DEFAULT_MARKET_SYMBOLS)

    try:
        quotes = fetch_stooq_quotes(selected_symbols, timeout_seconds=timeout_seconds)
    except (HTTPError, URLError, TimeoutError, OSError) as exc:
        return {
            "status": "unavailable",
            "source": "stooq.com",
            "fetched_at": fetched_at,
            "error": str(exc),
            "market": {"quotes": []},
        }

    return {
        "status": "ok",
        "source": "stooq.com",
        "fetched_at": fetched_at,
        "market": {"quotes": quotes},
    }


def fetch_stooq_quotes(
    symbols: Iterable[str],
    *,
    timeout_seconds: float = 5.0,
) -> list[dict]:
    """Fetch delayed quote data from Stooq's CSV endpoint."""

    symbol_list = [symbol.strip().lower() for symbol in symbols if symbol.strip()]
    if not symbol_list:
        return []

    encoded_symbols = quote(",".join(symbol_list), safe=",")
    url = (
        "https://stooq.com/q/l/"
        f"?s={encoded_symbols}&f=sd2t2ohlcv&h&e=csv"
    )

    with urlopen(url, timeout=timeout_seconds) as response:
        raw = response.read().decode("utf-8")

    reader = csv.DictReader(io.StringIO(raw))
    quotes = []
    for row in reader:
        close = _optional_float(row.get("Close"))
        symbol = (row.get("Symbol") or "").upper()
        if not symbol or close is None:
            continue

        quotes.append(
            {
                "symbol": symbol,
                "name": DEFAULT_MARKET_SYMBOLS.get(symbol, symbol),
                "date": row.get("Date"),
                "time": row.get("Time"),
                "open": _optional_float(row.get("Open")),
                "high": _optional_float(row.get("High")),
                "low": _optional_float(row.get("Low")),
                "close": close,
                "volume": _optional_int(row.get("Volume")),
            }
        )

    return quotes


def _optional_float(value: str | None) -> float | None:
    if value in (None, "", "N/D"):
        return None
    return float(value)


def _optional_int(value: str | None) -> int | None:
    if value in (None, "", "N/D"):
        return None
    return int(float(value))
