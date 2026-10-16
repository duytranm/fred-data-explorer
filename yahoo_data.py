"""
Stock, ETF, index, FX and crypto prices from Yahoo Finance via the unofficial `yfinance`
library. Yahoo has no official finance API: this reads the website's own data, which Yahoo
licenses for personal use, may rate-limit (especially from cloud servers), and can change
without notice.
"""

import datetime

import pandas as pd
import yfinance as yf

YF_PREFIX = "YF:"
FIELDS = {"CLOSE": "Close", "OPEN": "Open", "HIGH": "High", "LOW": "Low", "VOLUME": "Volume"}
INTERVALS = {"1d": "Daily", "1wk": "Weekly", "1mo": "Monthly"}


class YahooError(Exception):
    pass


def is_yf_id(sid: str) -> bool:
    return sid.startswith(YF_PREFIX)


def make_id(ticker: str, field: str = "CLOSE") -> str:
    """'YF:AAPL' for the (split- and dividend-adjusted) close, 'YF:AAPL:VOLUME' for other fields."""
    return f"{YF_PREFIX}{ticker}" + ("" if field == "CLOSE" else f":{field}")


def parse_id(sid: str) -> tuple[str, str]:
    """'YF:AAPL' -> ('AAPL', 'CLOSE'); 'YF:^GSPC:VOLUME' -> ('^GSPC', 'VOLUME')."""
    body = sid[len(YF_PREFIX):]
    ticker, sep, last = body.rpartition(":")
    if sep and last.upper() in FIELDS:
        field = last.upper()
    else:
        ticker, field = body, "CLOSE"
    if not ticker:
        raise YahooError(f"Not a Yahoo Finance series ID (expected YF:<ticker> or YF:<ticker>:<field>): {sid}")
    return ticker.upper(), field


def download(tickers: list[str], start: str, end: str, interval: str = "1d") -> dict[str, pd.DataFrame]:
    """Adjusted daily/weekly/monthly prices: {ticker: frame of Open/High/Low/Close/Volume}.
    Tickers Yahoo returns nothing for are simply absent from the result."""
    # yfinance's end date is exclusive; include the chosen end day.
    end_exclusive = (datetime.date.fromisoformat(end) + datetime.timedelta(days=1)).isoformat()
    try:
        df = yf.download(tickers, start=start, end=end_exclusive, interval=interval, auto_adjust=True,
                         progress=False, group_by="ticker", threads=True)
    except Exception as e:  # yfinance raises its own types (rate limits, site changes, network errors)
        raise YahooError(f"Yahoo Finance request failed ({type(e).__name__}: {e}).") from e
    if df is None or df.empty:
        raise YahooError(
            "Yahoo Finance returned no data: check the tickers and dates. Yahoo also sometimes limits "
            "requests from cloud servers; if so, try again in a few minutes."
        )
    out = {}
    for t in tickers:
        if isinstance(df.columns, pd.MultiIndex):
            if t not in df.columns.get_level_values(0):
                continue
            frame = df[t]
        else:
            frame = df
        frame = frame[[c for c in FIELDS.values() if c in frame.columns]].dropna(how="all")
        if not frame.empty:
            frame.index = pd.DatetimeIndex(frame.index).tz_localize(None)
            out[t] = frame
    return out
