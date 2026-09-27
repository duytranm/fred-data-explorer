"""
Fetch and manipulate multiple FRED (Federal Reserve Economic Data) series.

Setup (one time):
    1. Get a free API key: https://fredapi.stlouisfed.org/docs/api/api_key.html
    2. Store it once: python setup_api_key.py
       (saved to Windows Credential Manager via `keyring` - no env var needed again)
    - On Streamlit Community Cloud, instead add FRED_API_KEY under the app's Secrets
      settings; keyring has no backend there, so this module falls back automatically.

Usage:
    python fred_data.py
"""

import os
import pandas as pd
from fredapi import Fred

KEYRING_SERVICE = "fred_api"
KEYRING_USERNAME = "api_key"


def get_api_key() -> str | None:
    """Look for the key in, in order: Streamlit secrets (for Streamlit Cloud), the
    FRED_API_KEY env var, then the local OS credential store (keyring). Each lookup is
    wrapped so a method that isn't available in the current environment (e.g. no
    secrets.toml, or no keyring backend on a Linux container) is skipped rather than
    raising."""
    try:
        import streamlit as st
        key = st.secrets.get("FRED_API_KEY")
        if key:
            return key
    except Exception:
        pass

    env_key = os.environ.get("FRED_API_KEY")
    if env_key:
        return env_key

    try:
        import keyring
        return keyring.get_password(KEYRING_SERVICE, KEYRING_USERNAME)
    except Exception:
        return None


def get_recession_series() -> pd.Series:
    """NBER recession indicator (USREC): 1 during a recession month, 0 otherwise."""
    api_key = get_api_key()
    if not api_key:
        raise ValueError("No FRED API key found. Run `python setup_api_key.py` once to store your key.")
    return Fred(api_key=api_key).get_series("USREC")


def shade_recessions(ax, recession_series: pd.Series):
    """Shade NBER recession periods on a matplotlib Axes with a date x-axis."""
    xlim = ax.get_xlim()
    starts, ends = [], []
    in_rec = False
    for date, val in recession_series.items():
        if val == 1 and not in_rec:
            starts.append(date)
            in_rec = True
        elif val == 0 and in_rec:
            ends.append(date)
            in_rec = False
    if in_rec:
        ends.append(recession_series.index[-1])
    for s, e in zip(starts, ends):
        ax.axvspan(s, e, color="gray", alpha=0.15, zorder=0)
    ax.set_xlim(xlim)


class FredDatasets:
    """Fetch multiple FRED series and manipulate them together as one DataFrame."""

    def __init__(self, api_key: str | None = None):
        api_key = api_key or get_api_key()
        if not api_key:
            raise ValueError(
                "No FRED API key found. Run `python setup_api_key.py` once to store your key, "
                "or set the FRED_API_KEY environment variable, or pass api_key= explicitly."
            )
        self.fred = Fred(api_key=api_key)
        self.series: dict[str, pd.Series] = {}

    def add_series(self, series_id: str, label: str | None = None,
                    start: str | None = None, end: str | None = None) -> pd.Series:
        """Fetch one FRED series (e.g. 'GDP', 'UNRATE', 'CPIAUCSL') and store it under `label`."""
        label = label or series_id
        s = self.fred.get_series(series_id, observation_start=start, observation_end=end)
        s.name = label
        self.series[label] = s
        return s

    def add_many(self, series_map: dict[str, str],
                  start: str | None = None, end: str | None = None) -> pd.DataFrame:
        """series_map: {series_id: label}. Fetches each and returns the combined frame."""
        for series_id, label in series_map.items():
            self.add_series(series_id, label, start=start, end=end)
        return self.as_dataframe()

    def as_dataframe(self) -> pd.DataFrame:
        """Combine all fetched series into one DataFrame, aligned by date (outer join)."""
        if not self.series:
            return pd.DataFrame()
        return pd.concat(self.series.values(), axis=1)

    # --- manipulation helpers -------------------------------------------------

    def resample(self, freq: str = "ME", how: str = "mean") -> pd.DataFrame:
        """Resample combined data to a new frequency, e.g. 'ME' (month), 'QE' (quarter), 'YE' (year)."""
        df = self.as_dataframe()
        resampled = df.resample(freq)
        return getattr(resampled, how)()

    def pct_change(self, periods: int = 1) -> pd.DataFrame:
        """Percent change over `periods` observations, per column."""
        return self.as_dataframe().pct_change(periods=periods) * 100

    def yoy_change(self, freq: str = "ME") -> pd.DataFrame:
        """Year-over-year percent change, assuming monthly-ish data (12 periods back)."""
        df = self.resample(freq)
        return df.pct_change(periods=12) * 100

    def normalize(self, base_date: str | None = None) -> pd.DataFrame:
        """Rebase every series to 100 at `base_date` (or the first common non-NaN row)."""
        df = self.as_dataframe()
        if base_date:
            base = df.loc[base_date]
        else:
            base = df.dropna().iloc[0]
        return df.div(base) * 100

    def correlation(self) -> pd.DataFrame:
        """Correlation matrix across all series (pairwise, on overlapping dates)."""
        return self.as_dataframe().corr()

    def dropna_common(self) -> pd.DataFrame:
        """Rows where every series has data (inner-join equivalent)."""
        return self.as_dataframe().dropna()

    def to_csv(self, path: str):
        self.as_dataframe().to_csv(path)
        print(f"Saved {len(self.series)} series to {path}")


if __name__ == "__main__":
    from pathlib import Path

    fd = FredDatasets()

    # Example: unemployment rate, CPI, and real GDP
    fd.add_many(
        {
            "UNRATE": "Unemployment Rate",
            "CPIAUCSL": "CPI",
            "GDP": "Real GDP",
        },
        start="2000-01-01",
    )

    df = fd.as_dataframe()
    print("\nRaw data (tail):")
    print(df.tail())

    print("\nYear-over-year % change (tail):")
    print(fd.yoy_change().tail())

    print("\nCorrelation matrix:")
    print(fd.correlation())

    out_path = Path(__file__).resolve().parent / "fred_data.csv"
    fd.to_csv(str(out_path))
