"""
Client for the Bureau of Labor Statistics Public Data API v2
(https://www.bls.gov/developers/api_signature_v2.htm). Works without a key, but unregistered
use is limited to 25 requests a day, 10 years and 25 series per request, and no series titles.
A free key (https://data.bls.gov/registrationEngine/) raises that to 500 requests a day,
20 years and 50 series per request, and adds titles.
"""

import os
import re

import pandas as pd
import requests

BASE_URL = "https://api.bls.gov/publicAPI/v2"
BLS_PREFIX = "BLS:"
KEYRING_SERVICE = "bls_api"
KEYRING_USERNAME = "api_key"
SERIES_ID_RE = re.compile(r"^[A-Z0-9_#-]+$")  # BLS: no lower case or other special characters


class BLSError(Exception):
    pass


def get_bls_key() -> str | None:
    """Look for the key in, in order: Streamlit secrets, the BLS_API_KEY env var, then the
    local OS credential store (keyring), skipping any lookup this environment can't do."""
    try:
        import streamlit as st
        key = st.secrets.get("BLS_API_KEY")
        if key:
            return key
    except Exception:
        pass
    if os.environ.get("BLS_API_KEY"):
        return os.environ["BLS_API_KEY"]
    try:
        import keyring
        return keyring.get_password(KEYRING_SERVICE, KEYRING_USERNAME)
    except Exception:
        return None


def is_bls_id(sid: str) -> bool:
    return sid.startswith(BLS_PREFIX)


def limits(key: str | None) -> tuple[int, int]:
    """(max years, max series) per request."""
    return (20, 50) if key else (10, 25)


def _check(payload: dict) -> dict:
    if payload.get("status") != "REQUEST_SUCCEEDED":
        raise BLSError("BLS didn't process the request: " + " ".join(payload.get("message") or [payload.get("status", "")]))
    return payload


def _get(path: str, params: dict | None = None) -> dict:
    resp = requests.get(f"{BASE_URL}/{path}", params=params, timeout=60)
    resp.raise_for_status()
    return _check(resp.json())


def surveys() -> dict[str, str]:
    """Survey abbreviation -> name, e.g. {'CU': 'CPI for All Urban Consumers (CPI-U)', ...}."""
    rows = _get("surveys")["Results"]["survey"]
    return dict(sorted((s["survey_abbreviation"], s["survey_name"].strip()) for s in rows))


def popular(survey: str | None = None) -> list[str]:
    """Series IDs of the 25 most requested series, overall or for one survey."""
    results = _get("timeseries/popular", {"survey": survey} if survey else None)["Results"]
    return [s["seriesID"] for s in results.get("series", [])]


_PERIOD_START = {f"M{m:02d}": (m, "month") for m in range(1, 13)}
_PERIOD_START.update({f"Q0{q}": (3 * (q - 1) + 1, "quarter") for q in range(1, 5)})
_PERIOD_START.update({"S01": (1, "half"), "S02": (7, "half"), "A01": (1, "year")})


def to_series(observations: list[dict]) -> pd.Series:
    """BLS observations -> values indexed by period start. Annual averages (M13, Q05, S03) are
    dropped; an annual-only series keeps its A01 values."""
    rows = []
    for obs in observations:
        start = _PERIOD_START.get(obs.get("period", ""))
        if start:
            month, kind = start
            value = pd.to_numeric(str(obs.get("value", "")).replace(",", ""), errors="coerce")
            rows.append((pd.Timestamp(int(obs["year"]), month, 1), kind, value))
    if not rows:
        return pd.Series(dtype=float, index=pd.DatetimeIndex([]))
    df = pd.DataFrame(rows, columns=["date", "kind", "value"])
    if (df["kind"] != "year").any():
        df = df[df["kind"] != "year"]
    s = pd.Series(df["value"].to_numpy(dtype=float), index=pd.DatetimeIndex(df["date"]))
    return s[~s.index.duplicated()].sort_index()


def fetch(series_ids: list[str], start_year: int, end_year: int, key: str | None) -> tuple[dict[str, pd.Series], dict[str, str], list[str]]:
    """Data for several series over a year range, split into as many requests as BLS's
    per-request limits need. Returns ({id: values}, {id: title}, BLS's notes). Titles come
    only with a key (BLS's catalog option)."""
    max_years, max_series = limits(key)
    observations: dict[str, list[dict]] = {sid: [] for sid in series_ids}
    titles: dict[str, str] = {}
    notes: list[str] = []
    for i in range(0, len(series_ids), max_series):
        batch = series_ids[i:i + max_series]
        for y0 in range(start_year, end_year + 1, max_years):
            body = {"seriesid": batch, "startyear": str(y0), "endyear": str(min(y0 + max_years - 1, end_year))}
            if key:
                body.update(registrationkey=key, catalog=True)
            resp = requests.post(f"{BASE_URL}/timeseries/data/", json=body, timeout=90)
            resp.raise_for_status()
            payload = _check(resp.json())
            notes += [m for m in payload.get("message") or [] if m not in notes]
            for s in payload["Results"].get("series", []):
                observations.setdefault(s["seriesID"], []).extend(s.get("data", []))
                title = (s.get("catalog") or {}).get("series_title")
                if title:
                    titles[s["seriesID"]] = title.strip()
    return {sid: to_series(obs) for sid, obs in observations.items()}, titles, notes
