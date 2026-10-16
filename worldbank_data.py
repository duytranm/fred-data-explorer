"""
Client for the World Bank Indicators API v2
(https://datahelpdesk.worldbank.org/knowledgebase/topics/125589-developer-information).
Public, no key needed. A series is one indicator for one country (or aggregate) at one
frequency, e.g. GDP growth (NY.GDP.MKTP.KD.ZG) for the United States (USA), annual.
"""

import re

import pandas as pd
import requests

BASE_URL = "https://api.worldbank.org/v2"
WB_PREFIX = "WB:"
WDI_SOURCE = "2"
FREQUENCIES = {"A": "Annual", "Q": "Quarterly", "M": "Monthly"}


class WorldBankError(Exception):
    pass


def is_wb_id(sid: str) -> bool:
    return sid.startswith(WB_PREFIX)


def make_id(source: str, indicator: str, country: str, freq: str) -> str:
    return f"{WB_PREFIX}{source}:{indicator}:{country}:{freq}"


def parse_id(sid: str) -> tuple[str, str, str, str]:
    """'WB:2:NY.GDP.MKTP.KD.ZG:USA:A' -> ('2', 'NY.GDP.MKTP.KD.ZG', 'USA', 'A')."""
    parts = sid[len(WB_PREFIX):].split(":")
    if len(parts) != 4 or not all(parts) or parts[3].upper() not in FREQUENCIES:
        raise WorldBankError(
            f"Not a World Bank series ID (expected WB:<source>:<indicator>:<country>:<A|Q|M>): {sid}")
    return parts[0], parts[1], parts[2], parts[3].upper()


def _get(path: str, params: dict | None = None) -> tuple[dict, list]:
    """One page of a list endpoint as (meta, rows); the API reports errors as a message list."""
    resp = requests.get(f"{BASE_URL}/{path}", params={"format": "json", **(params or {})}, timeout=120)
    resp.raise_for_status()
    body = resp.json()
    if not isinstance(body, list) or len(body) < 2 or not isinstance(body[0], dict) or "page" not in body[0]:
        messages = body[0].get("message", []) if isinstance(body, list) and body and isinstance(body[0], dict) else []
        detail = "; ".join(dict.fromkeys(m.get("value", "") for m in messages)) or str(body)[:200]
        raise WorldBankError(f"The World Bank API rejected the request: {detail}")
    return body[0], body[1] or []


def _get_all(path: str, params: dict | None = None) -> list:
    rows, page = [], 1
    while True:
        meta, data = _get(path, {"per_page": 20000, "page": page, **(params or {})})
        rows += data
        if page >= int(meta.get("pages") or 1):
            return rows
        page += 1


def sources() -> dict[str, str]:
    """Database id -> name, e.g. {'2': 'World Development Indicators', ...}, by name."""
    rows = _get_all("sources")
    return dict(sorted(((s["id"], s["name"].strip()) for s in rows), key=lambda kv: kv[1].lower()))


def indicators(source: str) -> list[dict]:
    """A database's indicators: [{'id', 'name', 'note', 'organization'}], by name."""
    rows = _get_all("indicator", {"source": source})
    out = [{"id": r["id"], "name": (r.get("name") or r["id"]).strip(), "note": (r.get("sourceNote") or "").strip(),
            "organization": (r.get("sourceOrganization") or "").strip()} for r in rows]
    return sorted(out, key=lambda r: r["name"].lower())


def countries() -> list[dict]:
    """Countries and aggregates (regions, income groups): [{'id' (ISO3), 'name', 'aggregate'}]."""
    rows = _get_all("country")
    out = [{"id": c["id"], "name": c["name"].strip(), "aggregate": c.get("region", {}).get("value") == "Aggregates"}
           for c in rows]
    return sorted(out, key=lambda c: (c["aggregate"], c["name"]))


def date_range(freq: str, start_year: int, end_year: int) -> str:
    return {"A": f"{start_year}:{end_year}", "Q": f"{start_year}Q1:{end_year}Q4",
            "M": f"{start_year}M01:{end_year}M12"}[freq]


_PERIOD = [
    (re.compile(r"^(\d{4})M(\d{2})$"), lambda y, m: pd.Timestamp(int(y), int(m), 1)),
    (re.compile(r"^(\d{4})Q([1-4])$"), lambda y, q: pd.Timestamp(int(y), 3 * (int(q) - 1) + 1, 1)),
    (re.compile(r"^(\d{4})$"), lambda y: pd.Timestamp(int(y), 1, 1)),
]


def parse_period(value) -> pd.Timestamp:
    """'2024', '2024Q2', '2024M05' -> period start (NaT if unrecognised)."""
    v = str(value).strip()
    for pattern, build in _PERIOD:
        m = pattern.match(v)
        if m:
            return build(*m.groups())
    return pd.NaT


def get_data(source: str, indicator: str, country_codes: list[str], freq: str,
             start_year: int, end_year: int) -> pd.DataFrame:
    """Observations for one indicator across countries: country, country_name, date, Date,
    value (numeric). Missing values (null) are dropped."""
    rows = _get_all(
        f"country/{';'.join(country_codes)}/indicator/{indicator}",
        {"source": source, "date": date_range(freq, start_year, end_year)},
    )
    records = [
        {"country": r.get("countryiso3code") or (r.get("country") or {}).get("id"),
         "country_name": (r.get("country") or {}).get("value"), "date": r.get("date"), "value": r.get("value")}
        for r in rows if r.get("value") is not None
    ]
    df = pd.DataFrame(records, columns=["country", "country_name", "date", "value"])
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df["Date"] = pd.to_datetime(df["date"].map(parse_period))
    return df.dropna(subset=["value", "Date"]).sort_values(["country", "Date"]).reset_index(drop=True)


def series_values(df: pd.DataFrame, country: str) -> pd.Series:
    rows = df[df["country"].str.upper() == country.upper()]
    s = pd.Series(rows["value"].to_numpy(dtype=float), index=pd.DatetimeIndex(rows["Date"]))
    return s[~s.index.duplicated()].sort_index()
