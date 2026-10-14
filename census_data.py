"""
Client for the U.S. Census Bureau Data API (https://www.census.gov/data/developers/data-sets.html).
Metadata (dataset catalog, variables, geography) is public; data queries need a free key:
https://api.census.gov/data/key_signup.html
"""

import os
import re

import numpy as np
import pandas as pd
import requests

BASE_URL = "https://api.census.gov/data"
CENSUS_PREFIX = "CENSUS:"
KEYRING_SERVICE = "census_api"
KEYRING_USERNAME = "api_key"

STATE_FIPS = {
    "01": "Alabama", "02": "Alaska", "04": "Arizona", "05": "Arkansas", "06": "California",
    "08": "Colorado", "09": "Connecticut", "10": "Delaware", "11": "District of Columbia",
    "12": "Florida", "13": "Georgia", "15": "Hawaii", "16": "Idaho", "17": "Illinois",
    "18": "Indiana", "19": "Iowa", "20": "Kansas", "21": "Kentucky", "22": "Louisiana",
    "23": "Maine", "24": "Maryland", "25": "Massachusetts", "26": "Michigan", "27": "Minnesota",
    "28": "Mississippi", "29": "Missouri", "30": "Montana", "31": "Nebraska", "32": "Nevada",
    "33": "New Hampshire", "34": "New Jersey", "35": "New Mexico", "36": "New York",
    "37": "North Carolina", "38": "North Dakota", "39": "Ohio", "40": "Oklahoma", "41": "Oregon",
    "42": "Pennsylvania", "44": "Rhode Island", "45": "South Carolina", "46": "South Dakota",
    "47": "Tennessee", "48": "Texas", "49": "Utah", "50": "Vermont", "51": "Virginia",
    "53": "Washington", "54": "West Virginia", "55": "Wisconsin", "56": "Wyoming", "72": "Puerto Rico",
}


class CensusError(Exception):
    pass


def get_census_key() -> str | None:
    """Look for the key in, in order: Streamlit secrets, the CENSUS_API_KEY env var, then the
    local OS credential store (keyring), skipping any lookup this environment can't do."""
    try:
        import streamlit as st
        key = st.secrets.get("CENSUS_API_KEY")
        if key:
            return key
    except Exception:
        pass
    if os.environ.get("CENSUS_API_KEY"):
        return os.environ["CENSUS_API_KEY"]
    try:
        import keyring
        return keyring.get_password(KEYRING_SERVICE, KEYRING_USERNAME)
    except Exception:
        return None


def is_census_id(sid: str) -> bool:
    return sid.startswith(CENSUS_PREFIX)


def _get_json(url: str, params: dict | None = None, timeout: int = 120):
    resp = requests.get(url, params=params, timeout=timeout, allow_redirects=False)
    if resp.is_redirect:
        # A missing or invalid key is answered with a redirect to an HTML help page.
        if resp.headers.get("X-DataWebAPI-KeyError") or "key" in resp.headers.get("Location", ""):
            raise CensusError(
                "The Census API rejected the API key (missing, invalid, or not yet activated from "
                "the sign-up email)."
            )
        raise CensusError(f"The Census API redirected unexpectedly to {resp.headers.get('Location')}.")
    if resp.status_code == 204:  # valid query, no matching data
        return []
    if resp.status_code >= 400:
        body = resp.text.strip()
        detail = body[:400] if body and not body.startswith("<") else resp.reason  # 404s come back as HTML pages
        raise CensusError(f"The Census API rejected the request (HTTP {resp.status_code}): {detail}")
    try:
        return resp.json()
    except ValueError as e:
        raise CensusError("The Census API returned something other than JSON.") from e


def catalog() -> pd.DataFrame:
    """Every dataset the API offers: its request path (e.g. '2023/acs/acs1' or
    'timeseries/eits/marts'), title, vintage and kind."""
    rows = []
    for d in _get_json(f"{BASE_URL}.json")["dataset"]:
        if d.get("c_isAvailable") is False:
            continue
        name = "/".join(d["c_dataset"])
        vintage = d.get("c_vintage")
        rows.append({
            "path": f"{vintage}/{name}" if vintage else name,
            "title": d.get("title", name).strip(),
            "vintage": vintage,
            "timeseries": bool(d.get("c_isTimeseries")),
            "microdata": bool(d.get("c_isMicrodata")),
            "description": (d.get("description") or "").strip(),
        })
    df = pd.DataFrame(rows).drop_duplicates("path")
    return df.sort_values(["title", "vintage"], ascending=[True, False], na_position="first").reset_index(drop=True)


def variables(path: str) -> dict[str, dict]:
    return _get_json(f"{BASE_URL}/{path}/variables.json")["variables"]


def has_variable(path: str, name: str) -> bool:
    try:
        _get_json(f"{BASE_URL}/{path}/variables/{name}.json")
        return True
    except CensusError:
        return False


def groups(path: str) -> list[dict]:
    try:
        return _get_json(f"{BASE_URL}/{path}/groups.json").get("groups", [])
    except CensusError:
        return []


def group_variables(path: str, group: str) -> dict[str, dict]:
    return _get_json(f"{BASE_URL}/{path}/groups/{group}.json")["variables"]


def geography(path: str) -> list[dict]:
    """Geography levels, e.g. {'name': 'county', 'requires': ['state'], 'wildcard': ['state']}."""
    try:
        return _get_json(f"{BASE_URL}/{path}/geography.json").get("fips", [])
    except CensusError:
        return []


def examples(path: str) -> dict:
    """Census's own example query for a dataset: {'get': [...], 'predicates': {...}, 'fips': [...]}."""
    try:
        ex = _get_json(f"{BASE_URL}/{path}/examples.json")
    except CensusError:
        return {}
    return ex if isinstance(ex, dict) else {}


QUERY_CLAUSE_TYPES = {"fips-for", "fips-in", "ucgid", "datetime"}


def chooseable_variables(var_meta: dict[str, dict]) -> dict[str, dict]:
    """Variables that can go in `get=`: everything except the geography/time query clauses
    and the annotation fields. (Census's `predicateOnly` flag can't be used for this: group
    listings set it on every variable, including ordinary estimates.)"""
    return {
        name: m for name, m in var_meta.items()
        if m.get("predicateType") not in QUERY_CLAUSE_TYPES and name not in ("for", "in", "ucgid", "time")
        and not str(m.get("label", "")).startswith("Annotation of")
    }


_TIME_PATTERNS = [
    (re.compile(r"^(\d{4})-Q([1-4])$"), lambda y, q: pd.Timestamp(int(y), 3 * (int(q) - 1) + 1, 1)),
    (re.compile(r"^(\d{4})-(\d{2})-(\d{2})$"), lambda y, m, d: pd.Timestamp(int(y), int(m), int(d))),
    (re.compile(r"^(\d{4})-(\d{2})$"), lambda y, m: pd.Timestamp(int(y), int(m), 1)),
    (re.compile(r"^(\d{4})$"), lambda y: pd.Timestamp(int(y), 1, 1)),
]


def parse_time(value) -> pd.Timestamp:
    """Census `time` values ('2024', '2024-03', '2024-Q1', '2024-03-15') -> period start."""
    v = str(value).strip()
    for pattern, build in _TIME_PATTERNS:
        m = pattern.match(v)
        if m:
            return build(*m.groups())
    return pd.NaT


def get_data(path: str, get_vars: list[str], predicates: dict, key: str | None,
             numeric_vars: set[str], geo_cols: set[str]) -> pd.DataFrame:
    """Run a query and return it as a table, with value columns converted to numbers and, for
    time series, a parsed `Date` column. A column becomes numeric if Census types it int/float,
    or if at least 90% of its values parse as numbers (string-typed values like EITS's
    `cell_value` carry occasional markers such as '(S)', which become NaN). Geography columns,
    query filters, identifiers (`..._code`, `..._id`) and codes with leading zeros stay text."""
    params = {"get": ",".join(get_vars), **predicates}
    if key:
        params["key"] = key
    rows = _get_json(f"{BASE_URL}/{path}", params)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows[1:], columns=rows[0])
    df = df.loc[:, ~df.columns.duplicated()]
    for col in df.columns:
        if col in geo_cols or col == "time" or col in predicates:
            continue
        values = df[col].replace("", np.nan)
        if not values.notna().any():
            continue
        num = pd.to_numeric(values, errors="coerce")
        if col in numeric_vars:
            df[col] = num
            continue
        if re.search(r"(_code|_id)$", col, re.IGNORECASE):
            continue
        share_numeric = num.notna().sum() / values.notna().sum()
        code_like = values.dropna().astype(str).str.match(r"^0\d").any()
        if share_numeric >= 0.9 and not code_like:
            df[col] = num
    if "time" in df.columns:
        df["Date"] = pd.to_datetime(df["time"].map(parse_time))
    return df


def series_key_cols(df: pd.DataFrame) -> list[str]:
    """Columns that identify one time series in a result: every text column except time."""
    return [
        c for c in df.columns
        if c not in ("time", "Date") and not pd.api.types.is_numeric_dtype(df[c])
        and df[c].nunique(dropna=False) > 1
    ]


def series_values(df: pd.DataFrame, filters: dict[str, str], value_col: str) -> pd.Series:
    mask = pd.Series(True, index=df.index)
    for col, val in filters.items():
        mask &= df[col].astype(str) == str(val)
    rows = df[mask & df["Date"].notna()]
    s = pd.Series(pd.to_numeric(rows[value_col], errors="coerce").to_numpy(), index=pd.DatetimeIndex(rows["Date"]))
    return s[~s.index.duplicated()].sort_index()
