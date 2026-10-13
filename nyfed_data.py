"""
Client for the Federal Reserve Bank of New York's Markets Data APIs
(https://markets.newyorkfed.org/static/docs/markets-api.html). Public, no API key needed.
"""

import json
import re
from urllib.parse import quote

import numpy as np
import pandas as pd
import requests

BASE_URL = "https://markets.newyorkfed.org"
NYFED_PREFIX = "NYFED:"
PD_PREFIX = "NYFED:PD:"

# Request paths a user may type into the custom-endpoint box: JSON endpoints under /api (or
# /beta/api) on the NY Fed host only, so the app can't be pointed at arbitrary URLs.
CUSTOM_PATH_RE = re.compile(r"^(?!.*\.\.)/(beta/)?api/[\w\-./ %]+\.json(\?[\w\-.=&%+ ]*)?$")
BARE_STAR_RE = re.compile(r"(:\s*)\*(?=\s*[,}\]])")

RATE_TYPES = {"SOFR": "secured", "TGCR": "secured", "BGCR": "secured", "EFFR": "unsecured", "OBFR": "unsecured"}

SOMA_COMPONENTS = {
    "TOTAL": ("total", "total"),
    "BILLS": ("bills", "Treasury bills"),
    "NOTESBONDS": ("notesbonds", "Treasury notes & bonds"),
    "TIPS": ("tips", "TIPS"),
    "FRN": ("frn", "Treasury FRNs"),
    "MBS": ("mbs", "agency MBS"),
    "CMBS": ("cmbs", "agency CMBS"),
    "AGENCIES": ("agencies", "agency debt"),
}

# Time series that can be charted alongside FRED series: ID -> display label.
NYFED_SERIES = {
    **{f"{NYFED_PREFIX}{r}": f"{r} (%)" for r in RATE_TYPES},
    **{f"{NYFED_PREFIX}{r}_VOL": f"{r} volume ($bn)" for r in RATE_TYPES},
    f"{NYFED_PREFIX}SOFR_30D_AVG": "SOFR 30-day average (%)",
    f"{NYFED_PREFIX}SOFR_90D_AVG": "SOFR 90-day average (%)",
    f"{NYFED_PREFIX}SOFR_180D_AVG": "SOFR 180-day average (%)",
    f"{NYFED_PREFIX}SOFR_INDEX": "SOFR Index",
    **{f"{NYFED_PREFIX}SOMA_{k}": f"SOMA holdings: {label} ($bn)" for k, (_, label) in SOMA_COMPONENTS.items()},
    f"{NYFED_PREFIX}RRP_ACCEPTED": "Fed reverse repo operations: accepted ($bn)",
    f"{NYFED_PREFIX}REPO_ACCEPTED": "Fed repo operations: accepted ($bn)",
}

SOFRAI_FIELDS = {
    "SOFR_30D_AVG": "average30day",
    "SOFR_90D_AVG": "average90day",
    "SOFR_180D_AVG": "average180day",
    "SOFR_INDEX": "index",
}


class NYFedError(Exception):
    pass


def is_nyfed_id(sid: str) -> bool:
    return sid.startswith(NYFED_PREFIX)


def nyfed_label(sid: str) -> str | None:
    if sid in NYFED_SERIES:
        return NYFED_SERIES[sid]
    if sid.startswith(PD_PREFIX):
        return f"Primary dealers: {sid[len(PD_PREFIX):]}"
    return None


def get_json(path: str, params: dict | None = None, beta: bool = False, timeout: int = 90):
    """GET a JSON endpoint, e.g. get_json('/api/rates/all/latest.json'). `beta` prefixes the
    path with /beta (the API's test versions). A 400 carries the API's validation messages."""
    if beta and not path.startswith("/beta/"):
        path = "/beta" + path
    resp = requests.get(BASE_URL + quote(path, safe="/.?=&%+-_"), params=params, timeout=timeout)
    if resp.status_code == 400:
        try:
            detail = resp.json()
        except ValueError:
            detail = resp.text
        raise NYFedError(f"The NY Fed API rejected the request (HTTP 400): {detail or 'invalid path or parameters'}")
    resp.raise_for_status()
    try:
        return resp.json()
    except ValueError:
        # Some endpoints (e.g. market share) write suppressed values as a bare `*`, which isn't valid JSON.
        return json.loads(BARE_STAR_RE.sub(r'\1"*"', resp.text))


def _empty_series() -> pd.Series:
    return pd.Series(dtype=float, index=pd.DatetimeIndex([]))


def _to_series(records: list[dict], date_field: str, value_field: str, scale: float = 1.0) -> pd.Series:
    if not records:
        return _empty_series()
    df = pd.DataFrame(records)
    s = pd.Series(
        pd.to_numeric(df[value_field].replace("", np.nan), errors="coerce").to_numpy() / scale,
        index=pd.to_datetime(df[date_field]),
    )
    s = s[~s.index.duplicated()].sort_index()
    return s


def get_series(sid: str, start: str, end: str, beta: bool = False) -> pd.Series:
    """Fetch one charted NY Fed series (see NYFED_SERIES, or NYFED:PD:<keyid>) between
    `start` and `end` (yyyy-mm-dd), as a float Series with a DatetimeIndex."""
    key = sid[len(NYFED_PREFIX):]
    dates = {"startDate": start, "endDate": end}

    if key in RATE_TYPES or (key.endswith("_VOL") and key[:-4] in RATE_TYPES):
        rate = key[:-4] if key.endswith("_VOL") else key
        is_vol = key.endswith("_VOL")
        data = get_json(
            f"/api/rates/{RATE_TYPES[rate]}/{rate.lower()}/search.json",
            {**dates, "type": "volume" if is_vol else "rate"}, beta,
        )
        s = _to_series(data["refRates"], "effectiveDate", "volumeInBillions" if is_vol else "percentRate")
    elif key in SOFRAI_FIELDS:
        data = get_json("/api/rates/secured/sofrai/search.json", dates, beta)
        s = _to_series(data["refRates"], "effectiveDate", SOFRAI_FIELDS[key])
    elif key.startswith("SOMA_") and key[5:] in SOMA_COMPONENTS:
        data = get_json("/api/soma/summary.json", beta=beta)
        s = _to_series(data["soma"]["summary"], "asOfDate", SOMA_COMPONENTS[key[5:]][0], scale=1e9)
    elif key in ("RRP_ACCEPTED", "REPO_ACCEPTED"):
        op_type = "reverserepo" if key == "RRP_ACCEPTED" else "repo"
        data = get_json("/api/rp/results/search.json", {**dates, "operationTypes": op_type}, beta)
        ops = pd.DataFrame(data["repo"]["operations"])
        if ops.empty:
            return _empty_series()
        amounts = pd.to_numeric(ops["totalAmtAccepted"], errors="coerce") / 1e9
        s = amounts.groupby(pd.to_datetime(ops["operationDate"])).sum().sort_index()
    elif sid.startswith(PD_PREFIX):
        keyid = sid[len(PD_PREFIX):]
        data = get_json(f"/api/pd/get/{keyid}.json", beta=beta)
        s = _to_series(data["pd"]["timeseries"], "asofdate", "value")
    else:
        raise NYFedError(f"Unknown NY Fed series ID: {sid}")

    s.index.name = None
    return s.loc[start:end]


def list_pd_series(beta: bool = False) -> dict[str, str]:
    """Primary dealer statistics keyids -> descriptions (latest series break's wording)."""
    data = get_json("/api/pd/list/timeseries.json", beta=beta)
    out: dict[str, str] = {}
    for row in data["pd"]["timeseries"]:
        out.setdefault(row["keyid"], row.get("description", ""))
    return dict(sorted(out.items()))


def _flatten_dict(d: dict, prefix: str = "") -> dict:
    out = {}
    for k, v in d.items():
        name = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(_flatten_dict(v, f"{name}."))
        elif not isinstance(v, list):
            out[name] = v
    return out


ID_FIELDS = ("operationId", "operationDate", "tradeDate", "asOfDate", "asofdate", "effectiveDate")


def _is_record_list(v) -> bool:
    return isinstance(v, list) and bool(v) and all(isinstance(x, dict) for x in v)


def json_tables(obj, path: str = "") -> dict[str, pd.DataFrame]:
    """Every table in an API response: each list of records becomes a DataFrame keyed by its
    JSON path (e.g. 'ambs.auctions'). Nested record lists (e.g. an operation's 'details')
    become their own table, with the parent's ID/date columns carried over to join on. A
    dict's loose scalar fields (e.g. a release date and title) become a one-row table."""
    tables: dict[str, pd.DataFrame] = {}
    if isinstance(obj, dict):
        scalars = _flatten_dict({k: v for k, v in obj.items() if not isinstance(v, (list, dict))})
        has_children = any(isinstance(v, (list, dict)) for v in obj.values())
        if scalars and has_children:
            tables[f"{path} (fields)" if path else "fields"] = pd.DataFrame([scalars])
        for k, v in obj.items():
            if isinstance(v, (list, dict)):
                tables.update(json_tables(v, f"{path}.{k}" if path else k))
        if scalars and not has_children:
            tables[path or "fields"] = pd.DataFrame([scalars])
    elif _is_record_list(obj):
        rows, children = [], {}
        for rec in obj:
            row = {}
            for k, v in rec.items():
                if _is_record_list(v):
                    parent = {f: rec[f] for f in ID_FIELDS if f in rec}
                    children.setdefault(k, []).extend({**parent, **_flatten_dict(c)} for c in v)
                elif isinstance(v, dict):
                    row.update(_flatten_dict(v, f"{k}."))
                elif not isinstance(v, list):
                    row[k] = v
            rows.append(row)
        tables[path or "records"] = pd.DataFrame(rows)
        for k, child_rows in children.items():
            tables[f"{path}.{k}"] = pd.DataFrame(child_rows)
    elif isinstance(obj, list) and obj:
        tables[path or "values"] = pd.DataFrame({path.rsplit(".", 1)[-1] or "value": obj})
    return {name: coerce_types(df) for name, df in tables.items()}


def coerce_types(df: pd.DataFrame) -> pd.DataFrame:
    """The API returns many numbers and dates as strings; convert any column whose every
    non-blank value parses."""
    df = df.copy()
    for col in df.columns:
        if not (df[col].dtype == object or pd.api.types.is_string_dtype(df[col])):
            continue
        s = df[col].replace("", np.nan)
        if s.notna().sum() == 0:
            continue
        num = pd.to_numeric(s, errors="coerce")
        if num.notna().sum() == s.notna().sum():
            df[col] = num
            continue
        if "date" in col.lower():
            dt = pd.to_datetime(s, errors="coerce", format="mixed")
            if dt.notna().sum() == s.notna().sum():
                df[col] = dt
    return df
