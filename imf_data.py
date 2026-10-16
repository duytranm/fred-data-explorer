"""
Client for the IMF's SDMX 2.1 data API (https://data.imf.org/en/Resource-Pages/IMF-API).
Public datasets need no key. Series are identified by dataflow and SDMX key, e.g. CPI with
key USA.CPI._T.IX.M (country . index type . COICOP category . transformation . frequency).
"""

import requests

from sdmx_data import (  # noqa: F401  (re-exported for callers)
    clean_name, csv_to_frame, key_string, parse_availability, parse_period, parse_structure,
    series_key, series_label, series_values,
)

BASE_URL = "https://api.imf.org/external/sdmx/2.1"
IMF_PREFIX = "IMF:"
JSON = {"Accept": "application/json"}
CSV = {"Accept": "application/vnd.sdmx.data+csv;version=1.0.0"}


class IMFError(Exception):
    pass


def is_imf_id(sid: str) -> bool:
    return sid.startswith(IMF_PREFIX)


def make_id(agency: str, flow: str, key: str) -> str:
    return f"{IMF_PREFIX}{agency}:{flow}:{key}"


def parse_id(sid: str) -> tuple[str, str, str]:
    """'IMF:IMF.STA:CPI:USA.CPI._T.IX.M' -> ('IMF.STA', 'CPI', 'USA.CPI._T.IX.M')."""
    parts = sid[len(IMF_PREFIX):].split(":", 2)
    if len(parts) != 3 or not all(parts):
        raise IMFError(f"Not an IMF series ID (expected IMF:<agency>:<dataset>:<key>): {sid}")
    return parts[0], parts[1], parts[2]


def _get(path: str, headers: dict, params: dict | None = None, timeout: int = 120) -> requests.Response:
    resp = requests.get(f"{BASE_URL}/{path}", headers=headers, params=params, timeout=timeout)
    if resp.status_code == 403:  # what the API answers for unknown or non-public datasets
        raise IMFError("The IMF API refused the request: that dataset doesn't exist or isn't public.")
    if resp.status_code >= 400:
        detail = resp.text.strip()[:300] if resp.text.strip() and not resp.text.lstrip().startswith("<") else resp.reason
        raise IMFError(f"The IMF API rejected the request (HTTP {resp.status_code}): {detail or 'no details given'}")
    return resp


def dataflows(include_vintages: bool = False) -> list[dict]:
    """Available datasets: {'agency', 'id', 'version', 'name', 'description', 'vintage'}, by name.
    Vintages are frozen snapshots of other datasets (e.g. FA_2026_FEB_VINTAGE)."""
    flows = _get("dataflow/all/all/latest", JSON).json()["data"]["dataflows"]
    out = []
    for f in flows:
        vintage = any(a.get("id") == "VINTAGE" for a in f.get("annotations", []))
        if vintage and not include_vintages:
            continue
        out.append({
            "agency": f["agencyID"], "id": f["id"], "version": f["version"],
            "name": clean_name(f.get("name") or f["id"]),
            "description": (f.get("description") or "").strip(), "vintage": vintage,
        })
    return sorted(out, key=lambda f: f["name"].lower())


def structure(agency: str, flow: str) -> list[dict]:
    """The dataset's dimensions in key order: [{'id', 'name', 'codes': {code: name}}]."""
    return parse_structure(_get(f"dataflow/{agency}/{flow}/latest", JSON, {"references": "all"}).json()["data"])


def available(agency: str, flow: str, key: str) -> dict:
    """Which codes have data given a (partial) key; see sdmx_data.parse_availability."""
    data = _get(f"availableconstraint/{agency},{flow}/{key}", JSON, {"mode": "available"}).json()["data"]
    return parse_availability(data)


def get_data(agency: str, flow: str, key: str, dims: list[dict],
             start_year: int | None = None, end_year: int | None = None):
    """Observations for a key as a table; see sdmx_data.csv_to_frame."""
    params = {}
    if start_year:
        params["startPeriod"] = str(start_year)
    if end_year:
        params["endPeriod"] = str(end_year)
    return csv_to_frame(_get(f"data/{agency},{flow}/{key}", CSV, params, timeout=180).text, dims)
