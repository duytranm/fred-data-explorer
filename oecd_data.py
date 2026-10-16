"""
Client for the OECD Data Explorer SDMX API
(https://www.oecd.org/en/data/insights/data-explainers/2024/09/api.html). Free, no key, but
limited to 60 data downloads an hour per IP address, so callers should cache results.
Series are identified by dataflow and SDMX key, e.g. composite leading indicators
(DSD_STES@DF_CLI) with key USA.M.LI...AA...H.
"""

import requests

from sdmx_data import (  # noqa: F401  (re-exported for callers)
    clean_name, csv_to_frame, key_string, parse_availability, parse_period, parse_structure,
    series_key, series_label, series_values,
)

BASE_URL = "https://sdmx.oecd.org/public/rest"
OECD_PREFIX = "OECD:"
JSON = {"Accept": "application/vnd.sdmx.structure+json;version=1.0;charset=utf-8"}
CSV = {"Accept": "application/vnd.sdmx.data+csv; charset=utf-8"}


class OECDError(Exception):
    pass


def is_oecd_id(sid: str) -> bool:
    return sid.startswith(OECD_PREFIX)


def make_id(agency: str, flow: str, key: str) -> str:
    return f"{OECD_PREFIX}{agency}:{flow}:{key}"


def parse_id(sid: str) -> tuple[str, str, str]:
    """'OECD:OECD.SDD.STES:DSD_STES@DF_CLI:USA.M.LI...AA...H' -> (agency, dataflow, key)."""
    parts = sid[len(OECD_PREFIX):].split(":", 2)
    if len(parts) != 3 or not all(parts):
        raise OECDError(f"Not an OECD series ID (expected OECD:<agency>:<dataflow>:<key>): {sid}")
    return parts[0], parts[1], parts[2]


def _get(path: str, headers: dict, params: dict | None = None, timeout: int = 180) -> requests.Response:
    resp = requests.get(f"{BASE_URL}/{path}", headers=headers, params=params, timeout=timeout)
    if resp.status_code == 429:
        raise OECDError("The OECD API's limit of 60 data downloads an hour has been reached. Try again later.")
    if resp.status_code == 404:
        raise OECDError("The OECD API found no data for this selection.")
    if resp.status_code >= 400:
        detail = resp.text.strip()[:300] if resp.text.strip() and not resp.text.lstrip().startswith("<") else resp.reason
        raise OECDError(f"The OECD API rejected the request (HTTP {resp.status_code}): {detail or 'no details given'}")
    return resp


def dataflows() -> list[dict]:
    """Available datasets: {'agency', 'id', 'version', 'name', 'description'}, by name."""
    flows = _get("dataflow/all", JSON).json()["data"]["dataflows"]
    out = [{
        "agency": f["agencyID"], "id": f["id"], "version": f["version"],
        "name": clean_name(f.get("name") or f["id"]), "description": (f.get("description") or "").strip(),
    } for f in flows]
    return sorted(out, key=lambda f: f["name"].lower())


def structure(agency: str, flow: str) -> list[dict]:
    """The dataset's dimensions in key order: [{'id', 'name', 'codes': {code: name}}]."""
    return parse_structure(_get(f"dataflow/{agency}/{flow}/latest", JSON, {"references": "all"}).json()["data"])


def available(agency: str, flow: str, key: str) -> dict:
    """Which codes have data given a (partial) key; see sdmx_data.parse_availability (the OECD
    reports an observation count rather than a series count)."""
    data = _get(f"availableconstraint/{agency},{flow},/{key}", JSON, {"mode": "available"}).json()["data"]
    return parse_availability(data)


def get_data(agency: str, flow: str, key: str, dims: list[dict],
             start_year: int | None = None, end_year: int | None = None):
    """Observations for a key as a table (counts toward the hourly download limit)."""
    params = {}
    if start_year:
        params["startPeriod"] = str(start_year)
    if end_year:
        params["endPeriod"] = str(end_year)
    return csv_to_frame(_get(f"data/{agency},{flow},/{key}", CSV, params).text, dims)
