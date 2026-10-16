"""
Client for the IMF's SDMX 2.1 data API (https://data.imf.org/en/Resource-Pages/IMF-API).
Public datasets need no key. Series are identified by dataflow and SDMX key, e.g. CPI with
key USA.CPI._T.IX.M (country . index type . COICOP category . transformation . frequency).
"""

import io
import re

import pandas as pd
import requests

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
            "name": re.sub(r"\s+", " ", (f.get("name") or f["id"]).replace("�", " ")).strip(),
            "description": (f.get("description") or "").strip(), "vintage": vintage,
        })
    return sorted(out, key=lambda f: f["name"].lower())


def structure(agency: str, flow: str) -> list[dict]:
    """The dataset's dimensions in key order: [{'id', 'name', 'codes': {code: name}}]. Each
    dimension's code list comes from its own representation or, failing that, its concept's."""
    d = _get(f"dataflow/{agency}/{flow}/latest", JSON, {"references": "all"}).json()["data"]
    concepts = {}
    for cs in d.get("conceptSchemes", []):
        for c in cs.get("concepts", []):
            concepts[f"urn:sdmx:org.sdmx.infomodel.conceptscheme.Concept={cs['agencyID']}:{cs['id']}({cs['version']}).{c['id']}"] = c
    codelists = {f"{cl['agencyID']}:{cl['id']}({cl['version']})": cl for cl in d.get("codelists", [])}
    dsd = d["dataStructures"][0]["dataStructureComponents"]["dimensionList"]["dimensions"]
    dims = []
    for dim in sorted(dsd, key=lambda x: x.get("position", 0)):
        concept = concepts.get(dim.get("conceptIdentity", ""), {})
        enum = ((dim.get("localRepresentation") or {}).get("enumeration")
                or (concept.get("coreRepresentation") or {}).get("enumeration") or "")
        cl = codelists.get(enum.split("=")[-1], {})
        dims.append({
            "id": dim["id"], "name": concept.get("name") or dim["id"],
            "codes": {c["id"]: c.get("name") or c["id"] for c in cl.get("codes", [])},
        })
    return dims


def key_string(dims: list[dict], selections: dict[str, list[str]]) -> str:
    """SDMX key for the selections, e.g. 'USA+CAN..._T.IX.M'; an empty dimension means 'all'."""
    parts = ["+".join(selections.get(d["id"]) or []) for d in dims]
    return ".".join(parts) if any(parts) else "all"


def available(agency: str, flow: str, key: str) -> dict:
    """Which codes have data given a (partial) key: {'values': {dim: [codes]}, 'series': n,
    'start': 'YYYY-MM-DD', 'end': 'YYYY-MM-DD'}."""
    d = _get(f"availableconstraint/{agency},{flow}/{key}", JSON, {"mode": "available"}).json()["data"]
    cc = (d.get("contentConstraints") or [{}])[0]
    values = {}
    for region in cc.get("cubeRegions", []):
        for kv in region.get("keyValues", []):
            values[kv["id"]] = [v if isinstance(v, str) else v.get("value") for v in kv.get("values", [])]
    notes = {a.get("id"): a.get("title") for a in cc.get("annotations", [])}
    return {"values": values, "series": int(notes.get("series_count") or 0),
            "start": notes.get("time_period_start"), "end": notes.get("time_period_end")}


_PERIOD = [
    (re.compile(r"^(\d{4})-M(\d{2})$"), lambda y, m: pd.Timestamp(int(y), int(m), 1)),
    (re.compile(r"^(\d{4})-Q([1-4])$"), lambda y, q: pd.Timestamp(int(y), 3 * (int(q) - 1) + 1, 1)),
    (re.compile(r"^(\d{4})-S([12])$"), lambda y, s: pd.Timestamp(int(y), 6 * (int(s) - 1) + 1, 1)),
    (re.compile(r"^(\d{4})-(\d{2})-(\d{2})$"), lambda y, m, d: pd.Timestamp(int(y), int(m), int(d))),
    (re.compile(r"^(\d{4})-(\d{2})$"), lambda y, m: pd.Timestamp(int(y), int(m), 1)),
    (re.compile(r"^(\d{4})(-A)?$"), lambda y, _=None: pd.Timestamp(int(y), 1, 1)),
]


def parse_period(value) -> pd.Timestamp:
    """'2025', '2025-Q1', '2025-M01', '2025-S1', '2025-01-15' -> period start (NaT if unknown)."""
    v = str(value).strip()
    for pattern, build in _PERIOD:
        m = pattern.match(v)
        if m:
            return build(*m.groups())
    return pd.NaT


def get_data(agency: str, flow: str, key: str, dims: list[dict],
             start_year: int | None = None, end_year: int | None = None) -> pd.DataFrame:
    """Observations for a key: the dimension columns, TIME_PERIOD, OBS_VALUE (numeric), any
    UNIT/SCALE attributes, and a parsed `Date`. Attribute-only rows (no time period) and
    observations without a value are dropped."""
    params = {}
    if start_year:
        params["startPeriod"] = str(start_year)
    if end_year:
        params["endPeriod"] = str(end_year)
    text = _get(f"data/{agency},{flow}/{key}", CSV, params, timeout=180).text
    if not text.strip():
        return pd.DataFrame()
    df = pd.read_csv(io.StringIO(text), dtype=str)
    if "TIME_PERIOD" not in df.columns:
        return pd.DataFrame()
    df = df[df["TIME_PERIOD"].notna()].copy()
    df["OBS_VALUE"] = pd.to_numeric(df["OBS_VALUE"], errors="coerce")
    df = df[df["OBS_VALUE"].notna()]
    df["Date"] = pd.to_datetime(df["TIME_PERIOD"].map(parse_period))
    keep = [d["id"] for d in dims if d["id"] in df.columns] + ["TIME_PERIOD", "Date", "OBS_VALUE"]
    keep += [c for c in ("UNIT", "SCALE") if c in df.columns and df[c].notna().any()]
    return df[keep].reset_index(drop=True)


def series_key(row: pd.Series, dims: list[dict]) -> str:
    return ".".join(str(row[d["id"]]) for d in dims)


def series_label(row: pd.Series, dims: list[dict], varying: list[str]) -> str:
    """e.g. 'United States · All Items · Index' from the code names of the dimensions that differ
    between the series in a result (all dimensions if none differ)."""
    shown = [d for d in dims if d["id"] in varying] or dims
    return " · ".join(d["codes"].get(str(row[d["id"]]), str(row[d["id"]])) for d in shown)


def series_values(df: pd.DataFrame, filters: dict[str, str]) -> pd.Series:
    mask = pd.Series(True, index=df.index)
    for col, val in filters.items():
        mask &= df[col].astype(str) == str(val)
    rows = df[mask & df["Date"].notna()]
    s = pd.Series(rows["OBS_VALUE"].to_numpy(dtype=float), index=pd.DatetimeIndex(rows["Date"]))
    return s[~s.index.duplicated()].sort_index()
