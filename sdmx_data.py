"""
Shared helpers for SDMX 2.1 data APIs (used by the IMF and OECD clients): reading a dataset's
dimensions and code lists, availability constraints, CSV data and period strings.
"""

import io
import re

import pandas as pd


def parse_structure(data: dict) -> list[dict]:
    """A dataflow's dimensions in key order, from a `references=all` structure message:
    [{'id', 'name', 'codes': {code: name}}]. Each dimension's code list comes from its own
    representation or, failing that, its concept's."""
    concepts = {}
    for cs in data.get("conceptSchemes", []):
        for c in cs.get("concepts", []):
            concepts[f"urn:sdmx:org.sdmx.infomodel.conceptscheme.Concept={cs['agencyID']}:{cs['id']}({cs['version']}).{c['id']}"] = c
    codelists = {f"{cl['agencyID']}:{cl['id']}({cl['version']})": cl for cl in data.get("codelists", [])}
    dsd = data["dataStructures"][0]["dataStructureComponents"]["dimensionList"]["dimensions"]
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


def parse_availability(data: dict) -> dict:
    """Which codes have data, from an availableconstraint message: {'values': {dim: [codes]},
    'series': n or None, 'observations': n or None, 'start': 'YYYY-MM-DD', 'end': 'YYYY-MM-DD'}.
    Providers report counts and the time span differently (annotations or a TIME_PERIOD range)."""
    cc = (data.get("contentConstraints") or [{}])[0]
    values, start, end = {}, None, None
    for region in cc.get("cubeRegions", []):
        for kv in region.get("keyValues", []):
            if kv.get("timeRange"):
                start = ((kv["timeRange"].get("startPeriod") or {}).get("period") or "")[:10] or None
                end = ((kv["timeRange"].get("endPeriod") or {}).get("period") or "")[:10] or None
            if kv["id"] != "TIME_PERIOD":
                values[kv["id"]] = [v if isinstance(v, str) else v.get("value") for v in kv.get("values", [])]
    notes = {a.get("id"): a.get("title") for a in cc.get("annotations", [])}
    count = lambda k: int(notes[k]) if str(notes.get(k) or "").isdigit() else None
    return {"values": values, "series": count("series_count"), "observations": count("obs_count"),
            "start": start or notes.get("time_period_start"), "end": end or notes.get("time_period_end")}


def key_string(dims: list[dict], selections: dict[str, list[str]]) -> str:
    """SDMX key for the selections, e.g. 'USA+CAN..._T.IX.M'; an empty dimension means 'all'."""
    parts = ["+".join(selections.get(d["id"]) or []) for d in dims]
    return ".".join(parts) if any(parts) else "all"


_PERIOD = [
    (re.compile(r"^(\d{4})-M(\d{2})$"), lambda y, m: pd.Timestamp(int(y), int(m), 1)),
    (re.compile(r"^(\d{4})-Q([1-4])$"), lambda y, q: pd.Timestamp(int(y), 3 * (int(q) - 1) + 1, 1)),
    (re.compile(r"^(\d{4})-S([12])$"), lambda y, s: pd.Timestamp(int(y), 6 * (int(s) - 1) + 1, 1)),
    (re.compile(r"^(\d{4})-(\d{2})-(\d{2})$"), lambda y, m, d: pd.Timestamp(int(y), int(m), int(d))),
    (re.compile(r"^(\d{4})-(\d{2})$"), lambda y, m: pd.Timestamp(int(y), int(m), 1)),
    (re.compile(r"^(\d{4})(-A1?)?$"), lambda y, _=None: pd.Timestamp(int(y), 1, 1)),
]


def parse_period(value) -> pd.Timestamp:
    """'2025', '2025-A1', '2025-Q1', '2025-M01', '2025-01', '2025-S1', '2025-01-15' -> period
    start (NaT if unknown)."""
    v = str(value).strip()
    for pattern, build in _PERIOD:
        m = pattern.match(v)
        if m:
            return build(*m.groups())
    return pd.NaT


def csv_to_frame(text: str, dims: list[dict]) -> pd.DataFrame:
    """An SDMX-CSV data message as the dimension columns, TIME_PERIOD, OBS_VALUE (numeric), any
    unit/scale attributes, and a parsed `Date`. Attribute-only rows (no time period) and
    observations without a value are dropped."""
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
    keep += [c for c in ("UNIT", "SCALE", "UNIT_MULT") if c in df.columns and df[c].notna().any()]
    return df[keep].sort_values("Date").reset_index(drop=True)


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


def clean_name(name: str) -> str:
    """Tidy a dataset name: some catalogs carry U+FFFD where a dash or space was."""
    return re.sub(r"\s+", " ", name.replace("�", " ")).strip()
