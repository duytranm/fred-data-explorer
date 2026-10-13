"""
Helpers for the Bureau of Economic Analysis API, via BEA's own `beaapi` package
(https://github.com/us-bea/beaapi). Needs a free key: https://apps.bea.gov/API/signup/
"""

import os
import re

import beaapi
import pandas as pd

KEYRING_SERVICE = "bea_api"
KEYRING_USERNAME = "api_key"
BEA_PREFIX = "BEA:"

# beaapi.get_parameter_values_filtered isn't supported for these datasets.
NO_FILTERED_VALUES = {"NIPA", "NIUnderlyingDetail", "MNE", "FixedAssets", "APIDatasetMetaData"}

# GeoFips keywords the Regional dataset accepts but doesn't list among its values.
REGIONAL_GEO_KEYWORDS = {
    "STATE": "All states",
    "COUNTY": "All counties",
    "MSA": "All metropolitan statistical areas",
}

# Columns that hold the measured value or describe time rather than identify a series.
NON_SERIES_COLS = {"DataValue", "TimePeriod", "Year", "Quarter", "Frequency", "NoteRef", "NoteRefs"}

BEAError = (beaapi.BEAAPIError, beaapi.BEAAPIFailure, beaapi.BEAAPIPkgException, beaapi.BEAAPIResponseError)


def get_bea_key() -> str | None:
    """Look for the key in, in order: Streamlit secrets, the BEA_API_KEY env var, then the
    local OS credential store (keyring), skipping any lookup this environment can't do."""
    try:
        import streamlit as st
        key = st.secrets.get("BEA_API_KEY")
        if key:
            return key
    except Exception:
        pass
    if os.environ.get("BEA_API_KEY"):
        return os.environ["BEA_API_KEY"]
    try:
        import keyring
        return keyring.get_password(KEYRING_SERVICE, KEYRING_USERNAME)
    except Exception:
        return None


def is_bea_id(sid: str) -> bool:
    return sid.startswith(BEA_PREFIX)


def dataset_list(key: str) -> pd.DataFrame:
    df = beaapi.get_data_set_list(key)
    return df[df["DatasetName"] != "APIDatasetMetaData"].reset_index(drop=True)


def parameter_list(key: str, dataset: str) -> pd.DataFrame:
    return beaapi.get_parameter_list(key, dataset)


def parameter_values(key: str, dataset: str, param: str, filters: dict | None = None) -> dict[str, str]:
    """Allowed values for one parameter -> their descriptions. Where BEA supports it, the list
    is narrowed by parameters already chosen (`filters`), e.g. Regional lines for one table."""
    df = None
    if filters and dataset not in NO_FILTERED_VALUES:
        try:
            df = beaapi.get_parameter_values_filtered(key, dataset, param, **filters)
        except (*BEAError, KeyError):  # beaapi raises KeyError on some datasets' filtered responses
            df = None
    if df is None:
        df = beaapi.get_parameter_values(key, dataset, param)
    if df.empty:
        return {}
    key_col = next((c for c in ("Key", f"{param}ID", param) if c in df.columns), df.columns[0])
    desc_col = next((c for c in ("Desc", "Description") if c in df.columns), None)
    if desc_col is None:
        desc_col = next((c for c in df.columns if c != key_col), key_col)
    return dict(zip(df[key_col].astype(str), df[desc_col].astype(str)))


def get_table(key: str, dataset: str, params: dict) -> tuple[pd.DataFrame, list[str], str]:
    """A BEA data request with a parsed `Date` column (start of each period), the columns that
    identify each series in it, and the response's statistic name (e.g. 'Per capita personal
    income'; '' if BEA doesn't give one)."""
    df = beaapi.get_data(key, dataset, **params)
    keys = series_key_cols(df)
    statistic = str((df.attrs.get("detail") or {}).get("Statistic") or "").strip()
    df = df.copy()
    df["Date"] = period_dates(df)
    return df, keys, statistic


_PERIOD_PATTERNS = [
    (re.compile(r"^(\d{4})Q([1-4])$"), lambda y, q: pd.Timestamp(int(y), 3 * (int(q) - 1) + 1, 1)),
    (re.compile(r"^(\d{4})M(\d{2})$"), lambda y, m: pd.Timestamp(int(y), int(m), 1)),
    (re.compile(r"^(\d{4})A?$"), lambda y: pd.Timestamp(int(y), 1, 1)),
]
_ROMAN_QUARTERS = {"QI": 1, "QII": 2, "QIII": 3, "QIV": 4}


def parse_period(period) -> pd.Timestamp:
    """'2015Q1', '2015M01', '2015' or '2015A' -> the period's first day (NaT if unrecognised)."""
    p = str(period).strip()
    for pattern, build in _PERIOD_PATTERNS:
        m = pattern.match(p)
        if m:
            return build(*m.groups())
    return pd.NaT


def period_dates(df: pd.DataFrame) -> pd.Series:
    if "TimePeriod" in df.columns:
        return pd.to_datetime(df["TimePeriod"].map(parse_period))
    if "Year" in df.columns:
        # e.g. GDPbyIndustry: Year + Quarter (I..IV); IIP: Year + Frequency (A, QI..QIV)
        blank = pd.Series("", index=df.index)
        quarter = df["Quarter"].astype(str) if "Quarter" in df.columns else blank
        freq = df["Frequency"].astype(str) if "Frequency" in df.columns else blank
        dates = []
        for y, q, f in zip(df["Year"], quarter, freq):
            roman = next((r for r in (f"Q{q}", f) if r in _ROMAN_QUARTERS), None)
            dates.append(pd.Timestamp(int(y), 3 * (_ROMAN_QUARTERS[roman] - 1) + 1, 1) if roman else parse_period(y))
        return pd.to_datetime(pd.Series(dates, index=df.index))
    return pd.Series(pd.NaT, index=df.index)


def series_key_cols(df: pd.DataFrame) -> list[str]:
    """Columns that together identify one time series in a BEA table: beaapi's time-invariant
    keys when it reports them, else every non-value, non-time column."""
    keys = [c for c in df.attrs.get("time_invariant_keys", []) if c in df.columns and c not in NON_SERIES_COLS]
    if keys:
        return keys
    return [
        c for c in df.columns
        if c not in NON_SERIES_COLS and c != "Date"
        and (df[c].dtype == object or pd.api.types.is_string_dtype(df[c]))
    ]


def series_catalog(df: pd.DataFrame, keys: list[str], statistic: str = "") -> pd.DataFrame:
    """One row per distinct series in a BEA table (identified by `keys`, see series_key_cols):
    its key columns, a `unit` note, and a readable `label` built from the description fields,
    with key values appended only where needed to tell identically-described series apart.
    When rows only name a place (e.g. Regional's GeoName), `statistic` is prefixed so the
    label says what is measured."""
    cat = df.drop_duplicates(keys).reset_index(drop=True)
    desc_cols = [
        c for c in cat.columns
        if "Description" in c or c.endswith("Descr") or c in ("GeoName", "Desc", "Row", "Column", "SeriesName")
    ]
    varying = [c for c in desc_cols if cat[c].nunique() > 1]
    if varying:  # a description shared by every series (e.g. the table title) only adds length
        desc_cols = varying
    labels = (
        cat[desc_cols].astype(str).agg(" · ".join, axis=1) if desc_cols
        else pd.Series("", index=cat.index)
    )
    ambiguous = labels.duplicated(keep=False) | (labels == "")
    for col in [c for c in keys if c not in desc_cols]:
        if not ambiguous.any():
            break
        if cat.loc[ambiguous, col].nunique() < 2:
            continue
        labels[ambiguous] = (labels[ambiguous] + f" · {col} " + cat.loc[ambiguous, col].astype(str)).str.lstrip(" ·")
        ambiguous = labels.duplicated(keep=False)
    empty = labels == ""
    if empty.any():
        labels[empty] = cat.loc[empty, keys].astype(str).agg(" · ".join, axis=1)
    if statistic and set(desc_cols) <= {"GeoName"}:
        labels = statistic + " · " + labels
    cat["label"] = labels
    cat["unit"] = [unit_note(cat.iloc[[i]]) for i in range(len(cat))]
    return cat[keys + ["label", "unit"]]


def unit_note(rows: pd.DataFrame) -> str:
    """e.g. 'Current Dollars, Level, x10^6' from a series' METRIC_NAME/CL_UNIT/UNIT_MULT."""
    first = rows.iloc[0]
    bits = [str(first[c]) for c in ("METRIC_NAME", "CL_UNIT") if c in rows.columns and pd.notna(first[c])]
    if "UNIT_MULT" in rows.columns and pd.notna(first["UNIT_MULT"]) and str(first["UNIT_MULT"]) not in ("", "0"):
        bits.append(f"x10^{first['UNIT_MULT']}")
    return ", ".join(bits)


def series_values(df: pd.DataFrame, filters: dict[str, str]) -> pd.Series:
    """One series out of a BEA table: rows matching `filters` (key column -> value), as values
    indexed by period start date."""
    mask = pd.Series(True, index=df.index)
    for col, val in filters.items():
        mask &= df[col].astype(str) == str(val)
    rows = df[mask & df["Date"].notna()]
    s = pd.Series(pd.to_numeric(rows["DataValue"], errors="coerce").to_numpy(), index=pd.DatetimeIndex(rows["Date"]))
    return s[~s.index.duplicated()].sort_index()
