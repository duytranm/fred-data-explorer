"""
Client for the U.S. Treasury Fiscal Data API (https://fiscaldata.treasury.gov/api-documentation/).
Public, no key needed. Datasets, tables and fields come from Fiscal Data's metadata service.
"""

import pandas as pd
import requests

BASE_URL = "https://api.fiscaldata.treasury.gov/services/api/fiscal_service"
METADATA_URL = "https://api.fiscaldata.treasury.gov/services/dtg/metadata/"
TREASURY_PREFIX = "TREASURY:"
PAGE_SIZE = 10000  # the API's maximum
MAX_ROWS = 100_000  # some tables hold millions of rows; cap what one request pulls in

DATE_PART_TYPES = {"YEAR", "QUARTER", "MONTH", "DAY"}


class TreasuryError(Exception):
    pass


def is_treasury_id(sid: str) -> bool:
    return sid.startswith(TREASURY_PREFIX)


def is_value_type(data_type: str) -> bool:
    return data_type.startswith(("CURRENCY", "PERCENTAGE")) or data_type in ("NUMBER", "INTEGER")


def _get(endpoint: str, params: dict) -> dict:
    resp = requests.get(BASE_URL + endpoint, params=params, timeout=90)
    if resp.status_code >= 400:
        try:
            body = resp.json()
            detail = body.get("message") or body.get("error")
        except ValueError:
            detail = resp.reason
        raise TreasuryError(f"Fiscal Data rejected the request (HTTP {resp.status_code}): {detail}")
    return resp.json()


def catalog() -> list[dict]:
    """Datasets, most popular first, each with its tables:
    {'title', 'description', 'tables': [{'name', 'endpoint', 'description', 'frequency',
    'earliest', 'latest', 'rows', 'date_field', 'fields': [{'name', 'type', 'label'}]}]}."""
    resp = requests.get(METADATA_URL, timeout=90)
    resp.raise_for_status()
    datasets = []
    for d in sorted(resp.json(), key=lambda d: int(d.get("popularity_rank") or 9999)):
        tables = []
        for a in sorted(d.get("apis", []), key=lambda a: int(a.get("api_sort_order_nbr") or 0)):
            fields = [{"name": f["column_name"], "type": f.get("data_type") or "STRING",
                       "label": f.get("pretty_name") or f["column_name"]} for f in a.get("fields") or []]
            if not fields:
                continue
            date_field = "record_date" if any(f["name"] == "record_date" for f in fields) else next(
                (f["name"] for f in fields if f["type"] == "DATE"), None)
            tables.append({
                "name": a.get("table_name", "").strip(),
                "endpoint": a["endpoint_txt"].replace("/services/api/fiscal_service", "", 1),
                "description": (a.get("table_description") or "").strip(),
                "frequency": a.get("update_frequency") or "",
                "earliest": a.get("earliest_date"), "latest": a.get("latest_date"),
                "rows": int(a.get("row_count") or 0), "date_field": date_field, "fields": fields,
            })
        if tables:
            datasets.append({"title": d["title"].strip(), "description": (d.get("short_description") or "").strip(),
                             "tables": tables})
    return datasets


def _date_filter(date_field: str | None, start: str | None, end: str | None) -> list[str]:
    if not date_field:
        return []
    return ([f"{date_field}:gte:{start}"] if start else []) + ([f"{date_field}:lte:{end}"] if end else [])


def distinct_values(endpoint: str, field: str) -> list[str]:
    """Every value of one field (requesting a single field returns its distinct values)."""
    payload = _get(endpoint, {"fields": field, "page[size]": PAGE_SIZE})
    return sorted({str(r[field]) for r in payload["data"] if r.get(field) not in (None, "null")})


def _fetch_pages(endpoint: str, filters: list[str], limit: int) -> tuple[list[dict], int]:
    rows: list[dict] = []
    total = 0
    page = 1
    while True:
        params = {"page[size]": PAGE_SIZE, "page[number]": page}
        if filters:
            params["filter"] = ",".join(filters)
        payload = _get(endpoint, params)
        rows += payload["data"]
        total = int(payload["meta"].get("total-count") or len(rows))
        if page >= int(payload["meta"].get("total-pages") or 1) or len(rows) >= limit:
            return rows[:limit], total
        page += 1


def get_data(endpoint: str, fields: list[dict], date_field: str | None, start: str | None, end: str | None,
             filter_field: str | None = None, filter_values: tuple = ()) -> tuple[pd.DataFrame, int]:
    """Every field of the matching rows (asking for all fields avoids the API's automatic
    summing over omitted ones), typed per the metadata. Returns (table, total matching rows),
    the table capped at MAX_ROWS."""
    base = _date_filter(date_field, start, end)
    if filter_field and filter_values:
        if any(ch in v for v in filter_values for ch in "(),"):
            # `in:(...)` can't express values containing parentheses or commas (it silently
            # matches only the first), so ask for each value separately with `eq`.
            filter_sets = [base + [f"{filter_field}:eq:{v}"] for v in filter_values]
        else:
            filter_sets = [base + [f"{filter_field}:in:({','.join(filter_values)})"]]
    else:
        filter_sets = [base]
    rows, total = [], 0
    for filters in filter_sets:
        got, n = _fetch_pages(endpoint, filters, MAX_ROWS - len(rows))
        rows += got
        total += n
        if len(rows) >= MAX_ROWS:
            break
    df = pd.DataFrame(rows)
    types = {f["name"]: f["type"] for f in fields}
    for col in df.columns:
        t = types.get(col, "STRING")
        if is_value_type(t):
            df[col] = pd.to_numeric(df[col].replace("null", None), errors="coerce")
        elif t == "DATE":
            df[col] = pd.to_datetime(df[col].replace("null", None), errors="coerce")
    if date_field and date_field in df.columns:
        df = df.sort_values(date_field).reset_index(drop=True)
    return df, total


IDENTIFIER_SUFFIXES = ("_nbr", "_id", "_cd")


def series_key_cols(df: pd.DataFrame, fields: list[dict]) -> list[str]:
    """Text fields that vary across rows and so tell series apart. Date parts (e.g. fiscal year)
    vary over time rather than between series, and identifier codes (`..._id`, `..._cd`,
    `..._nbr`) can change with every publication (MTS assigns new classification IDs each
    month), so both are left out."""
    types = {f["name"]: f["type"] for f in fields}
    return [
        c for c in df.columns
        if types.get(c, "STRING") == "STRING" and not c.endswith(IDENTIFIER_SUFFIXES)
        and df[c].nunique(dropna=False) > 1
    ]


def value_cols(df: pd.DataFrame, fields: list[dict]) -> list[str]:
    """Numeric fields worth charting (not line numbers or other identifiers)."""
    types = {f["name"]: f["type"] for f in fields}
    return [
        c for c in df.columns
        if is_value_type(types.get(c, "STRING")) and not c.endswith(IDENTIFIER_SUFFIXES)
        and df[c].notna().any()
    ]


def series_values(df: pd.DataFrame, filters: dict[str, str], value_col: str, date_field: str) -> pd.Series:
    mask = pd.Series(True, index=df.index)
    for col, val in filters.items():
        mask &= df[col].astype(str) == str(val)
    rows = df[mask & df[date_field].notna()]
    s = pd.Series(rows[value_col].to_numpy(dtype=float), index=pd.DatetimeIndex(rows[date_field]))
    return s.groupby(level=0).sum(min_count=1).sort_index()
