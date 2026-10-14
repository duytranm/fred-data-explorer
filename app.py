"""
Interactive FRED data explorer.

Run with:
    streamlit run app.py

Requires a FRED API key already stored via `python setup_api_key.py`
(or set as the FRED_API_KEY environment variable).
"""

import datetime
import io
import json
import re
import zlib
from urllib.parse import parse_qsl, urlencode

import numpy as np
import pandas as pd
import requests
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from fred_data import FredDatasets, get_api_key, get_recession_series, shade_recessions
import nyfed_data as nyfed
import bea_data as bea
import census_data as census

st.set_page_config(page_title="FREDDIE, BEAtrice & ADLER", layout="wide")

DEFAULT_PALETTE = px.colors.qualitative.Plotly  # cycled through when picking default line colors
IMAGE_FORMATS = {"PNG": "image/png", "JPEG": "image/jpeg", "SVG": "image/svg+xml", "PDF": "application/pdf"}
COLORSCALES = ["Viridis", "Plasma", "Turbo", "Cividis", "Blues", "Reds", "Greens", "Rainbow", "Bluered"]
MPL_FORMATS = {"PNG": "image/png", "PDF": "application/pdf", "SVG": "image/svg+xml"}

# A handful of well-known series to start from; users can add any FRED series ID.
DEFAULT_SERIES = {
    "UNRATE": "Unemployment Rate",
    "CPIAUCSL": "CPI (All Urban Consumers)",
    "GDP": "Real GDP",
    "FEDFUNDS": "Federal Funds Rate",
    "DGS10": "10-Year Treasury Yield",
    "PAYEMS": "Nonfarm Payrolls",
    "M2SL": "M2 Money Supply",
    "HOUST": "Housing Starts",
    "JTSJOR": "Job Openings Rate",
}

TRANSFORMS = ["Raw", "Year-over-year % change", "Period % change", "Normalized (=100 at start)"]

TIME_RANGE_OPTIONS = ["1Y", "5Y", "10Y", "20Y", "YTD", "Max"]

# Display name -> (plotly symbol, matplotlib marker), for manually-grouped scatter points
SHAPE_OPTIONS = {
    "Circle": ("circle", "o"),
    "Square": ("square", "s"),
    "Diamond": ("diamond", "D"),
    "Triangle": ("triangle-up", "^"),
    "Star": ("star", "*"),
    "Cross": ("x", "x"),
}

FREQ_MAP = {"None": None, "Monthly": "ME", "Quarterly": "QE", "Yearly": "YE"}

# X series, Y series, X transform, Y transform
RELATIONSHIP_PRESETS = {
    "Beveridge Curve (Unemployment vs Job Openings)": ("UNRATE", "JTSJOR", "Raw", "Raw"),
    "Phillips Curve (Unemployment vs Inflation)": ("UNRATE", "CPIAUCSL", "Raw", "Year-over-year % change"),
    "Custom": None,
}

# Operations available when combining two series into a derived series (e.g. V/E ratio = A / B)
DERIVED_OPS = {
    "A / B (ratio)": (lambda a, b: a / b, "/"),
    "A - B (difference)": (lambda a, b: a - b, "-"),
    "A + B (sum)": (lambda a, b: a + b, "+"),
    "A * B (product)": (lambda a, b: a * b, "*"),
    "A as % of B": (lambda a, b: a / b * 100, "% of"),
}

DERIVED_PRESETS = {
    "None": None,
    "V/E Ratio (Job Openings Rate / Unemployment Rate)": ("JTSJOR", "A / B (ratio)", "UNRATE", "V/E Ratio"),
}

# Transforms available for turning one existing series into a new named series.
# name -> (fn(series, param) -> series, param label or None if the op takes no param, default param)
TRANSFORM_OPS = {
    "% change": (lambda s, p: s.pct_change(periods=int(p)) * 100, "Periods", 1),
    "Difference": (lambda s, p: s.diff(int(p)), "Periods", 1),
    "Rolling average": (lambda s, p: s.rolling(int(p)).mean(), "Window (periods)", 12),
    "Rolling std deviation": (lambda s, p: s.rolling(int(p)).std(), "Window (periods)", 12),
    "Lag (shift forward)": (lambda s, p: s.shift(int(p)), "Periods", 1),
    "Cumulative sum": (lambda s, p: s.cumsum(), None, 0),
    "Natural log": (lambda s, p: np.log(s), None, 0),
    "Z-score": (lambda s, p: (s - s.mean()) / s.std(), None, 0),
    "Invert (1 / x)": (lambda s, p: 1 / s, None, 0),
}

st.title("FREDDIE, BEAtrice & ADLER")
st.caption(
    "FRED Data Insights Explorer, BEA Tabulated Relevant Information Concerning Economics & "
    "All Data-Loaded Economic Resources (aka FBA for Full-Batch Analytics)"
)

if not get_api_key():
    st.warning(
        "No FRED API key is configured, so FRED series won't load. Add `FRED_API_KEY` to the app's "
        "Streamlit secrets (get a free key at https://fred.stlouisfed.org/docs/api/api_key.html), or, "
        "if you're running this locally, run `python setup_api_key.py` in this folder once and reload. "
        "The **NY Fed Markets** tab and `NYFED:` series work without a key."
    )


def get_bea_series() -> dict:
    """BEA series added from the BEA tab this session: ID -> {dataset, params, filters, label}."""
    return st.session_state.setdefault("bea_series", {})


def get_census_series() -> dict:
    """Census series added from the Census tab this session: ID -> query spec and label."""
    return st.session_state.setdefault("census_series", {})


def base_label(sid: str) -> str:
    """Default name for a base series: the curated FRED label, a NY Fed, BEA or Census series
    label, or the ID."""
    if sid in get_bea_series():
        return get_bea_series()[sid]["label"]
    if sid in get_census_series():
        return get_census_series()[sid]["label"]
    return DEFAULT_SERIES.get(sid) or nyfed.nyfed_label(sid) or sid


def nyfed_beta() -> bool:
    return st.session_state.get("nyfed_beta", False)


@st.cache_data(show_spinner="Fetching from FRED...")
def fetch(series_ids: tuple[str, ...], start: str, end: str, _api_key: str) -> pd.DataFrame:
    fd = FredDatasets(api_key=_api_key)
    for sid in series_ids:
        fd.add_series(sid, base_label(sid), start=start, end=end)
    return fd.as_dataframe()


@st.cache_data(ttl=3600, show_spinner="Fetching from the NY Fed...")
def fetch_nyfed(sid: str, start: str, end: str, beta: bool) -> pd.Series:
    return nyfed.get_series(sid, start, end, beta)


@st.cache_data(ttl=3600, show_spinner="Fetching from the NY Fed...")
def fetch_nyfed_json(path: str, params: dict, beta: bool):
    return nyfed.get_json(path, params or None, beta)


@st.cache_data(ttl=86400, show_spinner=False)
def nyfed_pd_series(beta: bool) -> dict[str, str]:
    return nyfed.list_pd_series(beta)


# BEA metadata changes rarely; data at most a few times a day. The key is left out of the cache
# key (leading underscore): results are the same public data whichever valid key fetched them.
@st.cache_data(ttl=86400, show_spinner="Loading BEA datasets...")
def bea_datasets(_key: str) -> pd.DataFrame:
    return bea.dataset_list(_key)


@st.cache_data(ttl=86400, show_spinner="Loading BEA parameters...")
def bea_params(dataset: str, _key: str) -> pd.DataFrame:
    return bea.parameter_list(_key, dataset)


@st.cache_data(ttl=86400, show_spinner="Loading BEA parameter values...")
def bea_param_values(dataset: str, param: str, filters: tuple, _key: str) -> dict[str, str]:
    return bea.parameter_values(_key, dataset, param, dict(filters))


@st.cache_data(ttl=3600, show_spinner="Fetching from BEA...")
def bea_table(dataset: str, params: tuple, _key: str) -> tuple[pd.DataFrame, list[str], str]:
    return bea.get_table(_key, dataset, dict(params))


@st.cache_data(ttl=86400, show_spinner="Loading the Census dataset catalog...")
def census_catalog() -> pd.DataFrame:
    return census.catalog()


@st.cache_data(ttl=86400, show_spinner="Loading Census variables...")
def census_vars(path: str) -> dict:
    return census.variables(path)


@st.cache_data(ttl=86400, show_spinner=False)
def census_groups(path: str) -> list[dict]:
    return census.groups(path)


@st.cache_data(ttl=86400, show_spinner="Loading the table's variables...")
def census_group_vars(path: str, group: str) -> dict:
    return census.group_variables(path, group)


@st.cache_data(ttl=86400, show_spinner=False)
def census_geo(path: str) -> list[dict]:
    return census.geography(path)


@st.cache_data(ttl=86400, show_spinner=False)
def census_examples(path: str) -> dict:
    return census.examples(path)


@st.cache_data(ttl=3600, show_spinner="Fetching from the Census Bureau...")
def census_table(path: str, get_vars: tuple, predicates: tuple, numeric: tuple, geo: tuple, _key: str) -> pd.DataFrame:
    """A Census query result as a typed table (see census_data.get_data)."""
    return census.get_data(path, list(get_vars), dict(predicates), _key, set(numeric), set(geo))


def fetch_base(series_ids: tuple[str, ...], start: str, end: str) -> pd.DataFrame:
    """Base series side by side: FRED IDs from FRED, `NYFED:` IDs from the NY Fed Markets API,
    `BEA:` IDs from BEA, `CENSUS:` IDs from the Census Bureau."""
    fred_ids = tuple(
        s for s in series_ids if not (nyfed.is_nyfed_id(s) or bea.is_bea_id(s) or census.is_census_id(s))
    )
    frames = [fetch(fred_ids, start, end, get_api_key())] if fred_ids else []
    for sid in series_ids:
        if nyfed.is_nyfed_id(sid):
            try:
                frames.append(fetch_nyfed(sid, start, end, nyfed_beta()).rename(base_label(sid)))
            except (nyfed.NYFedError, requests.RequestException, KeyError) as e:
                raise ValueError(f"Couldn't fetch {sid} from the NY Fed: {e}") from e
        elif bea.is_bea_id(sid):
            spec = get_bea_series().get(sid)
            if spec is None:
                raise ValueError(f"{sid} isn't a BEA series added in this session. Add it from the BEA tab.")
            if not bea.get_bea_key():
                raise ValueError("A BEA API key is needed for BEA series. Add BEA_API_KEY to the app's secrets.")
            try:
                table, _, _ = bea_table(spec["dataset"], tuple(spec["params"].items()), bea.get_bea_key())
            except (*bea.BEAError, OSError) as e:
                raise ValueError(f"Couldn't fetch {sid} from BEA: {e}") from e
            s = bea.series_values(table, spec["filters"]).loc[start:end]
            frames.append(s.rename(base_label(sid)))
        elif census.is_census_id(sid):
            spec = get_census_series().get(sid)
            if spec is None:
                raise ValueError(f"{sid} isn't a Census series added in this session. Add it from the Census tab.")
            try:
                table = census_table(*spec["query"], census.get_census_key())
            except (census.CensusError, requests.RequestException) as e:
                raise ValueError(f"Couldn't fetch {sid} from the Census Bureau: {e}") from e
            s = census.series_values(table, spec["filters"], spec["value_col"]).loc[start:end]
            frames.append(s.rename(base_label(sid)))
    return pd.concat(frames, axis=1, sort=True) if frames else pd.DataFrame()


def slugify(name: str) -> str:
    """Turn a display name like 'V/E Ratio' into a stable, typeable ID like 'V_E_RATIO'."""
    slug = re.sub(r"[^A-Za-z0-9]+", "_", name.strip()).strip("_").upper()
    return slug or "SERIES"


def get_derived_series() -> dict:
    return st.session_state.setdefault("derived_series", {})


def auto_series_label(sid: str) -> str:
    """The default display name for a series ID — a derived series' name, or the base FRED label."""
    derived = get_derived_series()
    if sid in derived:
        return derived[sid]["display"]
    return base_label(sid)


def series_display_label(sid: str) -> str:
    """Human-readable name for a series ID: a user-set custom name (used in legends, axis
    titles, exports, etc.) if one was typed into that series' rename box, else the default
    derived/base FRED label."""
    custom = st.session_state.get(f"custom_label_{sid}")
    if custom and custom.strip():
        return custom.strip()
    return auto_series_label(sid)


_rendered_rename_sids: set[str] = set()  # reset every rerun (this is a fresh module-level exec each time)


def rename_control(sid: str):
    """A small text input letting the user override how `sid` is displayed everywhere
    (legends, axis titles, tables, exports). Leaving it blank keeps the default name.

    The same series can be picked in both tabs at once (e.g. UNRATE in Time Series and as
    the Relationship tab's X series); both tabs' code runs every rerun regardless of which
    is visible, and Streamlit forbids two widgets sharing a key. So this renders the actual
    input only the first time a given `sid` is requested per run, and a read-only caption
    everywhere else that series comes up."""
    if sid in _rendered_rename_sids:
        st.caption(f"Display name: **{series_display_label(sid)}** (rename it where it's first configured)")
        return
    _rendered_rename_sids.add(sid)
    st.text_input(
        f"Display name for {sid}",
        value=st.session_state.get(f"custom_label_{sid}", auto_series_label(sid)),
        key=f"custom_label_{sid}",
    )


def _series_dependencies(sid: str, derived: dict) -> list[str]:
    """The direct inputs a derived/transformed series is built from (empty for a base ID)."""
    if sid not in derived:
        return []
    d = derived[sid]
    return [d["a_id"], d["b_id"]] if d["kind"] == "combine" else [d["source_id"]]


def collect_base_ids(sid: str, derived: dict, stack: tuple[str, ...] = ()) -> set[str]:
    """All base FRED IDs a series transitively depends on, recursing through any chain of
    derived/transformed series. Raises on a circular reference (A depends on B depends on A)."""
    if sid in stack:
        raise ValueError("Circular reference: " + " -> ".join(stack + (sid,)))
    deps = _series_dependencies(sid, derived)
    if not deps:
        return {sid}
    result: set[str] = set()
    for dep in deps:
        result |= collect_base_ids(dep, derived, stack + (sid,))
    return result


def resolve_series(sid: str, base_df: pd.DataFrame, derived: dict, cache: dict[str, pd.Series]) -> pd.Series:
    """Compute the values for `sid` (a base FRED ID or a derived/transformed series), resolving
    and memoizing any chain of dependencies first — so a derived series can be built from
    another derived series, to any depth."""
    if sid in cache:
        return cache[sid]
    if sid in derived:
        d = derived[sid]
        if d["kind"] == "combine":
            a = resolve_series(d["a_id"], base_df, derived, cache)
            b = resolve_series(d["b_id"], base_df, derived, cache)
            op_fn, _ = DERIVED_OPS[d["op"]]
            result = op_fn(a, b)
        else:
            src = resolve_series(d["source_id"], base_df, derived, cache)
            fn, _, _ = TRANSFORM_OPS[d["op"]]
            result = fn(src, d["param"])
    else:
        result = base_df[base_label(sid)]  # base_df's own column key is never renamed
    cache[sid] = result
    return result


def fetch_with_derived(selected_ids: tuple[str, ...], start: str, end: str) -> pd.DataFrame:
    """Like `fetch`, but transparently resolves any derived/transformed-series IDs into their
    formula over the underlying base FRED series — including series built from other derived
    or transformed series, chained to any depth."""
    derived = get_derived_series()
    base_needed: set[str] = set()
    for sid in selected_ids:
        base_needed |= collect_base_ids(sid, derived)

    base_df = fetch_base(tuple(sorted(base_needed)), start, end)

    cache: dict[str, pd.Series] = {}
    out = pd.DataFrame(index=base_df.index)
    for sid in selected_ids:
        out[series_display_label(sid)] = resolve_series(sid, base_df, derived, cache)
    return out


def slice_by_date(df: pd.DataFrame, start, end) -> pd.DataFrame:
    """Rows of `df` (DatetimeIndex) between `start` and `end` (inclusive, plain dates);
    returns `df` unchanged if either bound is None."""
    if start is None or end is None:
        return df
    if df.empty:
        return df
    return df.loc[str(start):str(end)]


def contrast_text_color(hex_color: str) -> str:
    """Pick black or white text so it stays legible against `hex_color`."""
    hex_color = hex_color.lstrip("#")
    r, g, b = (int(hex_color[i : i + 2], 16) for i in (0, 2, 4))
    luminance = 0.299 * r + 0.587 * g + 0.114 * b
    return "#000000" if luminance > 140 else "#FFFFFF"


def compute_preset_range(preset: str) -> tuple[datetime.date, datetime.date]:
    today = datetime.date.today()
    if preset == "YTD":
        return datetime.date(today.year, 1, 1), today
    if preset == "Max":
        return datetime.date(1900, 1, 1), today  # earlier than any FRED series
    years = int(preset.rstrip("Y"))
    return today - datetime.timedelta(days=365 * years), today


def time_window_control(key_prefix: str, start_key: str | None, end_key: str | None, range_key: str | None = None):
    """A row of quick-range buttons (1Y/5Y/10Y/.../Max) that set the date control(s) below
    when clicked. Applies only on the click itself (not every rerun), so a manual edit
    afterward isn't clobbered.

    Pass `start_key`/`end_key` for two separate date_input widgets, or `range_key` for a
    single range-slider widget (whose value is a (start, end) tuple)."""
    quick_range = st.segmented_control(
        "Quick range", options=TIME_RANGE_OPTIONS, key=f"{key_prefix}_quick_range"
    )
    prev_key = f"{key_prefix}_quick_range_prev"
    if quick_range and quick_range != st.session_state.get(prev_key):
        preset_start, preset_end = compute_preset_range(quick_range)
        if range_key:
            st.session_state[range_key] = (preset_start, preset_end)
        else:
            st.session_state[start_key] = preset_start
            st.session_state[end_key] = preset_end
    st.session_state[prev_key] = quick_range


def synced_checkbox(label: str, source_value: bool, key: str) -> bool:
    """A checkbox that defaults to (and re-syncs with) `source_value` whenever it changes,
    but a manual change to the checkbox itself sticks until `source_value` changes again.
    Needed because a checkbox inside a collapsed expander is still instantiated every rerun,
    so a plain `value=source_value` would only apply once and then silently go stale."""
    synced_key = f"{key}_synced_from"
    if source_value != st.session_state.get(synced_key):
        st.session_state[key] = source_value
        st.session_state[synced_key] = source_value
    return st.checkbox(label, key=key)


def apply_transform(series: pd.Series, transform: str, periods: int) -> tuple[pd.Series, str]:
    if transform == "Year-over-year % change":
        return series.pct_change(periods=periods) * 100, "YoY % change"
    if transform == "Period % change":
        return series.pct_change() * 100, "% change"
    if transform == "Normalized (=100 at start)":
        clean = series.dropna()
        base = clean.iloc[0] if not clean.empty else series.iloc[0]
        return series.div(base) * 100, "Index (start = 100)"
    return series, "Value"


def export_chart_section(build_fig_fn, key_prefix: str, file_stem: str, date_bounds=None):
    """Render format/size controls and a download button that rasterizes a figure via kaleido.

    `build_fig_fn(start, end) -> go.Figure` builds the figure to export; `date_bounds`
    (min_date, max_date), if given, shows a slider to confine the exported chart to a
    sub-range without affecting the on-screen chart."""
    st.markdown("**Export chart as image**")
    export_start, export_end = date_bounds if date_bounds else (None, None)
    if date_bounds and date_bounds[0] < date_bounds[1]:
        export_start, export_end = st.slider(
            "Confine export to time range",
            min_value=date_bounds[0], max_value=date_bounds[1], value=date_bounds,
            key=f"{key_prefix}_range",
        )

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        fmt = st.selectbox("Format", list(IMAGE_FORMATS.keys()), key=f"{key_prefix}_fmt")
    with c2:
        width = st.number_input("Width (px)", min_value=200, max_value=4000, value=1200, step=100, key=f"{key_prefix}_w")
    with c3:
        height = st.number_input("Height (px)", min_value=200, max_value=4000, value=700, step=100, key=f"{key_prefix}_h")
    with c4:
        scale = st.number_input("Scale", min_value=1.0, max_value=5.0, value=2.0, step=0.5, key=f"{key_prefix}_scale")

    if st.button("Generate image", key=f"{key_prefix}_gen"):
        try:
            fig = build_fig_fn(export_start, export_end)
            img_bytes = fig.to_image(format=fmt.lower(), width=int(width), height=int(height), scale=scale)
            st.session_state[f"{key_prefix}_img"] = img_bytes
        except Exception as e:
            st.error(
                f"Couldn't render the image ({e}). Kaleido needs a local Chromium; "
                "try running `kaleido_get_chrome` once in this environment."
            )

    img_bytes = st.session_state.get(f"{key_prefix}_img")
    if img_bytes:
        st.download_button(
            f"Download {fmt}",
            img_bytes,
            file_name=f"{file_stem}.{fmt.lower()}",
            mime=IMAGE_FORMATS[fmt],
            key=f"{key_prefix}_dl",
        )


@st.cache_data(show_spinner="Fetching NBER recession dates...")
def fetch_recessions(_api_key: str) -> pd.Series:
    return get_recession_series(_api_key)


def pick_year_interval(start, end) -> int:
    """A tick every N years, scaled so long histories don't get an unreadable axis."""
    years = (end - start).days / 365.25
    if years > 60:
        return 20
    if years > 30:
        return 10
    if years > 15:
        return 5
    if years > 6:
        return 2
    return 1


def matplotlib_fig_to_bytes(fig, fmt: str) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format=fmt.lower(), dpi=200, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()


def build_ts_figure(df: pd.DataFrame, label_colors: dict, y_label: str, bg_color: str, text_color: str) -> go.Figure:
    """The interactive multi-line time series chart, shared by the on-screen chart and its
    (possibly date-confined) export."""
    if df.empty:
        raise ValueError("No data in the selected export range.")
    fig = go.Figure()
    for col in df.columns:
        s = df[col].dropna()  # series at different frequencies share one index; gaps would break their lines
        fig.add_trace(go.Scatter(x=s.index, y=s, mode="lines", name=col, line=dict(color=label_colors.get(col))))
    fig.update_layout(
        height=500,
        yaxis_title=y_label,
        xaxis_title="Date",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=20, t=40, b=40),
        plot_bgcolor=bg_color,
        paper_bgcolor=bg_color,
        font=dict(color=text_color),
    )
    return fig


def build_panels_figure(plot_df: pd.DataFrame, label_colors: dict, y_label: str, recession_series):
    """One subplot per series, each with a bold title colored to match its line — see the
    Unemployment/Core PCE sample chart."""
    if plot_df.empty:
        raise ValueError("No data in the selected export range.")
    n = len(plot_df.columns)
    fig, axes = plt.subplots(1, n, figsize=(4.5 * n, 3))
    axes = [axes] if n == 1 else list(axes)
    interval = pick_year_interval(plot_df.index.min(), plot_df.index.max())
    for ax, col in zip(axes, plot_df.columns):
        color = label_colors.get(col, "#00205B")
        series = plot_df[col].dropna()
        ax.plot(series.index, series.values, color=color, linewidth=1.5)
        ax.set_title(f"{col} ({y_label})", fontsize=9, fontweight="bold", color=color)
        ax.tick_params(labelsize=6)
        ax.xaxis.set_major_locator(mdates.YearLocator(interval))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.grid(alpha=0.3)
        if recession_series is not None:
            shade_recessions(ax, recession_series)
    fig.tight_layout()
    return fig


def build_overlay_figure(
    plot_df: pd.DataFrame, label_colors: dict, y_label: str, recession_series, secondary_cols: list | None = None
):
    """All series overlaid on one panel with a legend — see the Inflation Indicators sample
    chart. Series in `secondary_cols` are plotted against a right-hand y-axis (twinx), so
    curves with very different scales/units can still be overlaid meaningfully."""
    if plot_df.empty:
        raise ValueError("No data in the selected export range.")
    secondary_cols = [c for c in (secondary_cols or []) if c in plot_df.columns]
    fig, ax = plt.subplots(figsize=(9, 4))
    ax2 = ax.twinx() if secondary_cols else None
    interval = pick_year_interval(plot_df.index.min(), plot_df.index.max())

    lines = []
    for col in plot_df.columns:
        target = ax2 if (ax2 is not None and col in secondary_cols) else ax
        series = plot_df[col].dropna()
        (line,) = target.plot(series.index, series.values, color=label_colors.get(col), linewidth=1.5, label=col)
        lines.append(line)

    title = plot_df.columns[0] if len(plot_df.columns) == 1 else f"FRED Series ({y_label})"
    ax.set_title(title, fontsize=11, fontweight="bold")
    primary_cols = [c for c in plot_df.columns if c not in secondary_cols]
    ax.set_ylabel(", ".join(primary_cols) if len(primary_cols) <= 2 else y_label, fontsize=8)
    ax.tick_params(labelsize=7)
    ax.xaxis.set_major_locator(mdates.YearLocator(interval))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.grid(alpha=0.3)
    if ax2 is not None:
        ax2.set_ylabel(", ".join(secondary_cols) if len(secondary_cols) <= 2 else "Secondary axis", fontsize=8)
        ax2.tick_params(labelsize=7)
    if len(plot_df.columns) > 1:
        ax.legend(lines, [l.get_label() for l in lines], fontsize=7, loc="upper left")
    if recession_series is not None:
        shade_recessions(ax, recession_series)
    fig.tight_layout()
    return fig


def linear_fit(x, y) -> tuple[float, float] | None:
    """Ordinary-least-squares slope/intercept for a scatter of points, or None if there
    aren't at least 2 points with non-constant X (a line isn't meaningful otherwise)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = ~np.isnan(x) & ~np.isnan(y)
    x, y = x[mask], y[mask]
    if len(x) < 2 or np.ptp(x) == 0:
        return None
    slope, intercept = np.polyfit(x, y, 1)
    return float(slope), float(intercept)


def build_relationship_publication_fig(
    merged: pd.DataFrame, x_label: str, y_label_name: str, point_color: str,
    highlight_color: str, highlight_latest: bool, show_label: bool,
    point_groups: list | None = None, groups: dict | None = None, show_regression: bool = False,
    regression_groups: list | None = None,
):
    """Muted scatter with the latest point called out — see the Beveridge Curve sample chart.
    If `point_groups` (a group name or None per row of `merged`) and `groups` (name -> {color,
    shape}) are given, manually-assigned points are drawn with their group's color/marker and
    get a legend; everything else stays the plain muted scatter. If `show_regression`, an OLS
    fit line is drawn for each group in `regression_groups`; if `regression_groups` also
    contains `"Overall"` (or there are no groups at all), one additional fit line covering
    every point is drawn too."""
    if merged.empty:
        raise ValueError("No data in the selected export range.")
    fig, ax = plt.subplots(figsize=(6, 4.5))
    groups = groups or {}
    regression_groups = list(groups.keys()) if regression_groups is None else regression_groups
    has_legend = False
    if point_groups:
        ungrouped_mask = [g not in groups for g in point_groups]
        ax.scatter(
            merged["x"][ungrouped_mask], merged["y"][ungrouped_mask],
            color=point_color, alpha=0.6, s=25, edgecolor="none", zorder=2,
        )
        for gname, gcfg in groups.items():
            mask = [g == gname for g in point_groups]
            if any(mask):
                _, marker = SHAPE_OPTIONS.get(gcfg["shape"], ("circle", "o"))
                ax.scatter(
                    merged["x"][mask], merged["y"][mask],
                    color=gcfg["color"], marker=marker, s=40, edgecolor="none", zorder=3, label=gname,
                )
                if show_regression and gname in regression_groups:
                    fit = linear_fit(merged["x"][mask], merged["y"][mask])
                    if fit:
                        slope, intercept = fit
                        gx = merged["x"][mask]
                        xs = np.linspace(gx.min(), gx.max(), 50)
                        ax.plot(
                            xs, slope * xs + intercept, color=gcfg["color"], linestyle="--",
                            linewidth=1.5, zorder=3, label=f"{gname} fit (β={slope:.3g})",
                        )
        has_legend = True
        if show_regression and "Overall" in regression_groups:
            fit = linear_fit(merged["x"], merged["y"])
            if fit:
                slope, intercept = fit
                xs = np.linspace(merged["x"].min(), merged["x"].max(), 50)
                ax.plot(
                    xs, slope * xs + intercept, color="black", linestyle=":",
                    linewidth=1.5, zorder=3, label=f"Overall fit (β={slope:.3g})",
                )
    else:
        ax.scatter(merged["x"], merged["y"], color=point_color, alpha=0.6, s=25, edgecolor="none", zorder=2)
        if show_regression:
            fit = linear_fit(merged["x"], merged["y"])
            if fit:
                slope, intercept = fit
                xs = np.linspace(merged["x"].min(), merged["x"].max(), 50)
                ax.plot(
                    xs, slope * xs + intercept, color="black", linestyle="--",
                    linewidth=1.5, zorder=3, label=f"Fit (β={slope:.3g})",
                )
                has_legend = True
    if has_legend:
        ax.legend(fontsize=7, loc="best")
    if highlight_latest:
        lx, ly = merged["x"].iloc[-1], merged["y"].iloc[-1]
        ax.scatter([lx], [ly], color=highlight_color, s=45, zorder=4)
        if show_label:
            ax.annotate(
                merged.index[-1].strftime("%Y-%m"),
                (lx, ly),
                textcoords="offset points",
                xytext=(6, 6),
                fontsize=8,
                color=highlight_color,
            )
    ax.set_title(f"{x_label} vs. {y_label_name}", fontsize=11, fontweight="bold")
    ax.set_xlabel(x_label, fontsize=9)
    ax.set_ylabel(y_label_name, fontsize=9)
    ax.tick_params(labelsize=7)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def build_relationship_figure(
    merged: pd.DataFrame, groups: dict, assignments: dict, rel_line_color: str, rel_colorscale: str,
    rel_latest_color: str, x_label: str, y_label_name: str, x_axis_label: str, y_axis_label: str,
    bg_color: str, text_color: str, show_regression: bool = False, regression_groups: list | None = None,
) -> go.Figure:
    """The interactive X-vs-Y scatter (path line + grouped/gradient markers + latest-point
    star), shared by the on-screen chart and its (possibly date-confined) export. If
    `show_regression`, an OLS fit line is added for each group in `regression_groups`; if
    `regression_groups` also contains `"Overall"` (or there are no groups at all), one
    additional fit line covering every point is added too."""
    regression_groups = list(groups.keys()) if regression_groups is None else regression_groups
    if merged.empty:
        raise ValueError("No data in the selected export range.")
    point_groups = [assignments.get(ts.isoformat()) for ts in merged.index]
    year_frac = merged.index.year + (merged.index.month - 1) / 12

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=merged["x"], y=merged["y"], mode="lines",
            line=dict(width=1, color=rel_line_color), hoverinfo="skip", showlegend=False,
        )
    )
    if groups:
        marker_dict = dict(
            size=9,
            color=[groups[g]["color"] if g in groups else "#B0B0B0" for g in point_groups],
            symbol=[SHAPE_OPTIONS[groups[g]["shape"]][0] if g in groups else "circle" for g in point_groups],
            line=dict(width=0.5, color="white"),
        )
    else:
        marker_dict = dict(
            size=7, color=year_frac, colorscale=rel_colorscale, showscale=True, colorbar=dict(title="Year"),
        )
    fig.add_trace(
        go.Scatter(
            x=merged["x"], y=merged["y"], mode="markers",
            marker=marker_dict,
            text=[d.strftime("%Y-%m") for d in merged.index],
            hovertemplate="%{text}<br>X=%{x:.2f}<br>Y=%{y:.2f}<extra></extra>",
            showlegend=False,
        )
    )
    if show_regression:
        if groups:
            for gname, gcfg in groups.items():
                if gname not in regression_groups:
                    continue
                mask = [g == gname for g in point_groups]
                if sum(mask) >= 2:
                    fit = linear_fit(merged["x"][mask], merged["y"][mask])
                    if fit:
                        slope, intercept = fit
                        gx = merged["x"][mask]
                        xs = np.linspace(gx.min(), gx.max(), 50)
                        fig.add_trace(
                            go.Scatter(
                                x=xs, y=slope * xs + intercept, mode="lines",
                                line=dict(color=gcfg["color"], dash="dash", width=2),
                                name=f"{gname} fit", showlegend=True,
                                hovertemplate=f"{gname} fit: y = {slope:.3g}x + {intercept:.3g}<extra></extra>",
                            )
                        )
            if "Overall" in regression_groups:
                fit = linear_fit(merged["x"], merged["y"])
                if fit:
                    slope, intercept = fit
                    xs = np.linspace(merged["x"].min(), merged["x"].max(), 50)
                    fig.add_trace(
                        go.Scatter(
                            x=xs, y=slope * xs + intercept, mode="lines",
                            line=dict(color=text_color, dash="dot", width=2),
                            name="Overall fit", showlegend=True,
                            hovertemplate=f"Overall fit: y = {slope:.3g}x + {intercept:.3g}<extra></extra>",
                        )
                    )
        else:
            fit = linear_fit(merged["x"], merged["y"])
            if fit:
                slope, intercept = fit
                xs = np.linspace(merged["x"].min(), merged["x"].max(), 50)
                fig.add_trace(
                    go.Scatter(
                        x=xs, y=slope * xs + intercept, mode="lines",
                        line=dict(color=text_color, dash="dash", width=2),
                        name="Fit", showlegend=True,
                        hovertemplate=f"Fit: y = {slope:.3g}x + {intercept:.3g}<extra></extra>",
                    )
                )
    fig.add_trace(
        go.Scatter(
            x=[merged["x"].iloc[-1]], y=[merged["y"].iloc[-1]],
            mode="markers", marker=dict(size=13, color=rel_latest_color, symbol="star"),
            name="Latest", showlegend=False,
            hovertemplate=f"Latest: {merged.index[-1].strftime('%Y-%m')}<extra></extra>",
        )
    )
    fig.update_layout(
        height=550,
        xaxis_title=f"{x_label} ({x_axis_label})",
        yaxis_title=f"{y_label_name} ({y_axis_label})",
        showlegend=show_regression,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=20, t=40, b=40),
        plot_bgcolor=bg_color,
        paper_bgcolor=bg_color,
        font=dict(color=text_color),
    )
    return fig


def publication_export_section(
    key_prefix: str, file_stem: str, build_fn, needs_recession: bool = False, date_bounds=None
):
    """Shared controls for the matplotlib 'publication style' export, used by both tabs.

    `build_fn` is called as `build_fn(recession_series, start, end)` if `needs_recession`,
    else `build_fn(start, end)`. `date_bounds` (min_date, max_date), if given, shows a
    slider to confine the exported chart to a sub-range without affecting the on-screen
    chart."""
    export_start, export_end = date_bounds if date_bounds else (None, None)
    if date_bounds and date_bounds[0] < date_bounds[1]:
        export_start, export_end = st.slider(
            "Confine export to time range",
            min_value=date_bounds[0], max_value=date_bounds[1], value=date_bounds,
            key=f"{key_prefix}_range",
        )

    recession_series = None
    if needs_recession:
        shade = st.checkbox("Shade NBER recessions", value=True, key=f"{key_prefix}_shade")
        if shade:
            try:
                recession_series = fetch_recessions(get_api_key())
            except Exception as e:
                st.warning(f"Couldn't load recession dates: {e}")

    fmt = st.selectbox("Format", list(MPL_FORMATS.keys()), key=f"{key_prefix}_fmt")

    if st.button("Generate image", key=f"{key_prefix}_gen"):
        try:
            fig = (
                build_fn(recession_series, export_start, export_end)
                if needs_recession
                else build_fn(export_start, export_end)
            )
            st.session_state[f"{key_prefix}_img"] = matplotlib_fig_to_bytes(fig, fmt)
        except Exception as e:
            st.error(f"Couldn't render the image: {e}")

    img_bytes = st.session_state.get(f"{key_prefix}_img")
    if img_bytes:
        if fmt == "PNG":
            st.image(img_bytes)
        st.download_button(
            f"Download {fmt}",
            img_bytes,
            file_name=f"{file_stem}.{fmt.lower()}",
            mime=MPL_FORMATS[fmt],
            key=f"{key_prefix}_dl",
        )


with st.sidebar:
    st.header("Derived series")
    with st.expander("Create a derived series"):
        st.caption(
            "Combine two series with a formula, e.g. **V/E Ratio** = Job Openings Rate / Unemployment Rate — "
            "a common labor-market tightness measure. Series A/B can be a FRED ID or the ID of another "
            "derived/transformed series below (chaining is fine, as long as it doesn't loop back on itself)."
        )
        derived_preset = st.selectbox("Quick preset", list(DERIVED_PRESETS.keys()), key="derived_preset")
        preset_vals = DERIVED_PRESETS[derived_preset]
        d_a_default, d_op_default, d_b_default, d_name_default = preset_vals or ("UNRATE", "A / B (ratio)", "CPIAUCSL", "")

        d_a_id = st.text_input("Series A ID", value=d_a_default, key=f"d_a_{derived_preset}").strip().upper()
        d_op = st.selectbox(
            "Operation", list(DERIVED_OPS.keys()), index=list(DERIVED_OPS.keys()).index(d_op_default), key=f"d_op_{derived_preset}"
        )
        d_b_id = st.text_input("Series B ID", value=d_b_default, key=f"d_b_{derived_preset}").strip().upper()
        d_name = st.text_input("Name for the new series", value=d_name_default, key=f"d_name_{derived_preset}")

        if st.button("Add derived series", key="add_derived"):
            if d_name.strip() and d_a_id and d_b_id:
                slug = slugify(d_name)
                get_derived_series()[slug] = {
                    "kind": "combine", "display": d_name.strip(), "a_id": d_a_id, "op": d_op, "b_id": d_b_id,
                }
                st.success(f"Added **{d_name.strip()}** — pick it from 'Pick series' below, or type `{slug}` as an X/Y series ID.")
            else:
                st.warning("Give it a name and both series IDs.")

    with st.expander("Create a transformed series"):
        st.caption(
            "Apply a transform to one existing series, e.g. **CPI YoY %** = 12-period % change "
            "of CPI. The source can be a FRED ID or the ID of another derived/transformed series "
            "(e.g. take a rolling average of a ratio you already built)."
        )
        t_source_id = st.text_input("Source series ID", value="CPIAUCSL", key="t_source_id").strip().upper()
        t_op = st.selectbox("Transform", list(TRANSFORM_OPS.keys()), key="t_op")
        _, t_param_label, t_default_param = TRANSFORM_OPS[t_op]
        if t_param_label:
            t_param = st.number_input(t_param_label, min_value=1, value=t_default_param, step=1, key="t_param")
        else:
            t_param = t_default_param
        t_name = st.text_input("Name for the new series", key="t_name")

        if st.button("Add transformed series", key="add_transform"):
            if t_name.strip() and t_source_id:
                slug = slugify(t_name)
                get_derived_series()[slug] = {
                    "kind": "transform", "display": t_name.strip(), "source_id": t_source_id,
                    "op": t_op, "param": t_param,
                }
                st.success(f"Added **{t_name.strip()}** — pick it from 'Pick series' below, or type `{slug}` as an X/Y series ID.")
            else:
                st.warning("Give it a name and a source series ID.")

    derived_defs = get_derived_series()
    if derived_defs:
        st.caption("Your derived/transformed series:")
        for slug, d in list(derived_defs.items()):
            c1, c2 = st.columns([5, 1])
            if d["kind"] == "combine":
                _, symbol = DERIVED_OPS[d["op"]]
                c1.write(f"**{d['display']}** (`{slug}`) = {d['a_id']} {symbol} {d['b_id']}")
            else:
                param_note = f" ({d['param']})" if TRANSFORM_OPS[d["op"]][1] else ""
                c1.write(f"**{d['display']}** (`{slug}`) = {d['op']}{param_note} of {d['source_id']}")
            if c2.button("✕", key=f"del_{slug}", help="Remove"):
                del derived_defs[slug]
                st.rerun()

NYFED_DATASETS = [
    "Reference rates (SOFR, EFFR, OBFR, TGCR, BGCR)",
    "SOFR averages and index",
    "Agency MBS operations",
    "Treasury securities operations",
    "Repo and reverse repo operations",
    "Reverse repo propositions",
    "Securities lending operations",
    "Central bank liquidity swaps",
    "SOMA holdings: weekly summary",
    "SOMA holdings: securities held as of a date",
    "Primary dealer statistics: one series",
    "Primary dealer statistics: latest survey",
    "Primary dealer market share",
    "Guide sheets",
    "Custom endpoint",
]


# Preferred default "Values" column when charting a NY Fed table, in priority order.
NYFED_HEADLINE_FIELDS = [
    "percentRate", "volumeInBillions", "totalAmtAccepted", "totalAmtAcceptedPar", "totalParAmtAccepted",
    "amtAccepted", "amount", "total", "parValue", "value",
]


def nyfed_date_range() -> dict | None:
    today = datetime.date.today()
    rng = st.date_input(
        "Date range", value=(today - datetime.timedelta(days=90), today),
        min_value=datetime.date(1990, 1, 1), max_value=today, key="nyfed_dates",
    )
    if len(rng) != 2:
        st.info("Pick an end date too.")
        return None
    return {"startDate": rng[0].isoformat(), "endDate": rng[1].isoformat()}


def nyfed_operation_controls(area: str, operations: list[str]) -> tuple[str, dict] | None:
    """Shared controls for the operations endpoints shaped /api/<area>/<op>/results/<include>/search."""
    c1, c2, c3 = st.columns([1, 1, 2])
    op = c1.selectbox("Operation", operations, key=f"nyfed_{area}_op", format_func=str.capitalize)
    include = c2.radio("Detail", ["summary", "details"], key=f"nyfed_{area}_include", format_func=str.capitalize)
    with c3:
        dates = nyfed_date_range()
    return (f"/api/{area}/{op}/results/{include}/search.json", dates) if dates else None


def nyfed_request_controls(dataset: str, beta: bool) -> tuple[str, dict] | None:
    """Widgets for one dataset's options; returns the (path, query params) to request, or None
    while the inputs are incomplete."""
    if dataset.startswith("Reference rates"):
        c1, c2 = st.columns([1, 2])
        report = c1.radio("Report", ["rate", "volume"], key="nyfed_rate_type", format_func=str.capitalize)
        with c2:
            dates = nyfed_date_range()
        return ("/api/rates/all/search.json", {**dates, "type": report}) if dates else None

    if dataset == "SOFR averages and index":
        dates = nyfed_date_range()
        return ("/api/rates/secured/sofrai/search.json", dates) if dates else None

    if dataset == "Agency MBS operations":
        return nyfed_operation_controls("ambs", ["all", "purchases", "sales", "roll", "swap"])

    if dataset == "Treasury securities operations":
        return nyfed_operation_controls("tsy", ["all", "purchases", "sales"])

    if dataset == "Securities lending operations":
        return nyfed_operation_controls("seclending", ["all", "seclending", "extensions"])

    if dataset == "Repo and reverse repo operations":
        c1, c2 = st.columns([1, 2])
        op_type = c1.selectbox(
            "Operation type", ["all", "repo", "reverserepo"], key="nyfed_rp_type",
            format_func={"all": "All", "repo": "Repo", "reverserepo": "Reverse repo"}.get,
        )
        with c2:
            dates = nyfed_date_range()
        if not dates:
            return None
        return "/api/rp/results/search.json", {**dates, **({"operationTypes": op_type} if op_type != "all" else {})}

    if dataset == "Reverse repo propositions":
        dates = nyfed_date_range()
        return ("/api/rp/reverserepo/propositions/search.json", dates) if dates else None

    if dataset == "Central bank liquidity swaps":
        c1, c2, c3 = st.columns([1, 1, 2])
        swap_type = c1.selectbox(
            "Swaps", ["all", "usdollar", "nonusdollar"], key="nyfed_fxs_type",
            format_func={"all": "All", "usdollar": "U.S. dollar", "nonusdollar": "Non-U.S. dollar"}.get,
        )
        date_type = c2.radio("Dates by", ["trade", "maturity"], key="nyfed_fxs_datetype", format_func=str.capitalize)
        with c3:
            dates = nyfed_date_range()
        return (f"/api/fxs/{swap_type}/search.json", {**dates, "dateType": date_type}) if dates else None

    if dataset == "SOMA holdings: weekly summary":
        return "/api/soma/summary.json", {}

    if dataset == "SOMA holdings: securities held as of a date":
        as_of_dates = fetch_nyfed_json("/api/soma/asofdates/list.json", {}, beta)["soma"]["asOfDates"]
        c1, c2, c3 = st.columns(3)
        portfolio = c1.radio("Portfolio", ["tsy", "agency"], key="nyfed_soma_portfolio",
                             format_func={"tsy": "Treasury", "agency": "Agency"}.get)
        holding_types = ["all", "bills", "notesbonds", "frn", "tips"] if portfolio == "tsy" else ["all", "agency debts", "mbs", "cmbs"]
        holding = c2.selectbox("Holding type", holding_types, key=f"nyfed_soma_holding_{portfolio}")
        as_of = c3.selectbox("As of", as_of_dates, key="nyfed_soma_asof")
        return f"/api/soma/{portfolio}/get/{holding}/asof/{as_of}.json", {}

    if dataset == "Primary dealer statistics: one series":
        options = nyfed_pd_series(beta)
        keyid = st.selectbox("Series", list(options), format_func=lambda k: f"{k} — {options[k]}", key="nyfed_pd_keyid")
        return f"/api/pd/get/{keyid}.json", {}

    if dataset == "Primary dealer statistics: latest survey":
        breaks = fetch_nyfed_json("/api/pd/list/seriesbreaks.json", {}, beta)["pd"]["seriesbreaks"]
        labels = {b["seriesbreak"]: f"{b['label']} ({b['seriesbreak']})" for b in breaks}
        sb = st.selectbox("Series break", list(labels), index=len(labels) - 1, format_func=labels.get, key="nyfed_pd_sb")
        return f"/api/pd/latest/{sb}.json", {}

    if dataset == "Primary dealer market share":
        period = st.radio("Period", ["qtrly", "ytd"], horizontal=True, key="nyfed_ms_period",
                          format_func={"qtrly": "Latest quarter", "ytd": "Year to date"}.get)
        return f"/api/marketshare/{period}/latest.json", {}

    if dataset == "Guide sheets":
        c1, c2 = st.columns(2)
        sheet = c1.selectbox("Guide sheet", ["si", "wi", "fs"], key="nyfed_gs_type",
                             format_func={"si": "FR 2004SI", "wi": "FR 2004WI", "fs": "FR 2004F-series"}.get)
        which = c2.radio("Edition", ["latest", "previous"], horizontal=True, key="nyfed_gs_which", format_func=str.capitalize)
        return f"/api/guidesheets/{sheet}/{which}.json", {}

    raw = st.text_input(
        "Endpoint path (any JSON endpoint from the API docs)",
        placeholder="/api/ambs/all/results/details/last/1.json", key="nyfed_custom_path",
    ).strip()
    st.caption(
        "Copy any path from the [API docs](https://markets.newyorkfed.org/static/docs/markets-api.html) "
        "with the `.json` format, including query parameters. Prefix it with `/beta` (or turn on beta "
        "endpoints above) for the test versions."
    )
    if not raw:
        return None
    raw = raw.removeprefix(nyfed.BASE_URL)
    if not nyfed.CUSTOM_PATH_RE.match(raw):
        st.warning("That doesn't look like a Markets Data API path: it should start with `/api/` (or `/beta/api/`) and end in `.json`.")
        return None
    path, _, query = raw.partition("?")
    return path, dict(parse_qsl(query))


def render_nyfed_response(data, file_stem: str):
    """Show every table in a response, with CSV download and a quick chart when it has dates."""
    tables = {k: v for k, v in nyfed.json_tables(data).items() if not v.empty}
    if not tables:
        st.info("The API returned no records for this request.")
        return
    names = list(tables)
    table_key = f"{zlib.crc32('|'.join(names).encode())}"
    name = st.selectbox("Table", names, key=f"nyfed_table_{table_key}") if len(names) > 1 else names[0]
    df = tables[name]
    widget_key = f"{table_key}_{zlib.crc32(name.encode())}"

    date_cols = [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]
    date_only = {
        c: st.column_config.DateColumn(c) for c in date_cols
        if (df[c].dropna() == df[c].dropna().dt.normalize()).all()
    }
    st.caption(f"{len(df):,} rows × {df.shape[1]} columns")
    st.dataframe(df, use_container_width=True, height=380, column_config=date_only)
    d1, d2 = st.columns(2)
    d1.download_button(
        "Download table as CSV", df.to_csv(index=False).encode("utf-8"),
        file_name=f"{file_stem}.csv", mime="text/csv", key=f"nyfed_csv_{widget_key}",
    )
    d2.download_button(
        "Download raw JSON", json.dumps(data).encode("utf-8"),
        file_name=f"{file_stem}.json", mime="application/json", key=f"nyfed_json_{widget_key}",
    )

    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and not pd.api.types.is_bool_dtype(df[c])]
    if not (date_cols and num_cols and len(df) > 1):
        return
    split_cols = [
        c for c in df.columns
        if c not in date_cols and c not in num_cols and 1 < df[c].nunique() <= 12
    ]
    default_y = next((c for c in NYFED_HEADLINE_FIELDS if c in num_cols), num_cols[0])
    split_options = ["(none)"] + split_cols
    st.markdown("**Chart this table**")
    c1, c2, c3, c4 = st.columns([2, 3, 2, 1])
    x = c1.selectbox("Date column", date_cols, key=f"nyfed_x_{widget_key}")
    ys = c2.multiselect("Values", num_cols, default=[default_y], key=f"nyfed_y_{widget_key}")
    split = c3.selectbox(
        "Split by", split_options, index=split_options.index("type") if "type" in split_cols else 0,
        key=f"nyfed_split_{widget_key}",
    )
    kind = c4.radio("Type", ["Line", "Bar"], key=f"nyfed_kind_{widget_key}")
    if not ys:
        return
    split = None if split == "(none)" else split
    id_vars = [x] + ([split] if split else [])
    plot = df[id_vars + ys].melt(id_vars=id_vars, value_vars=ys, var_name="field", value_name="value")
    if split and len(ys) > 1:
        plot["series"] = plot[split].astype(str) + " · " + plot["field"]
    else:
        plot["series"] = plot[split].astype(str) if split else plot["field"]
    plot = plot.groupby([x, "series"], as_index=False)["value"].sum(min_count=1).dropna().sort_values(x)
    fig = (px.line if kind == "Line" else px.bar)(plot, x=x, y="value", color="series")
    fig.update_layout(height=420, margin=dict(l=40, r=20, t=30, b=40), legend_title_text=None)
    st.plotly_chart(fig, use_container_width=True, key=f"nyfed_chart_{widget_key}")
    st.caption("Rows that share a date (and split value) are summed.")


# Parameters that pick a table come first so the remaining parameters' value lists can be
# narrowed to that table; Year goes last for the same reason.
BEA_PARAM_PRIORITY = ["TableName", "TableID", "Indicator", "Component", "TypeOfInvestment", "TypeOfService", "Frequency"]


def bea_param_order(names: list[str]) -> list[str]:
    def rank(item):
        i, name = item
        if name in BEA_PARAM_PRIORITY:
            return (0, BEA_PARAM_PRIORITY.index(name))
        return (2, i) if name == "Year" else (1, i)
    return [name for _, name in sorted(enumerate(names), key=rank)]


def bea_text(value) -> str:
    """A BEA metadata field as text; BEA leaves many blank, which beaapi returns as NA."""
    return "" if value is None or (not isinstance(value, str) and pd.isna(value)) else str(value).strip()


def bea_param_widget(dataset: str, p: pd.Series, chosen: dict, key: str) -> str:
    """One parameter's input, using BEA's list of allowed values when it has one. Returns the
    value to send ('' if none), with multiple picks joined by commas as BEA expects."""
    name = p["ParameterName"]
    filters = tuple(sorted(chosen.items())) if dataset not in bea.NO_FILTERED_VALUES else ()
    try:
        values = bea_param_values(dataset, name, filters, key)
    except (*bea.BEAError, OSError, KeyError):  # no usable value list: fall back to typing the value
        values = {}
    if dataset == "Regional" and name == "GeoFips" and values:
        values = {**bea.REGIONAL_GEO_KEYWORDS, **values}
    all_value = bea_text(p.get("AllValue"))
    default_value = bea_text(p.get("ParameterDefaultValue"))
    label = name + (" *" if p["required"] else "")
    widget_key = f"bea_{dataset}_{name}"
    help_text = bea_text(p.get("ParameterDescription")) or None

    if not values:
        return st.text_input(label, value=default_value, help=help_text, key=widget_key).strip()

    def fmt(k):
        return "All" if k == all_value else f"{k} — {values.get(k, '')}"[:160]

    if p["multi"]:
        options = ([all_value] if all_value else []) + [k for k in values if k != all_value]
        if name == "Year":
            default = [all_value] if all_value else sorted(values, reverse=True)[:10]
        else:
            default = [default_value] if default_value in options else []
        picked = st.multiselect(label, options, default=default, format_func=fmt, help=help_text, key=widget_key)
        if dataset == "Regional" and name == "GeoFips" and len(picked) > 1 and set(picked) & set(bea.REGIONAL_GEO_KEYWORDS):
            st.warning("BEA ignores an \"All ...\" choice combined with other areas. Pick either one \"All ...\" option or specific areas.")
        return ",".join(picked)

    options = list(values)
    index = options.index(default_value) if default_value in options else None
    picked = st.selectbox(
        label, options, index=index, format_func=fmt, placeholder="Choose...", help=help_text, key=widget_key,
    )
    return picked or ""


def bea_param_controls(dataset: str, key: str) -> tuple[dict, list[str]]:
    """Inputs for every parameter of a BEA dataset. Returns the chosen {parameter: value} and
    the required parameters still left empty."""
    params = bea_params(dataset, key).copy()
    params["required"] = params["ParameterIsRequiredFlag"].map(bea_text) == "1"
    params["multi"] = params["MultipleAcceptedFlag"].map(bea_text) == "1"
    # NIPA-style datasets accept either TableName or the deprecated TableID, so BEA flags
    # neither as required; treat the preferred one as required so it isn't buried.
    table_params = params["ParameterName"].isin(["TableName", "TableID"])
    if table_params.any() and not params.loc[table_params, "required"].any():
        preferred = "TableName" if (params["ParameterName"] == "TableName").any() else "TableID"
        params.loc[params["ParameterName"] == preferred, "required"] = True
    params = params.set_index("ParameterName", drop=False).loc[bea_param_order(list(params["ParameterName"]))]

    chosen, missing = {}, []
    required, optional = params[params["required"]], params[~params["required"]]
    if required.empty:  # e.g. ITA, IIP: every parameter is optional, but some must still be set
        required, optional = optional, optional.iloc[0:0]
    cols = st.columns(2)
    for i, (_, p) in enumerate(required.iterrows()):
        with cols[i % 2]:
            value = bea_param_widget(dataset, p, chosen, key)
        if value:
            chosen[p["ParameterName"]] = value
        elif p["required"]:
            missing.append(p["ParameterName"])
    if not optional.empty:
        with st.expander("Optional parameters"):
            for _, p in optional.iterrows():
                value = bea_param_widget(dataset, p, chosen, key)
                if value:
                    chosen[p["ParameterName"]] = value
    return chosen, missing


def render_bea_table(dataset: str, params: tuple, key: str):
    df, keys, statistic = bea_table(dataset, params, key)
    st.caption(
        f"**{dataset}** · " + " · ".join(f"{k}={v}" for k, v in params) + f" — {len(df):,} rows"
        + (f" · {statistic}" if statistic else "")
    )
    request_key = f"{zlib.crc32(repr((dataset, params)).encode())}"
    st.dataframe(
        df, use_container_width=True, height=320,
        column_config={"Date": st.column_config.DateColumn("Date")} if "Date" in df.columns else None,
    )
    st.download_button(
        "Download table as CSV", df.to_csv(index=False).encode("utf-8"),
        file_name=f"bea_{dataset.lower()}.csv", mime="text/csv", key=f"bea_csv_{request_key}",
    )

    catalog = bea.series_catalog(df, keys, statistic)
    if catalog.empty or not df["Date"].notna().any():
        return
    st.markdown("**Chart series from this table**")
    labels = catalog["label"].tolist()
    picked = st.multiselect(
        f"Series ({len(catalog):,} in this table)", list(range(len(catalog))), default=[0],
        format_func=lambda i: labels[i], key=f"bea_pick_{request_key}",
    )
    if not picked:
        return
    filters_for = {i: {k: str(catalog.iloc[i][k]) for k in keys} for i in picked}
    long = pd.concat([
        bea.series_values(df, filters_for[i]).rename("value").rename_axis("Date").reset_index().assign(series=labels[i])
        for i in picked
    ])
    fig = px.line(long.dropna(), x="Date", y="value", color="series", markers=len(long) <= 60)
    fig.update_layout(height=420, margin=dict(l=40, r=20, t=30, b=40), legend_title_text=None,
                      legend=dict(orientation="h", yanchor="top", y=-0.15))
    st.plotly_chart(fig, use_container_width=True, key=f"bea_chart_{request_key}")
    units = sorted({catalog.iloc[i]["unit"] for i in picked if catalog.iloc[i]["unit"]})
    if units:
        st.caption("Units: " + "; ".join(units))

    if st.button("Add to the Time Series and Relationship tabs", key=f"bea_add_{request_key}"):
        added = []
        for i in picked:
            unit = catalog.iloc[i]["unit"]
            spec = {
                "dataset": dataset, "params": dict(params), "filters": filters_for[i],
                "label": f"{labels[i]} ({unit})" if unit else labels[i],
            }
            sid = f"{bea.BEA_PREFIX}{slugify(labels[i])[:32]}_{zlib.crc32(repr(spec).encode()) & 0xFFFF:04X}"
            get_bea_series()[sid] = spec
            added.append(sid)
        st.session_state["ts_bea_series"] = list(dict.fromkeys(st.session_state.get("ts_bea_series", []) + added))
        st.success(
            "Added and selected in the Time Series sidebar (under **BEA series**). In the Relationship "
            "tab or a derived series, use the ID: " + ", ".join(f"`{s}`" for s in added)
        )


def census_default_time(example_time: str | None) -> str:
    """A recent 'from ...' range in the same granularity as the dataset's example time value
    (e.g. '2010-Q1' -> 'from 2018-Q1', '2013-01' -> 'from 2018-01', '2012' -> 'from 2018')."""
    this_year = datetime.date.today().year
    m = re.match(r"^\d{4}(-Q\d|-\d{2})?", str(example_time or ""))
    suffix = (m.group(1) or "") if m else ""
    start, end = ("-Q1", "-Q4") if suffix.startswith("-Q") else ("-01", "-12") if suffix else ("", "")
    # Bounded, since some datasets (e.g. QWI) reject open-ended ranges.
    return f"from {this_year - 8}{start} to {this_year}{end}"


def census_request_controls(key: str | None) -> tuple | None:
    """Inputs for one Census query. Returns the census_table() arguments (path, get variables,
    predicates, numeric variables, geography columns) or None while incomplete."""
    cat = census_catalog()
    if not st.toggle("Include microdata datasets", key="census_micro",
                     help="Individual survey responses (CPS, SIPP, ACS PUMS and similar). These need "
                          "survey weights to interpret and aren't economic time series."):
        cat = cat[~cat["microdata"]]
    vintage = cat["vintage"].map(lambda v: str(int(v)) if pd.notna(v) else "time series")
    labels = dict(zip(cat["path"], cat["title"] + " — " + vintage + " (" + cat["path"] + ")"))
    path = st.selectbox(
        f"Dataset ({len(labels):,} available — type to search)", list(labels), index=None,
        format_func=labels.get, placeholder="e.g. ACS 1-Year, County Business Patterns, Advance Monthly Retail Sales...",
        key="census_dataset",
    )
    if not path:
        return None
    ex = census_examples(path)

    groups = census_groups(path)
    group = None
    if len(groups) > 1:  # e.g. ACS: thousands of variables, organized into tables
        group_desc = {g["name"]: g.get("description", "") for g in groups}
        ex_group = next((v.rsplit("_", 1)[0] for v in ex.get("get", []) if v.rsplit("_", 1)[0] in group_desc), None)
        group = st.selectbox(
            f"Table ({len(group_desc):,})", list(group_desc),
            index=list(group_desc).index(ex_group) if ex_group else None,
            format_func=lambda g: f"{g} — {group_desc[g]}"[:170], placeholder="Choose a table...",
            key=f"census_group_{path}",
        )
        if not group:
            return None
        var_meta = census.chooseable_variables(census_group_vars(path, group))
    else:
        var_meta = census.chooseable_variables(census_vars(path))

    required = [n for n, m in var_meta.items() if str(m.get("required", "")).lower() == "true"]
    default = [v for v in ex.get("get", []) if v in var_meta]
    if not any(v not in ("NAME", "GEO_ID") for v in default):  # e.g. the example uses a different ACS table
        default = (["NAME"] if "NAME" in var_meta else []) + [
            next((n for n in var_meta if n.endswith("_001E")), next((n for n in var_meta if n not in ("NAME", "GEO_ID")), None))
        ]
    default = [d for d in dict.fromkeys(default + required) if d]
    picked = st.multiselect(
        "Variables", list(var_meta), default=default,
        format_func=lambda n: f"{n} — {var_meta[n].get('label', '')}"[:170],
        help="Required variables for this dataset are always included.", key=f"census_vars_{path}_{group}",
    )
    get_vars = list(dict.fromkeys(picked + required))
    if not get_vars:
        st.caption("Pick at least one variable.")
        return None

    predicates: dict[str, str] = {}
    geo_cols: set[str] = set()
    levels = census_geo(path)
    if levels:
        names = [lv["name"] for lv in levels]
        ex_level = next((f.get("name") for f in ex.get("fips", []) if f.get("name") in names), None)
        default_level = "state" if "state" in names else ex_level or names[0]
        c1, c2, c3 = st.columns(3)
        level = c1.selectbox("Geography", names, index=names.index(default_level), key=f"census_level_{path}")
        spec = levels[names.index(level)]
        if spec.get("wildcard") is False:  # e.g. QWI: one state at a time
            example_value = next((f.get("exampleValue", "") for f in ex.get("fips", []) if f.get("name") == level), "")
            which = c2.text_input(f"Which {level} (code; * isn't allowed here)", value=example_value,
                                  key=f"census_which_{path}_{level}")
        else:
            which = c2.text_input(f"Which {level} (codes, or * for all)", value="*", key=f"census_which_{path}_{level}")
        if not which.strip():
            st.caption(f"Enter a {level} code.")
            return None
        predicates["for"] = f"{level}:{which.strip()}"
        geo_cols.add(level)
        in_parts = []
        for parent in spec.get("requires") or []:
            wildcard_ok = parent in (spec.get("wildcard") or [])
            if parent == "state":
                options = (["*"] if wildcard_ok else []) + list(census.STATE_FIPS)
                value = c3.selectbox(
                    "In state", options, key=f"census_in_{path}_{level}_state",
                    format_func=lambda k: "All states" if k == "*" else f"{k} — {census.STATE_FIPS[k]}",
                )
            else:
                value = c3.text_input(
                    f"In {parent} (code{', or * for all' if wildcard_ok else ''})",
                    value="*" if wildcard_ok else "", key=f"census_in_{path}_{level}_{parent}",
                ).strip()
            if not value:
                st.caption(f"Enter a {parent} code for this geography.")
                return None
            in_parts.append(f"{parent}:{value}")
            geo_cols.add(parent)
        if in_parts:
            predicates["in"] = " ".join(in_parts)

    ex_predicates = dict(ex.get("predicates") or {})
    c4, c5 = st.columns(2)
    if path.startswith("timeseries/") or "time" in ex_predicates:
        time_value = c4.text_input(
            "Time", value=census_default_time(ex_predicates.get("time")), key=f"census_time_{path}",
            help="A period like 2024, 2024-03 or 2024-Q1, or a range like 'from 2015' or 'from 2015 to 2020'.",
        ).strip()
        if time_value:
            predicates["time"] = time_value
    ex_predicates.pop("time", None)
    extra = c5.text_input(
        "Extra filters (optional)", value=urlencode(ex_predicates), key=f"census_extra_{path}",
        help="Further predicates as name=value pairs joined by &, e.g. category_code=44X72&seasonally_adj=yes",
    ).strip()
    predicates.update(dict(parse_qsl(extra)))

    numeric = {n for n in get_vars if var_meta.get(n, {}).get("predicateType") in ("int", "float")}
    # Fields Census labels as codes (e.g. "2022 NAICS code") stay text even when they look numeric.
    geo_cols |= {n for n in get_vars if re.search(r"\bcode\b", var_meta.get(n, {}).get("label", ""), re.IGNORECASE)}
    return path, tuple(get_vars), tuple(predicates.items()), tuple(sorted(numeric - geo_cols)), tuple(sorted(geo_cols)), var_meta


def render_census_table(path: str, get_vars: tuple, predicates: tuple, numeric: tuple, geo: tuple,
                        var_meta: dict, key: str):
    df = census_table(path, get_vars, predicates, numeric, geo, key)
    st.caption(f"**{path}** · get={','.join(get_vars)} · " + " · ".join(f"{k}={v}" for k, v in predicates)
               + f" — {len(df):,} rows")
    if df.empty:
        st.info("The Census API returned no rows for this query.")
        return
    request_key = f"{zlib.crc32(repr((path, get_vars, predicates)).encode())}"
    st.dataframe(
        df, use_container_width=True, height=320,
        column_config={"Date": st.column_config.DateColumn("Date")} if "Date" in df.columns else None,
    )
    st.download_button(
        "Download table as CSV", df.to_csv(index=False).encode("utf-8"),
        file_name=f"census_{slugify(path).lower()}.csv", mime="text/csv", key=f"census_csv_{request_key}",
    )

    value_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and c not in geo]
    if not value_cols:
        return

    def value_label(c):
        return f"{c} — {var_meta.get(c, {}).get('label', '')}".rstrip(" —")[:120]

    st.markdown("**Chart**")
    value_col = st.selectbox("Value", value_cols, format_func=value_label, key=f"census_value_{request_key}")

    if "Date" not in df.columns or not df["Date"].notna().any():  # a cross-section, e.g. one ACS year by state
        name_col = "NAME" if "NAME" in df.columns else next((c for c in df.columns if c not in value_cols), None)
        top = df.dropna(subset=[value_col]).nlargest(30, value_col)
        fig = px.bar(top, x=value_col, y=name_col, orientation="h")
        fig.update_layout(height=max(300, 22 * len(top)), margin=dict(l=40, r=20, t=30, b=40),
                          yaxis=dict(autorange="reversed", title=None))
        st.plotly_chart(fig, use_container_width=True, key=f"census_bar_{request_key}")
        st.caption(f"Top {len(top)} by {value_col}. Pick a time-series dataset to chart values over time.")
        return

    cat = census_catalog()
    title = cat.loc[cat["path"] == path, "title"].iloc[0] if (cat["path"] == path).any() else path
    short_title = title.rsplit(": ", 1)[-1].strip()
    keys = census.series_key_cols(df)
    catalog = df.drop_duplicates(keys)[keys].reset_index(drop=True) if keys else pd.DataFrame(index=[0])
    order = ["NAME"] + [k for k in keys if k != "NAME"]
    labels = [
        " · ".join(str(row[k]) for k in order if k in row.index) or short_title
        for _, row in catalog.iterrows()
    ]
    picked = st.multiselect(
        f"Series ({len(catalog):,} in this result)", list(range(len(catalog))), default=[0],
        format_func=lambda i: labels[i], key=f"census_pick_{request_key}",
    )
    if not picked:
        return
    filters_for = {i: {k: str(catalog.iloc[i][k]) for k in keys} for i in picked}
    long = pd.concat([
        census.series_values(df, filters_for[i], value_col).rename("value").rename_axis("Date").reset_index()
        .assign(series=labels[i])
        for i in picked
    ])
    fig = px.line(long.dropna(), x="Date", y="value", color="series", markers=len(long) <= 60)
    fig.update_layout(height=420, margin=dict(l=40, r=20, t=30, b=40), legend_title_text=None,
                      legend=dict(orientation="h", yanchor="top", y=-0.15))
    st.plotly_chart(fig, use_container_width=True, key=f"census_chart_{request_key}")

    if st.button("Add to the Time Series and Relationship tabs", key=f"census_add_{request_key}"):
        added = []
        for i in picked:
            label = f"{short_title}: {labels[i]}" if keys else short_title
            if len(value_cols) > 1:
                label += f" ({value_col})"
            spec = {
                "query": (path, get_vars, predicates, numeric, geo), "filters": filters_for[i],
                "value_col": value_col, "label": label,
            }
            sid = f"{census.CENSUS_PREFIX}{slugify(labels[i])[:32]}_{zlib.crc32(repr(spec).encode()) & 0xFFFF:04X}"
            get_census_series()[sid] = spec
            added.append(sid)
        st.session_state["ts_census_series"] = list(
            dict.fromkeys(st.session_state.get("ts_census_series", []) + added)
        )
        st.success(
            "Added and selected in the Time Series sidebar (under **Census series**). In the "
            "Relationship tab or a derived series, use the ID: " + ", ".join(f"`{s}`" for s in added)
        )


tab_series, tab_relationship, tab_nyfed, tab_bea, tab_census = st.tabs(
    ["Time Series", "Relationship (X vs Y)", "NY Fed Markets", "BEA", "Census"]
)

# --- Census tab ------------------------------------------------------------
# Rendered before the Time Series tab so a series added here is selectable there on the same run.

with tab_census:
    st.write(
        "Query the U.S. Census Bureau's [Data API](https://www.census.gov/data/developers/data-sets.html): "
        "the American Community Survey, decennial census, County Business Patterns, economic indicators "
        "(retail sales, construction, trade, manufacturing...), international trade, QWI and more. Time "
        "series can be added to the Time Series and Relationship tabs alongside FRED, NY Fed and BEA data."
    )
    census_key = census.get_census_key()
    if not census_key:
        st.info(
            "Browsing datasets and variables works without a key, but fetching data needs a free Census "
            "API key: request one at https://api.census.gov/data/key_signup.html, then add it as "
            "`CENSUS_API_KEY` in the app's Streamlit secrets, or run `python setup_api_key.py --census` "
            "when running locally."
        )
    try:
        request = census_request_controls(census_key)
        if request:
            if st.button("Get data", type="primary", disabled=not census_key, key="census_get"):
                st.session_state["census_request"] = request
                st.session_state.pop("census_error", None)
        saved = st.session_state.get("census_request")
        if saved and request and saved[0] == request[0]:
            try:
                render_census_table(*saved, census_key)
            except (census.CensusError, requests.RequestException) as e:
                # Forget a failed request so later reruns don't resend it.
                st.session_state.pop("census_request")
                st.session_state["census_error"] = (saved[0], str(e))
        error = st.session_state.get("census_error")
        if error and request and error[0] == request[0]:
            st.error(f"Census request failed: {error[1]}")
    except (census.CensusError, requests.RequestException) as e:
        st.error(f"Couldn't load Census metadata: {e}")
    except Exception as e:  # an unexpected response shape shouldn't take down the other tabs
        st.error(f"Couldn't display this Census data ({type(e).__name__}: {e}).")
    st.caption(
        "This product uses the Census Bureau Data API but is not endorsed or certified by the Census Bureau."
    )

# --- BEA tab ---------------------------------------------------------------
# Rendered before the Time Series tab so a series added here is selectable there on the same run.

with tab_bea:
    st.write(
        "Pull data from the U.S. Bureau of Economic Analysis using BEA's official "
        "[`beaapi`](https://github.com/us-bea/beaapi) library: GDP and the national accounts (NIPA), "
        "GDP by industry, regional income and GDP by state, county and metro area, international "
        "transactions and investment, input-output tables and more. Add any series to the Time Series "
        "and Relationship tabs to chart it alongside FRED and NY Fed data."
    )
    bea_key = bea.get_bea_key()
    if not bea_key:
        st.info(
            "BEA data needs a free API key. Sign up at https://apps.bea.gov/API/signup/ (BEA emails the "
            "key to you), then add it as `BEA_API_KEY` in the app's Streamlit secrets, or run "
            "`python setup_api_key.py --bea` when running locally."
        )
    else:
        try:
            datasets = bea_datasets(bea_key)
            descriptions = dict(zip(datasets["DatasetName"], datasets["DatasetDescription"]))
            dataset = st.selectbox(
                "Dataset", list(descriptions), format_func=lambda n: f"{n} — {descriptions[n]}", key="bea_dataset",
            )
            chosen, missing = bea_param_controls(dataset, bea_key)
            if missing:
                st.caption("Still needed: " + ", ".join(missing) + " (* marks required parameters).")
            if st.button("Get data", type="primary", disabled=bool(missing), key="bea_get"):
                st.session_state["bea_request"] = (dataset, tuple(chosen.items()))
                st.session_state.pop("bea_error", None)
            request = st.session_state.get("bea_request")
            if request and request[0] == dataset:
                try:
                    render_bea_table(*request, bea_key)
                except (*bea.BEAError, OSError) as e:
                    # Forget a failed request so later reruns don't resend it: BEA throttles
                    # keys that make too many failing calls.
                    st.session_state.pop("bea_request")
                    st.session_state["bea_error"] = (dataset, str(e))
            if st.session_state.get("bea_error", (None,))[0] == dataset:
                st.error(f"BEA request failed: {st.session_state['bea_error'][1]}")
        except (*bea.BEAError, OSError) as e:
            st.error(f"BEA request failed: {e}")
        except Exception as e:  # an unexpected response shape shouldn't take down the other tabs
            st.error(f"Couldn't display this BEA data ({type(e).__name__}: {e}). Try different parameters.")
    st.caption("Data: U.S. Bureau of Economic Analysis.")

# --- NY Fed Markets tab ----------------------------------------------------
# Rendered before the other tabs because they call st.stop() when they have nothing to show.

with tab_nyfed:
    st.write(
        "Browse the Federal Reserve Bank of New York's [Markets Data APIs]"
        "(https://markets.newyorkfed.org/static/docs/markets-api.html): reference rates, open market "
        "operations, SOMA holdings, primary dealer statistics and more. No API key needed. Time-series "
        "datasets can also be charted with FRED data: pick them under **NY Fed Markets series** in the "
        "Time Series sidebar."
    )
    beta = st.toggle(
        "Use beta endpoints", key="nyfed_beta",
        help="Sends every NY Fed request, here and in the other tabs, to the API's /beta versions, which "
             "preview upcoming changes. Beta data are for testing only and aren't regularly maintained.",
    )
    dataset = st.selectbox("Dataset", NYFED_DATASETS, key="nyfed_dataset")
    try:
        request = nyfed_request_controls(dataset, beta)
        if request:
            path, params = request
            shown_path = path if path.startswith("/beta/") or not beta else "/beta" + path
            url = nyfed.BASE_URL + shown_path + (f"?{urlencode(params)}" if params else "")
            st.caption(f"Request: `{url}`")
            data = fetch_nyfed_json(path, params, beta)
            render_nyfed_response(data, "nyfed_" + slugify(dataset).lower())
    except (nyfed.NYFedError, requests.RequestException, KeyError, ValueError) as e:
        st.error(f"Couldn't load this from the NY Fed: {e}")
    st.caption(
        "Data: Federal Reserve Bank of New York, subject to its "
        "[Terms of Use](https://www.newyorkfed.org/privacy/termsofuse)."
    )

# --- Time Series tab -----------------------------------------------------

with tab_series:
    with st.sidebar:
        st.header("Series")
        series_options = list(DEFAULT_SERIES.keys()) + list(derived_defs.keys())
        chosen_labels = st.multiselect(
            "Pick series",
            options=series_options,
            default=["UNRATE", "CPIAUCSL"],
            format_func=lambda sid: f"🧮 {derived_defs[sid]['display']}" if sid in derived_defs else f"{sid} — {DEFAULT_SERIES[sid]}",
        )
        custom = st.text_input("Add a custom FRED series ID (comma-separated)", "")
        custom_ids = [c.strip().upper() for c in custom.split(",") if c.strip()]

        with st.expander("NY Fed Markets series"):
            nyfed_chosen = st.multiselect(
                "Reference rates, SOMA holdings, repo operations",
                options=list(nyfed.NYFED_SERIES), format_func=nyfed.NYFED_SERIES.get, key="ts_nyfed_series",
            )
            try:
                pd_options = nyfed_pd_series(nyfed_beta())
            except (nyfed.NYFedError, requests.RequestException, KeyError) as e:
                pd_options = {}
                st.caption(f"Couldn't load the primary dealer series list: {e}")
            pd_chosen = st.multiselect(
                "Primary dealer statistics", options=list(pd_options),
                format_func=lambda k: f"{k} — {pd_options[k]}", key="ts_nyfed_pd",
            )
            st.caption(
                "From the New York Fed's Markets Data APIs, no key needed. These IDs (e.g. `NYFED:SOFR`, "
                "`NYFED:PD:PDPOSMBS-TOT`) also work anywhere a series ID is typed, including derived "
                "series and the Relationship tab. Rates are daily and SOMA/dealer data weekly, so set "
                "the resample frequency to Monthly when mixing them with monthly FRED series."
            )

        bea_defs = get_bea_series()
        with st.expander("BEA series", expanded=bool(st.session_state.get("ts_bea_series"))):
            bea_chosen = st.multiselect(
                "Series added from the BEA tab", options=list(bea_defs),
                format_func=lambda sid: bea_defs[sid]["label"], key="ts_bea_series",
            )
            if not bea_defs:
                st.caption("Fetch a table in the **BEA** tab, then add series from it to chart them here.")

        census_defs = get_census_series()
        with st.expander("Census series", expanded=bool(st.session_state.get("ts_census_series"))):
            census_chosen = st.multiselect(
                "Series added from the Census tab", options=list(census_defs),
                format_func=lambda sid: census_defs[sid]["label"], key="ts_census_series",
            )
            if not census_defs:
                st.caption("Fetch a time series in the **Census** tab, then add series from it to chart them here.")

        series_ids = tuple(dict.fromkeys(  # dedupe, keep order
            chosen_labels + nyfed_chosen + [nyfed.PD_PREFIX + k for k in pd_chosen] + bea_chosen
            + census_chosen + custom_ids
        ))

        if series_ids:
            with st.expander("Rename series (legends, axes, exports)"):
                for sid in series_ids:
                    rename_control(sid)

        st.header("Date range")
        time_window_control("ts", "ts_start", "ts_end")
        start_date = st.date_input(
            "Start", value=datetime.date(2000, 1, 1),
            min_value=datetime.date(1900, 1, 1), max_value=datetime.date.today(), key="ts_start",
        )
        end_date = st.date_input(
            "End", value=datetime.date.today(),
            min_value=datetime.date(1900, 1, 1), max_value=datetime.date.today(), key="ts_end",
        )

        st.header("Transform")
        view = st.radio("View data as", TRANSFORMS, key="ts_view")
        freq_label = st.selectbox("Resample frequency", list(FREQ_MAP.keys()), key="ts_freq")
        freq = FREQ_MAP[freq_label]

        st.header("Colors")
        ts_bg_color = st.color_picker("Background", "#FFFFFF", key="ts_bg")
        series_colors = {}
        for i, sid in enumerate(series_ids):
            default_color = DEFAULT_PALETTE[i % len(DEFAULT_PALETTE)]
            series_colors[sid] = st.color_picker(
                series_display_label(sid), default_color, key=f"ts_color_{sid}"
            )

    if not series_ids:
        st.info("Pick at least one series from the sidebar.")
        st.stop()

    try:
        df = fetch_with_derived(series_ids, start_date.isoformat(), end_date.isoformat())
    except ValueError as e:
        st.error(str(e))
        st.stop()

    if freq:
        df = df.resample(freq).mean()

    periods = 12 if freq in (None, "ME") else (4 if freq == "QE" else 1)
    plot_df = pd.DataFrame({col: apply_transform(df[col], view, periods)[0] for col in df.columns})
    y_label = apply_transform(df[df.columns[0]], view, periods)[1]
    plot_df = plot_df.dropna(how="all")

    label_colors = {series_display_label(sid): series_colors[sid] for sid in series_ids}
    ts_text_color = contrast_text_color(ts_bg_color)

    st.subheader("Chart")
    fig = build_ts_figure(plot_df, label_colors, y_label, ts_bg_color, ts_text_color)
    st.plotly_chart(fig, use_container_width=True)

    ts_date_bounds = (plot_df.index.min().date(), plot_df.index.max().date())

    with st.expander("Export chart as image (interactive style)"):
        export_chart_section(
            lambda s, e: build_ts_figure(slice_by_date(plot_df, s, e), label_colors, y_label, ts_bg_color, ts_text_color),
            "ts_export", "fred_timeseries", date_bounds=ts_date_bounds,
        )

    with st.expander("Publication-style export (matplotlib, NBER shading)"):
        layout = st.radio(
            "Layout",
            ["Side-by-side panels", "Single overlay"],
            index=0 if len(plot_df.columns) <= 2 else 1,
            horizontal=True,
            key="ts_pub_layout",
        )
        if layout == "Side-by-side panels":
            build_fn = lambda rec, s, e: build_panels_figure(slice_by_date(plot_df, s, e), label_colors, y_label, rec)
        else:
            secondary_cols = st.multiselect(
                "Series on secondary (right) axis — use this when curves have very different scales/units",
                options=list(plot_df.columns),
                key="ts_pub_secondary",
            )
            build_fn = lambda rec, s, e: build_overlay_figure(
                slice_by_date(plot_df, s, e), label_colors, y_label, rec, secondary_cols
            )
        publication_export_section(
            "ts_pub", "fred_timeseries_publication", build_fn, needs_recession=True, date_bounds=ts_date_bounds
        )

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Data")
        st.dataframe(plot_df, use_container_width=True, height=350)
        st.download_button(
            "Download CSV",
            plot_df.to_csv().encode("utf-8"),
            file_name="fred_export.csv",
            mime="text/csv",
        )

    with col2:
        if len(series_ids) > 1:
            st.subheader("Correlation")
            corr = df.corr()
            heat = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1)
            heat.update_layout(height=350, margin=dict(l=20, r=20, t=20, b=20))
            st.plotly_chart(heat, use_container_width=True)
        else:
            st.subheader("Summary stats")
            st.dataframe(df.describe(), use_container_width=True)

# --- Relationship (X vs Y) tab --------------------------------------------

with tab_relationship:
    st.write(
        "Plot one series against another over time — e.g. the **Beveridge curve** "
        "(unemployment vs. job openings) or the **Phillips curve** (unemployment vs. inflation). "
        "Points are connected in chronological order and colored by date, or you can manually "
        "select and group points with your own colors and shapes (see below the chart)."
    )

    preset_choice = st.selectbox("Preset", list(RELATIONSHIP_PRESETS.keys()), key="rel_preset")
    preset = RELATIONSHIP_PRESETS[preset_choice]
    default_x, default_y, default_xt, default_yt = preset if preset else ("UNRATE", "CPIAUCSL", "Raw", "Raw")

    if derived_defs:
        st.caption("You can also type a derived series ID here, e.g. `" + next(iter(derived_defs)) + "`.")

    # Keying widgets by the preset forces them to pick up new defaults when the
    # preset changes, while still letting the user freely edit within a preset.
    colA, colB = st.columns(2)
    with colA:
        st.markdown("**X axis**")
        x_id = st.text_input("X series ID", value=default_x, key=f"x_id_{preset_choice}").strip().upper()
        x_transform = st.selectbox(
            "X transform", TRANSFORMS, index=TRANSFORMS.index(default_xt), key=f"x_t_{preset_choice}"
        )
        if x_id:
            rename_control(x_id)
    with colB:
        st.markdown("**Y axis**")
        y_id = st.text_input("Y series ID", value=default_y, key=f"y_id_{preset_choice}").strip().upper()
        y_transform = st.selectbox(
            "Y transform", TRANSFORMS, index=TRANSFORMS.index(default_yt), key=f"y_t_{preset_choice}"
        )
        if y_id:
            rename_control(y_id)

    time_window_control("rel", None, None, range_key="rel_time_range")
    colC, colD = st.columns([3, 1])
    with colC:
        rel_start, rel_end = st.slider(
            "Date range",
            min_value=datetime.date(1900, 1, 1), max_value=datetime.date.today(),
            value=st.session_state.get("rel_time_range", (datetime.date(2000, 1, 1), datetime.date.today())),
            key="rel_time_range",
        )
    with colD:
        rel_freq_label = st.selectbox("Resample frequency", list(FREQ_MAP.keys()), index=1, key="rel_freq")
        rel_freq = FREQ_MAP[rel_freq_label]

    st.markdown("**Colors**")
    colF, colG, colH, colI = st.columns(4)
    with colF:
        rel_line_color = st.color_picker("Path line", "#A0A0A0", key="rel_line_color")
    with colG:
        rel_colorscale = st.selectbox("Point gradient (by year)", COLORSCALES, key="rel_colorscale")
    with colH:
        rel_latest_color = st.color_picker("Latest point", "#FF0000", key="rel_latest_color")
    with colI:
        rel_bg_color = st.color_picker("Background", "#FFFFFF", key="rel_bg_color")
    rel_point_color = st.color_picker(
        "Publication-style point color (used in the matplotlib export below)", "#4C72B0", key="rel_point_color"
    )

    groups = st.session_state.setdefault("rel_groups", {})
    assignments = st.session_state.setdefault("rel_point_assignments", {})

    show_regression = st.checkbox(
        "Show regression lines (one per group, or one overall if you haven't defined any groups)",
        key="rel_show_regression",
    )
    regression_groups = list(groups.keys())
    if show_regression and groups:
        regression_groups = st.multiselect(
            "Which groups get a regression line? ('Overall' fits one line across every point, "
            "regardless of group)",
            options=["Overall"] + list(groups.keys()), default=list(groups.keys()), key="rel_regression_groups",
        )

    with st.expander("Manually segregate points (color/shape by group)"):
        st.caption(
            "Box- or lasso-select points on the chart below, then assign them to a named group. "
            "Grouped points get their own color and marker shape, in place of the year-gradient coloring."
        )
        gc1, gc2, gc3, gc4 = st.columns([2, 1, 1, 1])
        with gc1:
            new_group_name = st.text_input("Group name", key="rel_new_group_name")
        with gc2:
            new_group_color = st.color_picker(
                "Color", DEFAULT_PALETTE[len(groups) % len(DEFAULT_PALETTE)], key="rel_new_group_color"
            )
        with gc3:
            new_group_shape = st.selectbox("Shape", list(SHAPE_OPTIONS.keys()), key="rel_new_group_shape")
        with gc4:
            st.write("")
            st.write("")
            if st.button("Add group", key="rel_add_group"):
                if new_group_name.strip():
                    groups[new_group_name.strip()] = {"color": new_group_color, "shape": new_group_shape}
                else:
                    st.warning("Give the group a name.")

        if groups:
            st.caption("Your groups:")
            for gname, gcfg in list(groups.items()):
                r1, r2, r3 = st.columns([2, 2, 1])
                r1.color_picker(gname, gcfg["color"], key=f"rel_group_swatch_{gname}", disabled=True)
                r2.write(f"Shape: {gcfg['shape']}")
                if r3.button("Remove", key=f"rel_del_group_{gname}"):
                    del groups[gname]
                    for k in [k for k, v in assignments.items() if v == gname]:
                        del assignments[k]
                    st.rerun()

    if x_id and y_id:
        try:
            rel_df = fetch_with_derived((x_id, y_id), rel_start.isoformat(), rel_end.isoformat())
        except ValueError as e:
            st.error(str(e))
            st.stop()

        if rel_df.shape[1] < 2:
            st.error("Couldn't fetch both series — check that both FRED (or derived) series IDs are valid.")
        else:
            if rel_freq:
                rel_df = rel_df.resample(rel_freq).mean()
            rel_periods = 12 if rel_freq in (None, "ME") else (4 if rel_freq == "QE" else 1)

            x_label = series_display_label(x_id)
            y_label_name = series_display_label(y_id)
            x_col, x_axis_label = apply_transform(rel_df[x_label], x_transform, rel_periods)
            y_col, y_axis_label = apply_transform(rel_df[y_label_name], y_transform, rel_periods)

            merged = pd.concat([x_col.rename("x"), y_col.rename("y")], axis=1).dropna()

            if merged.empty:
                st.warning("No overlapping data for these two series/date range.")
            else:
                st.subheader(f"{x_label} vs. {y_label_name}")
                rel_text_color = contrast_text_color(rel_bg_color)
                point_groups = [assignments.get(ts.isoformat()) for ts in merged.index]

                fig2 = build_relationship_figure(
                    merged, groups, assignments, rel_line_color, rel_colorscale, rel_latest_color,
                    x_label, y_label_name, x_axis_label, y_axis_label, rel_bg_color, rel_text_color,
                    show_regression=show_regression, regression_groups=regression_groups,
                )
                event = st.plotly_chart(
                    fig2, use_container_width=True, key="rel_scatter_chart",
                    on_select="rerun", selection_mode=["points", "box", "lasso"],
                )

                selected_points = event.selection.points if event and event.selection else []
                selected_indices = [p["point_index"] for p in selected_points if p.get("curve_number") == 1]

                if groups:
                    sc1, sc2, sc3 = st.columns([2, 2, 1])
                    with sc1:
                        st.caption(f"{len(selected_indices)} point(s) selected (box/lasso/click on the chart).")
                    with sc2:
                        assign_choice = st.selectbox(
                            "Assign selection to", list(groups.keys()),
                            key="rel_assign_choice", label_visibility="collapsed",
                        )
                    with sc3:
                        if st.button("Assign", key="rel_assign_btn", disabled=not selected_indices):
                            for i in selected_indices:
                                assignments[merged.index[i].isoformat()] = assign_choice
                            st.rerun()
                    if assignments and st.button("Clear all group assignments", key="rel_clear_assignments"):
                        assignments.clear()
                        st.rerun()
                elif selected_indices:
                    st.caption(f"{len(selected_indices)} point(s) selected — define a group above to assign them.")

                st.markdown("**Or group by time**")
                data_min, data_max = merged.index.min().date(), merged.index.max().date()
                range_start, range_end = st.slider(
                    "Date range",
                    min_value=data_min, max_value=data_max, value=(data_min, data_max),
                    key="rel_time_group_range",
                )
                tc3, tc4 = st.columns([2, 1])
                with tc3:
                    range_group = st.selectbox(
                        "Assign range to group", list(groups.keys()),
                        key="rel_time_group_choice", disabled=not groups,
                    )
                with tc4:
                    st.write("")
                    if st.button("Assign range", key="rel_time_group_assign", disabled=not groups):
                        for ts in merged.index:
                            if range_start <= ts.date() <= range_end:
                                assignments[ts.isoformat()] = range_group
                        st.rerun()

                ac1, ac2 = st.columns([2, 1])
                with ac1:
                    auto_period = st.selectbox(
                        "...or auto-create one group per", ["Year", "Decade"], key="rel_auto_period"
                    )
                with ac2:
                    st.write("")
                    if st.button("Auto-group every point", key="rel_auto_group_btn"):
                        for ts in merged.index:
                            label = str(ts.year) if auto_period == "Year" else f"{(ts.year // 10) * 10}s"
                            if label not in groups:
                                groups[label] = {
                                    "color": DEFAULT_PALETTE[len(groups) % len(DEFAULT_PALETTE)],
                                    "shape": "Circle",
                                }
                            assignments[ts.isoformat()] = label
                        st.rerun()

                rel_date_bounds = (merged.index.min().date(), merged.index.max().date())

                with st.expander("Export chart as image (interactive style)"):
                    export_show_regression = synced_checkbox(
                        "Include regression lines in this export", show_regression, "rel_export_show_regression",
                    )
                    export_chart_section(
                        lambda s, e: build_relationship_figure(
                            slice_by_date(merged, s, e), groups, assignments, rel_line_color, rel_colorscale,
                            rel_latest_color, x_label, y_label_name, x_axis_label, y_axis_label,
                            rel_bg_color, rel_text_color, show_regression=export_show_regression,
                            regression_groups=regression_groups,
                        ),
                        "rel_export", "fred_relationship", date_bounds=rel_date_bounds,
                    )

                with st.expander("Publication-style export (matplotlib)"):
                    highlight_latest = st.checkbox("Highlight latest point", value=True, key="rel_pub_highlight")
                    show_label = st.checkbox("Show date label on latest point", value=True, key="rel_pub_label")
                    pub_show_regression = synced_checkbox(
                        "Include regression lines in this export", show_regression, "rel_pub_show_regression",
                    )

                    def build_fn(s, e):
                        sub_merged = slice_by_date(merged, s, e)
                        sub_point_groups = [assignments.get(ts.isoformat()) for ts in sub_merged.index]
                        return build_relationship_publication_fig(
                            sub_merged, x_label, y_label_name, rel_point_color, rel_latest_color,
                            highlight_latest, show_label, point_groups=sub_point_groups, groups=groups,
                            show_regression=pub_show_regression, regression_groups=regression_groups,
                        )

                    publication_export_section(
                        "rel_pub", "fred_relationship_publication", build_fn,
                        needs_recession=False, date_bounds=rel_date_bounds,
                    )

                st.download_button(
                    "Download CSV",
                    merged.rename(columns={"x": x_label, "y": y_label_name}).to_csv().encode("utf-8"),
                    file_name="fred_relationship.csv",
                    mime="text/csv",
                )
