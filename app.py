"""
Interactive FRED data explorer.

Run with:
    streamlit run app.py

Requires a FRED API key already stored via `python setup_api_key.py`
(or set as the FRED_API_KEY environment variable).
"""

import datetime
import io
import re

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from fred_data import FredDatasets, get_api_key, get_recession_series, shade_recessions

st.set_page_config(page_title="FRED Data Explorer", layout="wide")

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

st.title("FRED Data Explorer")

with st.sidebar:
    st.text_input(
        "Your own FRED API key (optional)",
        type="password",
        key="user_api_key",
        help=(
            "Paste your own free FRED API key to use it for this session instead of the "
            "key this app is configured with. Get one at "
            "https://fred.stlouisfed.org/docs/api/api_key.html"
        ),
    )
    st.divider()


def resolve_api_key() -> str | None:
    """The key to actually use: one typed into the sidebar for this session, else the
    app's own configured key (Streamlit secrets / env var / local keyring)."""
    typed = st.session_state.get("user_api_key", "").strip()
    return typed or get_api_key()


if not resolve_api_key():
    st.error(
        "No FRED API key found. Paste your own free key in the sidebar (get one at "
        "https://fred.stlouisfed.org/docs/api/api_key.html), or, if you're running this "
        "locally, run `python setup_api_key.py` in this folder once and reload."
    )
    st.stop()


@st.cache_data(show_spinner="Fetching from FRED...")
def fetch(series_ids: tuple[str, ...], start: str, end: str, _api_key: str) -> pd.DataFrame:
    fd = FredDatasets(api_key=_api_key)
    for sid in series_ids:
        label = DEFAULT_SERIES.get(sid, sid)
        fd.add_series(sid, label, start=start, end=end)
    return fd.as_dataframe()


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
    return DEFAULT_SERIES.get(sid, sid)


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
        result = base_df[DEFAULT_SERIES.get(sid, sid)]  # base_df's own column key is never renamed
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

    base_df = fetch(tuple(sorted(base_needed)), start, end, resolve_api_key())

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
        fig.add_trace(go.Scatter(x=df.index, y=df[col], mode="lines", name=col, line=dict(color=label_colors.get(col))))
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
                recession_series = fetch_recessions(resolve_api_key())
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

tab_series, tab_relationship = st.tabs(["Time Series", "Relationship (X vs Y)"])

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
        series_ids = tuple(dict.fromkeys(chosen_labels + custom_ids))  # dedupe, keep order

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
