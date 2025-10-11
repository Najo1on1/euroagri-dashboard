# pages/2_🌡️_Climate.py
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st

# Plotly first, fallback to Matplotlib if needed
PLOTLY_OK = True
try:
    import plotly.express as px
except Exception:
    PLOTLY_OK = False
import matplotlib.pyplot as plt

# -----------------------------
# Streamlit page config
# -----------------------------
st.set_page_config(page_title="Climate", page_icon="🌡️", layout="wide")
st.title("🌡️ Climate — Monthly Windows & Derivatives")

# -----------------------------
# Repo-local paths ONLY
# -----------------------------
# This file lives in repo_root/pages/, so repo root is one parent up.
REPO_ROOT   = Path(__file__).resolve().parents[1]
DATA_SAMPLE = REPO_ROOT / "data" / "sample"
CLI_SAMPLE  = DATA_SAMPLE / "climate_windows.parquet"

# -----------------------------
# Helpers
# -----------------------------
def canon(df: pd.DataFrame) -> pd.DataFrame:
    """Light canonicalization for keys/dtypes."""
    if df.empty:
        return df
    out = df.copy()
    if "region_iso" in out.columns:
        out["region_iso"] = out["region_iso"].astype(str).str.strip()
    if "country" in out.columns:
        out["country"]   = out["country"].astype(str).str.strip()
    if "year" in out.columns:
        out["year"]  = pd.to_numeric(out["year"], errors="coerce").astype("Int64")
    if "month" in out.columns:
        out["month"] = pd.to_numeric(out["month"], errors="coerce").astype("Int64")
    # numeric climate metrics (only if present)
    for c in ["t2m_C", "d2m_C", "tp_mm", "ssrd_MJm2", "ssr_MJm2", "gdd_base_5", "heat_hours_gt30C"]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out

def existing_metrics(df: pd.DataFrame) -> list[str]:
    candidates = [
        "t2m_C",          # mean temp °C
        "tp_mm",          # precip mm
        "d2m_C",          # dewpoint °C
        "ssrd_MJm2",      # solar down MJ m-2
        "ssr_MJm2",       # solar net MJ m-2
        "gdd_base_5",     # growing degree days
        "heat_hours_gt30C"
    ]
    return [c for c in candidates if c in df.columns]

def aggregate_monthly(df: pd.DataFrame, all_years: bool, years: list[int] | None, metric_cols: list[str]) -> pd.DataFrame:
    """Return tidy long df with columns: region_iso, month, metric, value (optionally across selected years)."""
    if df.empty or not metric_cols:
        return pd.DataFrame(columns=["region_iso", "month", "metric", "value"])
    base = df.copy()
    # filter by selected years (if not aggregating across all years)
    if not all_years:
        if years:
            base = base[base["year"].isin(years)]
    # group by region & month, average metrics
    grp = (base.groupby(["region_iso", "month"], as_index=False)[metric_cols]
                .mean(numeric_only=True))
    # long format for plotting
    long = grp.melt(id_vars=["region_iso", "month"], var_name="metric", value_name="value")
    # sort by metric, region, month
    long = long.sort_values(["metric", "region_iso", "month"])
    return long

def load_parquet_or_uploaded(default_path: Path, label: str) -> tuple[pd.DataFrame, str]:
    """
    1) If user uploads a parquet, use it for this session.
    2) Else load repo sample from data/sample/.
    3) Else stop with friendly error.
    Returns (df, source_str).
    """
    uploaded = st.file_uploader(f"Upload {label} (.parquet)", type=["parquet"])
    if uploaded is not None:
        try:
            df = pd.read_parquet(uploaded)
            return df, "uploaded file"
        except Exception as e:
            st.error(f"Could not read uploaded file: {e}")
            st.stop()

    if default_path.exists():
        try:
            df = pd.read_parquet(default_path)
            return df, f"{default_path.relative_to(REPO_ROOT)}"
        except Exception as e:
            st.error(f"Failed to read sample file at {default_path}: {e}")
            st.stop()

    st.error(f"Missing sample file: {default_path}. Please upload a Parquet.")
    st.stop()

def plot_plotly(long_df: pd.DataFrame, metric: str, regions: list[str]):
    sub = long_df[long_df["metric"] == metric].copy()
    if sub.empty:
        st.warning(f"No data for {metric}.")
        return
    fig = px.line(
        sub[sub["region_iso"].isin(regions)],
        x="month", y="value", color="region_iso",
        markers=True, title=f"{metric} — monthly profile"
    )
    fig.update_layout(
        legend_title_text="Region",
        xaxis=dict(dtick=1),
        margin=dict(l=10, r=10, t=40, b=10),
        height=420,
    )
    cfg = {
        "displaylogo": False,
        "modeBarButtonsToRemove": ["select2d", "lasso2d"],
        "toImageButtonOptions": {"format": "png", "filename": f"climate_{metric}"}
    }
    st.plotly_chart(fig, config=cfg, width="stretch")

def plot_matplotlib(long_df: pd.DataFrame, metric: str, regions: list[str]):
    sub = long_df[long_df["metric"] == metric].copy()
    if sub.empty:
        st.warning(f"No data for {metric}.")
        return
    fig, ax = plt.subplots(figsize=(8, 4))
    for reg, g in sub[sub["region_iso"].isin(regions)].groupby("region_iso"):
        ax.plot(g["month"], g["value"], marker="o", label=str(reg))
    ax.set_title(f"{metric} — monthly profile")
    ax.set_xlabel("Month")
    ax.set_ylabel(metric)
    ax.set_xticks(range(1, 13))
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=9)
    st.pyplot(fig, clear_figure=True)

# -----------------------------
# Load data (uploader → sample fallback), then canonicalize
# -----------------------------
with st.sidebar:
    st.subheader("Data source")

cli_raw, src_str = load_parquet_or_uploaded(CLI_SAMPLE, "climate windows")
cli = canon(cli_raw)

with st.sidebar:
    st.code(str(src_str))

# -----------------------------
# Sidebar filters
# -----------------------------
with st.sidebar:
    st.header("Filters")
    if cli.empty:
        st.error("Climate file is empty after loading.")
        st.stop()

    # Regions
    regions = sorted(cli["region_iso"].dropna().unique().tolist()) if "region_iso" in cli.columns else []
    default_regions = regions[:3] if len(regions) >= 3 else regions
    sel_regions = st.multiselect("Regions", options=regions, default=default_regions)

    # Years
    years = sorted([int(y) for y in cli["year"].dropna().unique().tolist()]) if "year" in cli.columns else []
    all_years_toggle = st.toggle("Aggregate across all years (monthly mean)", value=True, help="If off, select specific years below.")
    sel_years = []
    if not all_years_toggle and years:
        sel_years = st.multiselect("Years", options=years, default=years)

    # Metrics
    avail_metrics = existing_metrics(cli)
    pretty = {
        "t2m_C": "T2M (°C)",
        "tp_mm": "Precip (mm)",
        "d2m_C": "Dewpoint (°C)",
        "ssrd_MJm2": "Solar down (MJ m⁻²)",
        "ssr_MJm2": "Solar net (MJ m⁻²)",
        "gdd_base_5": "GDD base 5 (°C·d)",
        "heat_hours_gt30C": "Heat hours >30°C",
    }
    options_labels = [pretty.get(m, m) for m in avail_metrics]
    label_to_metric = {pretty.get(m, m): m for m in avail_metrics}
    default_metrics = [pretty.get("t2m_C", "t2m_C"), pretty.get("tp_mm", "tp_mm")] if avail_metrics else []
    sel_metric_labels = st.multiselect("Metrics", options=options_labels, default=default_metrics, help="Only existing columns are shown.")
    sel_metrics = [label_to_metric[lbl] for lbl in sel_metric_labels]

st.caption(
    f"Loaded climate rows: **{len(cli):,}** "
    f"| regions: **{len(set(cli['region_iso'])) if 'region_iso' in cli.columns else 0}**"
)

if cli.empty:
    st.stop()
if not sel_regions:
    st.warning("Select at least one region to visualize.")
    st.stop()
if not sel_metrics:
    st.warning("Select at least one metric to visualize.")
    st.stop()

# -----------------------------
# Build tidy dataset for plotting
# -----------------------------
tidy = aggregate_monthly(cli, all_years=all_years_toggle, years=sel_years, metric_cols=sel_metrics)

# Download button for the filtered/aggregated data
csv_bytes = tidy.to_csv(index=False).encode("utf-8")
st.download_button(
    "⬇️ Download CSV (current view)",
    data=csv_bytes,
    file_name="climate_monthly_view.csv",
    mime="text/csv",
)

# -----------------------------
# Render charts
# -----------------------------
for metric in sel_metrics:
    try:
        if PLOTLY_OK:
            plot_plotly(tidy, metric, sel_regions)
        else:
            plot_matplotlib(tidy, metric, sel_regions)
    except Exception as e:
        st.warning(f"Plotly failed for {metric} ({e}); falling back to Matplotlib.")
        plot_matplotlib(tidy, metric, sel_regions)

# Extra context
with st.expander("What am I looking at?"):
    st.markdown("""
- **Lines** show monthly averages for each selected region and metric.
- If **“Aggregate across all years”** is on, values are averaged over all years.
- Toggle it off to pick one or more **specific years**.
- Metrics list is **auto-detected** from your climate parquet and hides missing columns.
- We render with **Plotly** (hover/zoom/export). If it fails, we **fallback to Matplotlib** automatically.
    """)
