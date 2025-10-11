# pages/4_🦠_Disease.py
import io
import logging
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st

# -------------------------------
# Quiet common deprecation chatter in-app (best effort)
# -------------------------------
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=r".*deprecated.*Plotly.*", category=UserWarning)
warnings.filterwarnings("ignore", message=r".*use_container_width.*", category=UserWarning)
for name in ("streamlit", "streamlit.runtime", "streamlit.web", "streamlit.logger"):
    logging.getLogger(name).setLevel(logging.ERROR)
logging.getLogger("plotly").setLevel(logging.ERROR)

# Plotly first (with fallback to Matplotlib)
try:
    import plotly.graph_objects as go
    PLOTLY_OK = True
except Exception:
    PLOTLY_OK = False
import matplotlib.pyplot as plt

# -------------------------------
# Repo-local paths ONLY (no D:/HOME fallbacks)
# -------------------------------
# This file lives in repo_root/pages/, so repo root is one parent up.
REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_SAMPLE = REPO_ROOT / "data" / "sample"
DISEASE_SAMPLE = DATA_SAMPLE / "disease_risk.parquet"

# -------------------------------
# Small utils
# -------------------------------
def load_parquet_or_uploaded(default_path: Path, label: str) -> tuple[pd.DataFrame, str]:
    """
    1) If a user uploads a parquet, use it for this session.
    2) Else read the repo sample from data/sample/.
    3) Else stop with a friendly error.
    Returns (df, source_str).
    """
    uploaded = st.file_uploader(f"Upload {label} (.parquet)", type=["parquet"], key=f"uploader_{label}")
    if uploaded is not None:
        try:
            df = pd.read_parquet(uploaded)
            return df, "uploaded file"
        except Exception as e:
            st.error(f"Could not read uploaded {label}: {e}")
            st.stop()

    if default_path.exists():
        try:
            df = pd.read_parquet(default_path)
            return df, str(default_path.relative_to(REPO_ROOT))
        except Exception as e:
            st.error(f"Failed to read sample {label} at {default_path}: {e}")
            st.stop()

    st.error(f"Missing sample {label}: {default_path}. Please upload a Parquet.")
    st.stop()

def canon(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    for col in ("region_iso", "crop", "disease"):
        if col in out.columns:
            out[col] = out[col].astype(str).str.strip()
            if col == "crop":
                out[col] = out[col].str.lower()
    for col in ("year", "month"):
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce").astype("Int64")
    if "risk_0_1" in out.columns:
        out["risk_0_1"] = pd.to_numeric(out["risk_0_1"], errors="coerce")
    return out

def disease_map_for_crop(crop: str) -> list[str]:
    mapping = {
        "wheat":    ["wheat_rust"],
        "potato":   ["potato_blight"],
        "barley":   ["powdery_mildew"],
        "maize":    ["powdery_mildew"],
        "rapeseed": ["powdery_mildew"],
        "sugarbeet":["powdery_mildew"],
        "beans":    ["powdery_mildew"],
        "peas":     ["powdery_mildew"],
    }
    return mapping.get(crop, [])

# Distinct color cycle (used for Matplotlib & as backup for Plotly)
COLOR_CYCLE = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
    "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
    "#bcbd22", "#17becf"
]

# -------------------------------
# Page
# -------------------------------
st.set_page_config(page_title="Disease Risk", page_icon="🦠", layout="wide")
st.title("🦠 Disease Risk")

# Load & canonicalize via uploader → sample
with st.sidebar:
    st.subheader("Data source")
risk_raw, src_str = load_parquet_or_uploaded(DISEASE_SAMPLE, "disease risk")
risk = canon(risk_raw)
with st.sidebar:
    st.code(f"source: {src_str}")

# Verify required columns
needed = {"region_iso", "year", "month", "crop", "disease", "risk_0_1"}
missing = needed - set(risk.columns)
if missing:
    st.error(f"Missing required columns: {sorted(missing)}")
    st.stop()

# -------------------------------
# Sidebar filters
# -------------------------------
with st.sidebar:
    st.subheader("Filters")

    regions = sorted(risk["region_iso"].dropna().unique().tolist())
    crops   = sorted(risk["crop"].dropna().unique().tolist())
    years   = sorted([int(y) for y in risk["year"].dropna().unique().tolist()])

    sel_regions = st.multiselect("Regions", options=regions, default=regions[:3])
    sel_crops   = st.multiselect("Crops",   options=crops,   default=["wheat"] if "wheat" in crops else (crops[:1] if crops else []))

    # Year selection + average toggle
    sel_years   = st.multiselect("Years (each will be a separate line)", options=years, default=[])
    show_avg    = st.checkbox("Show 'All years (avg)' line when no years selected (or in addition)", value=True)

    st.divider()
    relevance_only = st.checkbox("Show only crop-relevant diseases", value=True)
    st.caption("You can refine diseases after 'Apply' below.")

# Apply base filtering (region/crop)
base = risk[
    risk["region_iso"].isin(sel_regions) &
    risk["crop"].isin(sel_crops)
].copy()

if base.empty:
    st.info("No rows for the selected region/crop filters.")
    st.stop()

# Disease subset based on relevance
if relevance_only:
    allowed = set()
    for c in sel_crops:
        allowed.update(disease_map_for_crop(c))
    present = set(base["disease"].unique())
    allowed = [d for d in allowed if d in present]
    if not allowed:
        allowed = sorted(present)
    base = base[base["disease"].isin(allowed)]
    initial_diseases = allowed
else:
    initial_diseases = sorted(base["disease"].unique())

# Let the user refine diseases
with st.sidebar:
    sel_diseases = st.multiselect("Diseases", options=sorted(base["disease"].unique()), default=initial_diseases)
    st.caption("Tip: deselect to compare a subset.")
    st.divider()

# Final filtered slice
f = base[base["disease"].isin(sel_diseases)].copy()

if f.empty:
    st.info("No rows after applying disease selection.")
    st.stop()

# -------------------------------
# Helpers for aggregation
# -------------------------------
def monthly_avg(df: pd.DataFrame) -> pd.DataFrame:
    """Average across years by month for a given (region, crop, disease)."""
    return (df.groupby(["month"], as_index=False)["risk_0_1"]
              .mean()
              .assign(series="All years (avg)"))

def monthly_year(df: pd.DataFrame, year: int) -> pd.DataFrame:
    """Monthly series for a single year."""
    sub = df[df["year"] == year]
    if sub.empty:
        return pd.DataFrame(columns=["month", "risk_0_1", "series"])
    return (sub.groupby(["month"], as_index=False)["risk_0_1"]
               .mean()
               .assign(series=str(year)))

# -------------------------------
# Plot area
# -------------------------------
st.markdown("### Monthly risk profiles")

# One section per (region, crop)
for reg in sel_regions:
    for crop in sel_crops:
        slice_rc = f[(f["region_iso"] == reg) & (f["crop"] == crop)]
        if slice_rc.empty:
            continue

        st.markdown(f"**Region:** `{reg}` &nbsp;&nbsp; **Crop:** `{crop}`")
        cols = st.columns(max(1, min(3, len(sel_diseases))))  # up to 3 plots per row

        for i, disease in enumerate(sel_diseases):
            sub = slice_rc[slice_rc["disease"] == disease]
            if sub.empty:
                continue

            # Build series: avg and/or selected years
            series_frames = []
            if sel_years:
                if show_avg:
                    series_frames.append(monthly_avg(sub))
                for y in sel_years:
                    series_frames.append(monthly_year(sub, y))
            else:
                if show_avg:
                    series_frames.append(monthly_avg(sub))

            if not series_frames:
                with cols[i % len(cols)]:
                    st.info(f"No series selected for **{disease}**.")
                continue

            to_plot = pd.concat(series_frames, ignore_index=True).sort_values(["series", "month"])
            title = f"{disease} — risk (0–1)"

            # Try Plotly first
            plot_ok = False
            if PLOTLY_OK:
                try:
                    fig = go.Figure()
                    series_list = list(to_plot["series"].unique())
                    for idx, sname in enumerate(series_list):
                        sdata = to_plot[to_plot["series"] == sname]
                        fig.add_trace(
                            go.Scatter(
                                x=sdata["month"],
                                y=sdata["risk_0_1"],
                                mode="lines+markers",
                                name=str(sname),
                                line=dict(color=COLOR_CYCLE[idx % len(COLOR_CYCLE)], width=2),
                                marker=dict(size=6),
                                hovertemplate="Month %{x}<br>Risk %{y:.2f}<extra>" + str(sname) + "</extra>",
                            )
                        )
                    fig.update_layout(
                        title=title,
                        xaxis_title="Month",
                        yaxis_title="risk_0_1",
                        template="plotly_white",
                        legend_title_text="Series",
                        margin=dict(l=10, r=10, t=40, b=10),
                        height=360,
                    )
                    cfg = {
                        "displaylogo": False,
                        "modeBarButtonsToRemove": ["select2d", "lasso2d"],
                        "toImageButtonOptions": {"format": "png", "filename": f"{reg}_{crop}_{disease}_risk"}
                    }
                    with cols[i % len(cols)]:
                        st.plotly_chart(fig, config=cfg, width="stretch")
                    plot_ok = True
                except Exception:
                    plot_ok = False

            # Fallback to Matplotlib
            if not plot_ok:
                with cols[i % len(cols)]:
                    fig, ax = plt.subplots(figsize=(6, 3))
                    series_list = list(to_plot["series"].unique())
                    for idx, sname in enumerate(series_list):
                        sdata = to_plot[to_plot["series"] == sname]
                        ax.plot(
                            sdata["month"].values,
                            sdata["risk_0_1"].values,
                            "-o",
                            label=str(sname),
                            color=COLOR_CYCLE[idx % len(COLOR_CYCLE)],
                        )
                    ax.set_title(title)
                    ax.set_xlabel("Month")
                    ax.set_ylabel("risk_0_1")
                    ax.set_ylim(0, 1)
                    ax.grid(True, alpha=0.3)
                    ax.legend(loc="best")
                    st.pyplot(fig, width="stretch")

# -------------------------------
# Download the filtered slice
# -------------------------------
st.divider()
st.markdown("#### Download filtered slice")
csv_buf = io.StringIO()
export_cols = ["region_iso", "year", "month", "crop", "disease", "risk_0_1"]
f[export_cols].sort_values(["region_iso", "crop", "disease", "year", "month"]).to_csv(csv_buf, index=False)
st.download_button(
    label="Download CSV",
    data=csv_buf.getvalue(),
    file_name="disease_risk_filtered.csv",
    mime="text/csv",
)
