# pages/1_🌾_Recommendations.py
import os
from pathlib import Path
import pandas as pd
import numpy as np
import streamlit as st

st.set_page_config(page_title="🌾 Recommendations", page_icon="🌾", layout="wide")

st.title("🌾 Recommendations")
st.caption("EuroAgri-Advisor Live — variety + irrigation + nutrients, with disease awareness")

# ------------------------------------------------------------
# Paths (WSL D: preferred, fallback to HOME)
# ------------------------------------------------------------
WSL_D = Path("/mnt/d/Colab/Ecosystem/Deep RL and LLM/Agri")
HOME  = Path("/home/najo1o11/euroagri-advisor")

PROJECT_ROOT = WSL_D if (WSL_D / "data").exists() else HOME
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
RECS_PARQ_PRIMARY  = DATA_PROCESSED / "recommendations.parquet"
RECS_PARQ_FALLBACK = HOME / "data" / "processed" / "recommendations.parquet"

# The exact columns you want to display (order preserved)
DISPLAY_COLS = [
    "region_iso", "year", "crop", "variety",
    "plan_score", "variety_fit", "irrigation_fit", "nplan_fit",
    "mean_t_ha",
    "disease", "disease_risk_annual", "disease_penalty", "disease_fit",
]

@st.cache_data(show_spinner=False)
def load_recs() -> pd.DataFrame:
    # try primary, then fallback, else empty
    if RECS_PARQ_PRIMARY.exists():
        df = pd.read_parquet(RECS_PARQ_PRIMARY)
        src = str(RECS_PARQ_PRIMARY)
    elif RECS_PARQ_FALLBACK.exists():
        df = pd.read_parquet(RECS_PARQ_FALLBACK)
        src = str(RECS_PARQ_FALLBACK)
    else:
        return pd.DataFrame(), "<missing>"

    # Canonicalize minimal keys/dtypes
    if "region_iso" in df.columns:
        df["region_iso"] = df["region_iso"].astype(str).str.strip()
    if "crop" in df.columns:
        df["crop"] = df["crop"].astype(str).str.lower().str.strip()
    if "year" in df.columns:
        df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")

    return df, src

df, src = load_recs()

with st.sidebar:
    st.subheader("Data source")
    st.code(src)

# ------------------------------------------------------------
# Guard: empty or missing file
# ------------------------------------------------------------
if df.empty:
    st.error("No recommendations found. Make sure you've run the pipeline to produce `data/processed/recommendations.parquet`.")
    st.stop()

# ------------------------------------------------------------
# Safe-guard: ensure all display columns exist (create empty if missing)
# ------------------------------------------------------------
missing_cols = [c for c in DISPLAY_COLS if c not in df.columns]
if missing_cols:
    for c in missing_cols:
        df[c] = pd.NA
    with st.sidebar:
        st.warning(
            "Some expected columns were missing and were added as empty: "
            + ", ".join(missing_cols)
        )

# ------------------------------------------------------------
# Sidebar filters (Region + Year, with guards)
# ------------------------------------------------------------
with st.sidebar:
    st.subheader("Filters")

    # Region filter
    regions = sorted(df["region_iso"].dropna().unique()) if "region_iso" in df.columns else []
    selected_regions = st.multiselect(
        "Region(s)", options=regions, default=regions[: min(5, len(regions))] if regions else []
    )

    # Year filter (use only if present and not all NA)
    if "year" in df.columns and df["year"].notna().any():
        years = sorted(df["year"].dropna().unique().tolist())
        default_years = years[-1:]  # default to latest year if possible
        selected_years = st.multiselect("Year(s)", options=years, default=default_years)
    else:
        selected_years = None

# Apply filters
flt = df.copy()
if selected_regions:
    flt = flt[flt["region_iso"].isin(selected_regions)]
if selected_years is not None and len(selected_years) > 0:
    flt = flt[flt["year"].isin(selected_years)]

if flt.empty:
    st.warning("No rows after applying filters. Try broadening your selection.")
    st.stop()

# ------------------------------------------------------------
# Main table (only requested columns)
# ------------------------------------------------------------
# Reorder and keep only the desired columns safely
safe_cols = [c for c in DISPLAY_COLS if c in flt.columns]
table = flt[safe_cols].copy()

# Nice formatting for scores (keep data types robust)
for col in ["plan_score", "variety_fit", "irrigation_fit", "nplan_fit",
            "disease_risk_annual", "disease_penalty", "disease_fit"]:
    if col in table.columns:
        table[col] = pd.to_numeric(table[col], errors="coerce").round(3)

st.subheader("Filtered recommendations")
st.dataframe(table, use_container_width=True, hide_index=True)

# ------------------------------------------------------------
# Small summary (safe-guards)
# ------------------------------------------------------------
with st.expander("Summary"):
    cols_present = [c for c in ["plan_score","variety_fit","irrigation_fit","nplan_fit","disease_fit"] if c in flt.columns]
    if cols_present:
        desc = flt[cols_present].apply(pd.to_numeric, errors="coerce").describe().T
        st.dataframe(desc, use_container_width=True)
    else:
        st.info("No numeric score columns available for summary.")

# ------------------------------------------------------------
# Download CSV of what’s shown
# ------------------------------------------------------------
csv_bytes = table.to_csv(index=False).encode("utf-8")
st.download_button(
    label="⬇️ Download filtered table (CSV)",
    data=csv_bytes,
    file_name="recommendations_filtered.csv",
    mime="text/csv",
)