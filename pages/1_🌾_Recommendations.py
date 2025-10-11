# pages/1_🌾_Recommendations.py
import io
from pathlib import Path
import pandas as pd
import numpy as np
import streamlit as st

st.set_page_config(page_title="🌾 Recommendations", page_icon="🌾", layout="wide")

st.title("🌾 Recommendations")
st.caption("Climate aware adaptive crop planning for EU regions — variety + irrigation + nutrients, with disease awareness")

# ------------------------------------------------------------------------------
# Repo-local paths only (no D:/HOME fallbacks)
# ------------------------------------------------------------------------------
# This file lives in repo_root/pages/, so the repo root is one parent up.
REPO_ROOT   = Path(__file__).resolve().parents[1]
DATA_SAMPLE = REPO_ROOT / "data" / "sample"
RECS_SAMPLE = DATA_SAMPLE / "recommendations.parquet"

# Columns to display (in this order, if present)
DISPLAY_COLS = [
    "region_iso", "year", "crop", "variety",
    "plan_score", "variety_fit", "irrigation_fit", "nplan_fit",
    "mean_t_ha",
    "disease", "disease_risk_annual", "disease_penalty", "disease_fit",
]

# ------------------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------------------
def canon(df: pd.DataFrame) -> pd.DataFrame:
    """Light canonicalization for keys/dtypes."""
    if df.empty:
        return df
    out = df.copy()
    if "region_iso" in out.columns:
        out["region_iso"] = out["region_iso"].astype(str).str.strip()
    if "crop" in out.columns:
        out["crop"] = out["crop"].astype(str).str.lower().str.strip()
    if "year" in out.columns:
        out["year"] = pd.to_numeric(out["year"], errors="coerce").astype("Int64")
    # numeric scoring cols if present
    for col in ["plan_score","variety_fit","irrigation_fit","nplan_fit",
                "disease_risk_annual","disease_penalty","disease_fit","mean_t_ha"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out

@st.cache_data(show_spinner=False)
def load_parquet_local(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path)

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
            df = load_parquet_local(default_path)
            return df, f"{default_path.relative_to(REPO_ROOT)}"
        except Exception as e:
            st.error(f"Failed to read sample file at {default_path}: {e}")
            st.stop()

    st.error(f"Missing sample file: {default_path}. Please upload a Parquet.")
    st.stop()

# ------------------------------------------------------------------------------
# Load data (uploader → sample fallback), then canonicalize
# ------------------------------------------------------------------------------
with st.sidebar:
    st.subheader("Data source")

recs_raw, src_str = load_parquet_or_uploaded(RECS_SAMPLE, "recommendations")
df = canon(recs_raw)

with st.sidebar:
    st.code(str(src_str))

# ------------------------------------------------------------------------------
# Guard: ensure expected columns present (create empty if missing)
# ------------------------------------------------------------------------------
missing_cols = [c for c in DISPLAY_COLS if c not in df.columns]
if missing_cols:
    for c in missing_cols:
        df[c] = pd.NA
    with st.sidebar:
        st.warning("Some expected columns were missing and were added as empty: " + ", ".join(missing_cols))

if df.empty:
    st.error("No recommendations found in the provided file.")
    st.stop()

# ------------------------------------------------------------------------------
# Sidebar filters
# ------------------------------------------------------------------------------
with st.sidebar:
    st.subheader("Filters")

    regions = sorted(df["region_iso"].dropna().unique()) if "region_iso" in df.columns else []
    selected_regions = st.multiselect(
        "Region(s)", options=regions, default=regions[: min(5, len(regions))] if regions else []
    )

    if "year" in df.columns and df["year"].notna().any():
        years = sorted([int(y) for y in df["year"].dropna().unique().tolist()])
        default_years = years[-1:] if years else []
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

# ------------------------------------------------------------------------------
# Main table (only requested columns), with modern width API
# ------------------------------------------------------------------------------
safe_cols = [c for c in DISPLAY_COLS if c in flt.columns]
table = flt[safe_cols].copy()

# Pretty numeric rounding for display
for col in ["plan_score", "variety_fit", "irrigation_fit", "nplan_fit",
            "disease_risk_annual", "disease_penalty", "disease_fit", "mean_t_ha"]:
    if col in table.columns:
        table[col] = pd.to_numeric(table[col], errors="coerce").round(3)

st.subheader("Filtered recommendations")
st.dataframe(table, width="stretch", hide_index=True)

# ------------------------------------------------------------------------------
# Summary
# ------------------------------------------------------------------------------
with st.expander("Summary"):
    cols_present = [c for c in ["plan_score","variety_fit","irrigation_fit","nplan_fit","disease_fit"] if c in flt.columns]
    if cols_present:
        desc = flt[cols_present].apply(pd.to_numeric, errors="coerce").describe().T
        st.dataframe(desc, width="stretch", hide_index=True)
    else:
        st.info("No numeric score columns available for summary.")

# ------------------------------------------------------------------------------
# Download CSV of what’s shown
# ------------------------------------------------------------------------------
csv_bytes = table.to_csv(index=False).encode("utf-8")
st.download_button(
    label="⬇️ Download filtered table (CSV)",
    data=csv_bytes,
    file_name="recommendations_filtered.csv",
    mime="text/csv",
)
