# pages/3_🌍_Soils_Overview.py
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st

# Prefer Plotly; fall back to Matplotlib if needed
_PLOTLY_OK = True
try:
    import plotly.express as px
except Exception:
    _PLOTLY_OK = False
import matplotlib.pyplot as plt

st.set_page_config(page_title="Soils Overview", page_icon="🌍", layout="wide")
st.title("🌍 Soils Overview")

# --------------------------------------------------------------------------------------
# Repo-local paths ONLY (no D:/HOME fallbacks)
# --------------------------------------------------------------------------------------
# This file lives in repo_root/pages/, so repo root is one parent up.
REPO_ROOT   = Path(__file__).resolve().parents[1]
DATA_SAMPLE = REPO_ROOT / "data" / "sample"
SOILS_SAMPLE = DATA_SAMPLE / "soils.parquet"
DIAG_SAMPLE  = DATA_SAMPLE / "soils_v01_diagnostics.parquet"

# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------
def load_parquet_or_uploaded(default_path: Path, label: str) -> tuple[pd.DataFrame, str]:
    """
    1) If user uploads a parquet, use it for this session.
    2) Else load repo sample from data/sample/.
    3) Else stop with friendly error.
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
            # show path relative to repo root for clarity
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
    if "region_iso" in out.columns:
        out["region_iso"] = out["region_iso"].astype(str).str.strip()
    if "country" in out.columns:
        out["country"] = out["country"].astype(str).str.strip()
    if "texture_class" in out.columns:
        out["texture_class"] = out["texture_class"].astype(str).str.lower().str.strip()
    # numeric guards (only if present)
    for c in ["ph","om","n_kg_ha","p_kg_ha","k_kg_ha","awc_mm"]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out

# --------------------------------------------------------------------------------------
# Load data (uploader → sample fallback), then canonicalize
# --------------------------------------------------------------------------------------
with st.sidebar:
    st.subheader("Data sources")

soil_raw, soil_src = load_parquet_or_uploaded(SOILS_SAMPLE, "soils")
diag_raw, diag_src = load_parquet_or_uploaded(DIAG_SAMPLE, "soils diagnostics (optional)")
soil    = canon(soil_raw)
soil_dg = canon(diag_raw)

with st.sidebar:
    st.caption("Loaded files")
    st.code(f"soils: {soil_src}")
    st.code(f"diagnostics: {diag_src}")

# Top metrics
col1, col2, col3 = st.columns(3)
col1.metric("Soil rows", f"{len(soil):,}")
col2.metric("Diagnostics rows", f"{len(soil_dg):,}")
n_countries = soil["country"].nunique() if "country" in soil.columns else 0
col3.metric("Countries (soils)", f"{n_countries:,}")

if soil.empty:
    st.error("Soils frame is empty after loading.")
    st.stop()

# --------------------------------------------------------------------------------------
# Sidebar filters
# --------------------------------------------------------------------------------------
with st.sidebar:
    st.header("Filters")
    if "country" in soil.columns and soil["country"].notna().any():
        countries_all = sorted(soil["country"].dropna().unique().tolist())
        default = countries_all[: min(5, len(countries_all))]
        selected_countries = st.multiselect("Country (multi-select)", countries_all, default=default)
    else:
        selected_countries = []

# Apply country filter
if selected_countries and "country" in soil.columns:
    soil_f = soil[soil["country"].isin(selected_countries)].copy()
else:
    soil_f = soil.copy()

if not soil_dg.empty and "country" in soil_dg.columns and selected_countries:
    soil_dg_f = soil_dg[soil_dg["country"].isin(selected_countries)].copy()
else:
    soil_dg_f = soil_dg.copy()

# --------------------------------------------------------------------------------------
# Panel 1 — quality_flag distribution (overall + by country)
# --------------------------------------------------------------------------------------
st.subheader("Quality Flags")

if "quality_flag" not in soil_f.columns:
    st.warning("`quality_flag` not found in soils — skipping this panel.")
else:
    # Overall distribution
    dist = (
        soil_f["quality_flag"].value_counts(dropna=False)
        .rename_axis("quality_flag")
        .reset_index(name="count")
        .sort_values("quality_flag")
    )

    st.markdown("**Overall distribution**")
    try:
        if not _PLOTLY_OK:
            raise RuntimeError("Plotly unavailable")
        fig = px.bar(dist, x="quality_flag", y="count", text="count",
                     title="Soil quality_flag distribution (overall)")
        fig.update_traces(textposition="outside")
        fig.update_layout(yaxis_title="count", xaxis_title="quality_flag", margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig, config={"displaylogo": False}, width="stretch")
    except Exception:
        fig, ax = plt.subplots(figsize=(6, 4), dpi=140)
        ax.bar(dist["quality_flag"].astype(str), dist["count"])
        for i, v in enumerate(dist["count"].values):
            ax.text(i, v + max(dist["count"]) * 0.01, str(v), ha="center", va="bottom", fontsize=9)
        ax.set_title("Soil quality_flag distribution (overall)")
        ax.set_xlabel("quality_flag"); ax.set_ylabel("count")
        ax.grid(True, axis="y", alpha=0.3)
        st.pyplot(fig)

    # By-country stacked bars (if country column exists)
    if "country" in soil_f.columns and soil_f["country"].notna().any():
        st.markdown("**By country (stacked)**")
        byc = soil_f.groupby(["country", "quality_flag"]).size().reset_index(name="count")
        pivot = byc.pivot(index="country", columns="quality_flag", values="count").fillna(0)

        try:
            if not _PLOTLY_OK:
                raise RuntimeError("Plotly unavailable")
            melt = pivot.reset_index().melt(id_vars="country", var_name="quality_flag", value_name="count")
            fig = px.bar(
                melt, x="country", y="count", color="quality_flag", barmode="stack",
                title="Soil quality_flag by country (stacked counts)",
            )
            fig.update_layout(yaxis_title="count", xaxis_title="country", margin=dict(l=10, r=10, t=40, b=10))
            st.plotly_chart(fig, config={"displaylogo": False}, width="stretch")
        except Exception:
            fig, ax = plt.subplots(figsize=(9, 5), dpi=140)
            bottom = np.zeros(len(pivot))
            for qf in pivot.columns:
                ax.bar(pivot.index, pivot[qf].values, bottom=bottom, label=str(qf))
                bottom += pivot[qf].values
            ax.set_title("Soil quality_flag by country (stacked counts)")
            ax.set_xlabel("country"); ax.set_ylabel("count")
            ax.legend(title="quality_flag", loc="best")
            ax.grid(True, axis="y", alpha=0.3)
            st.pyplot(fig)
    else:
        st.info("No `country` column available to display by-country breakdown.")

# --------------------------------------------------------------------------------------
# Panel 2 — Share of npk_method == 'heuristic_v01' by country
# --------------------------------------------------------------------------------------
st.subheader("NPK Heuristic Usage by Country")

# Prefer diagnostics if it has npk_method; else fall back to soils (if present there)
npk_src = None
if ("npk_method" in soil_dg_f.columns) and not soil_dg_f.empty:
    npk_src = soil_dg_f.copy()
elif ("npk_method" in soil_f.columns) and not soil_f.empty:
    npk_src = soil_f.copy()

if npk_src is None:
    st.warning("`npk_method` not found (neither diagnostics nor soils) — skipping heuristic share panel.")
else:
    # Ensure country present
    if "country" not in npk_src.columns or npk_src["country"].isna().all():
        if "region_iso" in npk_src.columns:
            npk_src["country"] = npk_src["region_iso"].astype(str).str[:2]
        else:
            npk_src["country"] = "??"

    grp = (
        npk_src
        .assign(is_heur=npk_src["npk_method"].astype(str).eq("heuristic_v01"))
        .groupby("country", as_index=False)
        .agg(total=("npk_method", "size"), heur=("is_heur", "sum"))
    )
    if grp.empty:
        st.info("No rows available to compute heuristic share.")
    else:
        grp["share_heuristic"] = (grp["heur"] / grp["total"]).fillna(0.0)

        try:
            if not _PLOTLY_OK:
                raise RuntimeError("Plotly unavailable")
            d = grp.sort_values("share_heuristic", ascending=False)
            fig = px.bar(
                d, x="country", y="share_heuristic",
                title="Share of regions using npk_method = 'heuristic_v01' by country",
                labels={"share_heuristic": "share (0–1)"}
            )
            fig.update_yaxes(range=[0, 1])
            fig.update_layout(margin=dict(l=10, r=10, t=40, b=10))
            st.plotly_chart(fig, config={"displaylogo": False}, width="stretch")
        except Exception:
            fig, ax = plt.subplots(figsize=(9, 5), dpi=140)
            d = grp.sort_values("share_heuristic", ascending=False)
            ax.bar(d["country"], d["share_heuristic"])
            ax.set_ylim(0, 1)
            ax.set_title("Share of regions using npk_method = 'heuristic_v01' by country")
            ax.set_ylabel("share (0–1)")
            ax.grid(True, axis="y", alpha=0.3)
            st.pyplot(fig)

# --------------------------------------------------------------------------------------
# Optional raw tables (collapsible)
# --------------------------------------------------------------------------------------
with st.expander("Show raw soils table"):
    # Streamlit 1.40+: prefer width="stretch" instead of use_container_width
    st.dataframe(soil_f, width="stretch")

with st.expander("Show diagnostics table (if any)"):
    if soil_dg_f.empty:
        st.info("Diagnostics frame is empty.")
    else:
        st.dataframe(soil_dg_f, width="stretch")
