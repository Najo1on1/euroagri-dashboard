# pages/2_🌍_Soils_Overview.py
import os
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st

# Try Plotly first; we'll fall back to matplotlib if anything goes wrong
_PLOTLY_OK = True
try:
    import plotly.express as px
except Exception:
    _PLOTLY_OK = False

import matplotlib.pyplot as plt

st.set_page_config(page_title="Soils Overview", page_icon="🌍", layout="wide")
st.title("🌍 Soils Overview")

# --------------------------------------------------------------------------------------
# Paths (D: preferred, HOME fallback)
# --------------------------------------------------------------------------------------
WSL_D = Path("/mnt/d/Colab/Ecosystem/Deep RL and LLM/Agri")
HOME  = Path("/home/najo1o11/euroagri-advisor")

INTERIM_D = WSL_D / "data" / "interim"
INTERIM_H = HOME  / "data" / "interim"

SOILS_PARQ_D   = INTERIM_D / "soils.parquet"
SOILS_PARQ_H   = INTERIM_H / "soils.parquet"
SOILS_DIAG_D   = INTERIM_D / "soils_v01_diagnostics.parquet"
SOILS_DIAG_H   = INTERIM_H / "soils_v01_diagnostics.parquet"

# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------
def _where(p: Path, q: Path) -> Path | None:
    if p.exists():
        return p
    if q.exists():
        return q
    return None

@st.cache_data(show_spinner=False)
def load_parquet(primary: Path, fallback: Path) -> pd.DataFrame:
    path = _where(primary, fallback)
    if path is None:
        return pd.DataFrame()
    return pd.read_parquet(path)

def canon(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    if "region_iso" in df.columns:
        df["region_iso"] = df["region_iso"].astype(str).str.strip()
    if "country" in df.columns:
        df["country"] = df["country"].astype(str).str.strip()
    # normalize texture text a bit
    if "texture_class" in df.columns:
        df["texture_class"] = df["texture_class"].astype(str).str.lower().str.strip()
    return df

# --------------------------------------------------------------------------------------
# Load data
# --------------------------------------------------------------------------------------
soil    = canon(load_parquet(SOILS_PARQ_D, SOILS_PARQ_H))
soil_dg = canon(load_parquet(SOILS_DIAG_D, SOILS_DIAG_H))  # may be empty

col1, col2, col3 = st.columns(3)
col1.metric("Soil rows", f"{len(soil):,}")
col2.metric("Diagnostics rows", f"{len(soil_dg):,}")
n_countries = soil["country"].nunique() if "country" in soil.columns else 0
col3.metric("Countries (soils)", f"{n_countries:,}")

if soil.empty:
    st.error("Could not load `soils.parquet` from D: or HOME. Please check your paths.")
    st.stop()

# --------------------------------------------------------------------------------------
# Sidebar filters
# --------------------------------------------------------------------------------------
with st.sidebar:
    st.header("Filters")
    if "country" in soil.columns:
        countries_all = sorted(soil["country"].dropna().unique().tolist())
        selected_countries = st.multiselect(
            "Country (multi-select)", countries_all, default=countries_all[: min(5, len(countries_all))]
        )
    else:
        selected_countries = []

# Apply country filter to both frames (if available)
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
    dist = (soil_f["quality_flag"].value_counts(dropna=False)
            .rename_axis("quality_flag")
            .reset_index(name="count")
            .sort_values("quality_flag"))

    st.markdown("**Overall distribution**")
    try:
        if not _PLOTLY_OK:
            raise RuntimeError("Plotly unavailable")
        fig = px.bar(dist, x="quality_flag", y="count", text="count",
                     title="Soil quality_flag distribution (overall)")
        fig.update_traces(textposition="outside")
        fig.update_layout(yaxis_title="count", xaxis_title="quality_flag")
        st.plotly_chart(fig, use_container_width=True)
    except Exception:
        # Fallback to Matplotlib
        fig, ax = plt.subplots(figsize=(6, 4), dpi=140)
        ax.bar(dist["quality_flag"].astype(str), dist["count"])
        for i, v in enumerate(dist["count"].values):
            ax.text(i, v + max(dist["count"])*0.01, str(v), ha="center", va="bottom", fontsize=9)
        ax.set_title("Soil quality_flag distribution (overall)")
        ax.set_xlabel("quality_flag"); ax.set_ylabel("count")
        ax.grid(True, axis="y", alpha=0.3)
        st.pyplot(fig)

    # By-country stacked bars (if country column exists)
    if "country" in soil_f.columns:
        st.markdown("**By country (stacked)**")
        byc = (soil_f.groupby(["country", "quality_flag"]).size()
               .reset_index(name="count"))
        pivot = byc.pivot(index="country", columns="quality_flag", values="count").fillna(0)

        try:
            if not _PLOTLY_OK:
                raise RuntimeError("Plotly unavailable")
            fig = px.bar(
                pivot.reset_index().melt(id_vars="country", var_name="quality_flag", value_name="count"),
                x="country", y="count", color="quality_flag", barmode="stack",
                title="Soil quality_flag by country (stacked counts)"
            )
            fig.update_layout(yaxis_title="count", xaxis_title="country")
            st.plotly_chart(fig, use_container_width=True)
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

# prefer diagnostics if it has npk_method; else fall back to soils (if present there)
npk_src = None
if ("npk_method" in soil_dg_f.columns) and not soil_dg_f.empty:
    npk_src = soil_dg_f.copy()
elif ("npk_method" in soil_f.columns) and not soil_f.empty:
    npk_src = soil_f.copy()

if npk_src is None:
    st.warning("`npk_method` not found (neither diagnostics nor soils) — skipping heuristic share panel.")
else:
    # Ensure country present
    if "country" not in npk_src.columns:
        if "region_iso" in npk_src.columns:
            npk_src["country"] = npk_src["region_iso"].astype(str).str[:2]
        else:
            npk_src["country"] = "??"

    grp = (npk_src
           .assign(is_heur=npk_src["npk_method"].astype(str).eq("heuristic_v01"))
           .groupby("country", as_index=False)
           .agg(total=("npk_method", "size"),
                heur=("is_heur", "sum")))
    if grp.empty:
        st.info("No rows available to compute heuristic share.")
    else:
        grp["share_heuristic"] = (grp["heur"] / grp["total"]).fillna(0.0)

        try:
            if not _PLOTLY_OK:
                raise RuntimeError("Plotly unavailable")
            fig = px.bar(
                grp.sort_values("share_heuristic", ascending=False),
                x="country", y="share_heuristic",
                title="Share of regions using npk_method = 'heuristic_v01' by country",
                labels={"share_heuristic": "share (0–1)"}
            )
            fig.update_yaxes(range=[0, 1])
            st.plotly_chart(fig, use_container_width=True)
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
    st.dataframe(soil_f, use_container_width=True)

with st.expander("Show diagnostics table (if any)"):
    if soil_dg_f.empty:
        st.info("Diagnostics frame is empty.")
    else:
        st.dataframe(soil_dg_f, use_container_width=True)
