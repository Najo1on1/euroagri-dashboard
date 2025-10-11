from pathlib import Path
import pandas as pd

# Prefer your WSL D: drive; fallback to HOME project
D_ROOT = Path("/mnt/d/Colab/Ecosystem/Deep RL and LLM/Agri")
H_ROOT = Path("/home/najo1o11/euroagri-advisor")
PROJECT_ROOT = D_ROOT if (D_ROOT / "data").exists() else H_ROOT

DATA_INTERIM   = PROJECT_ROOT / "data" / "interim"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"

# Canonical files used across notebooks
CLI_PARQ       = DATA_INTERIM   / "climate_windows.parquet"
DISEASE_PARQ   = DATA_INTERIM   / "disease_risk.parquet"
SOILS_PARQ     = DATA_INTERIM   / "soils.parquet"
SOILS_DIAG_PARQ= DATA_INTERIM   / "soils_v01_diagnostics.parquet"
RECS_PARQ      = DATA_PROCESSED / "recommendations.parquet"

def load_df(primary: Path, fallback: Path | None = None) -> pd.DataFrame:
    """Load parquet if it exists; else return empty DataFrame."""
    if primary.exists():
        return pd.read_parquet(primary)
    if fallback and fallback.exists():
        return pd.read_parquet(fallback)
    return pd.DataFrame()

def exists_summary() -> dict:
    return {
        "PROJECT_ROOT": str(PROJECT_ROOT),
        "CLI_PARQ": CLI_PARQ.exists(),
        "DISEASE_PARQ": DISEASE_PARQ.exists(),
        "SOILS_PARQ": SOILS_PARQ.exists(),
        "SOILS_DIAG_PARQ": SOILS_DIAG_PARQ.exists(),
        "RECS_PARQ": RECS_PARQ.exists(),
    }
