from pathlib import Path
import streamlit as st
from src import data_paths as P  # keep your existing helper

# ---- Page config
ICON_PATH = Path("assets/logo.png")
PAGE_ICON = str(ICON_PATH) if ICON_PATH.exists() else "🌾"

st.set_page_config(
    page_title="EuroAgri-Advisor Live",
    page_icon=PAGE_ICON,
    layout="wide",
    initial_sidebar_state="expanded",  # desktop expanded; mobile will auto-collapse
)

# ---- Load custom CSS (optional but nice)
css_path = Path(".streamlit/styles.css")
if css_path.exists():
    st.markdown(f"<style>{css_path.read_text()}</style>", unsafe_allow_html=True)

# ---- Title & tagline
st.title("EuroAgri-Advisor Live")
st.caption("_Climate aware adaptive crop planning for EU regions_")

st.markdown("""
This dashboard bundles the pipeline outputs:
- **Recommendations** (variety + irrigation + N-plan scores)
- **Climate Windows** (monthly rollups + derivatives)
- **Soils (harmonized)** (with heuristic NPK v01)
- **Disease Risk** (rule-based demo)
""")

# ---- Sidebar: environment + links
with st.sidebar:
    st.header("Environment")
    for k, v in P.exists_summary().items():
        st.write(f"**{k}**: {v}")

    st.divider()
    st.header("Links")
    # GitHub repo: you’ll paste the URL here when ready
    GITHUB_URL = ""  # TODO: add your repo URL when created
    LINKEDIN_URL = "https://www.linkedin.com/in/muwanguzi-jonathan-a0b766124/"

    if GITHUB_URL:
        st.markdown(f"[GitHub Repository]({GITHUB_URL})")
    st.markdown(f"[LinkedIn – Muwanguzi Jonathan]({LINKEDIN_URL})")

st.info("Use the left sidebar **Pages** to navigate.")

# ---- Custom footer
st.markdown(
    """
    <div id="custom-footer">
      <span>© Muwanguzi Jonathan • </span>
      <a href="https://www.linkedin.com/in/muwanguzi-jonathan-a0b766124/" target="_blank">LinkedIn</a>
      <span> • Made with Streamlit & Plotly</span>
    </div>
    """,
    unsafe_allow_html=True,
)