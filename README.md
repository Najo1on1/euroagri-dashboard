# EuroAgri-Advisor Live (Streamlit)

**Climate aware adaptive crop planning for EU regions.**

This repo hosts the interactive dashboard showcasing:
- Recommendations (variety + irrigation + N-plan + disease fit)
- Climate windows & derivatives
- Harmonized soils (with heuristic NPK v01)
- Disease risk (rule-based demo)

## Run locally
\`\`\`bash
conda activate ircli  # or your env
pip install -r requirements.txt
streamlit run app.py
\`\`\`

## Data
The app looks for data in this order:
1. D: WSL path (`/mnt/d/...`)
2. HOME path (`/home/najo1o11/...`)
3. `data/sample/` (bundled small files so the app runs on cloud)

## Deploy
Works on Streamlit Community Cloud and Hugging Face Spaces.

## Author
[Muwanguzi Jonathan](https://www.linkedin.com/in/muwanguzi-jonathan-a0b766124/)
