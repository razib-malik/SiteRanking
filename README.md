# Automated Identification & Ranking of Website Pages

A Streamlit app that crawls a website and ranks pages by **structure, content, and relevance**.

## Signals
- **Structure**: PageRank, in-degree, out-degree, discovery depth
- **Content**: TF-IDF relevance to user query, title/header keyword hits, content length
- **Freshness**: Presence of `Last-Modified` header (weak hint)

## Scoring
Weighted sum of normalized signals (weights adjustable in UI).

## Run locally
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
streamlit run streamlit_app.py
