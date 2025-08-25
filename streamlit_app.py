#!/usr/bin/env python3
import streamlit as st
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

# ---------------------------
# Helper Functions
# ---------------------------

def fetch_page(url):
    """Fetch the HTML content of a single webpage."""
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        return response.text
    except Exception as e:
        st.error(f"Error fetching {url}: {e}")
        return ""

def extract_links(base_url, html_content):
    """Extract and normalize all internal links from a page."""
    soup = BeautifulSoup(html_content, "html.parser")
    links = set()
    for a_tag in soup.find_all("a", href=True):
        href = a_tag["href"]
        full_url = urljoin(base_url, href)
        if base_url in full_url:  # keep only internal links
            links.add(full_url.split("#")[0])  # strip fragments
    return links

def crawl_website(base_url, max_pages=10):
    """Crawl website and return a dictionary of {url: text_content}."""
    visited = set()
    to_visit = [base_url]
    pages = {}

    while to_visit and len(visited) < max_pages:
        url = to_visit.pop(0)
        if url in visited:
            continue

        html = fetch_page(url)
        if not html:
            continue

        soup = BeautifulSoup(html, "html.parser")
        text = soup.get_text(" ", strip=True)
        pages[url] = text

        new_links = extract_links(base_url, html)
        to_visit.extend(new_links - visited)
        visited.add(url)

    return pages

def rank_pages(pages, query):
    """Rank pages by cosine similarity to query using TF-IDF."""
    documents = list(pages.values())
    urls = list(pages.keys())

    # Fit TF-IDF on documents + query
    vectorizer = TfidfVectorizer(stop_words="english")
    doc_vectors = vectorizer.fit_transform(documents)
    query_vector = vectorizer.transform([query])

    # Debugging: check shapes
    st.write(f"Document vectors shape: {doc_vectors.shape}")
    st.write(f"Query vector shape: {query_vector.shape}")

    # Cosine similarity
    scores = cosine_similarity(doc_vectors, query_vector).flatten()

    # Rank
    results = pd.DataFrame({
        "url": urls,
        "relevance_score": scores
    }).sort_values(by="relevance_score", ascending=False)

    return results

# ---------------------------
# Streamlit App
# ---------------------------

st.title("🔎 Website Page Relevance Ranker")

# User inputs
website = st.text_input("Enter website URL:", "www.egain.com")
query = st.text_input("Relevance Topic / Query (comma-separated keywords):", "AI, customer service")
max_pages = st.slider("Max pages to crawl:", 5, 30, 10)

if st.button("Analyze"):
    # Ensure https:// prefix
    if not website.startswith("http"):
        website = "https://" + website

    st.write(f"Crawling: {website}")

    # Crawl website
    pages = crawl_website(website, max_pages=max_pages)

    if not pages:
        st.error("No pages fetched. Please try again with a different site.")
    else:
        st.success(f"Fetched {len(pages)} pages ✅")

        # Rank pages
        results = rank_pages(pages, query)

        # Show results
        st.subheader("Ranked Pages")
        st.dataframe(results)

        # Show top 3
        st.subheader("Top 3 Pages")
        for i, row in results.head(3).iterrows():
            st.markdown(f"**[{row['url']}]({row['url']})** — Relevance: {row['relevance_score']:.4f}")

