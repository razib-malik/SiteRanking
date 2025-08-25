#!/usr/bin/env python3
import re
import time
import math
import requests
import tldextract
import pandas as pd
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urldefrag, urlparse
from collections import deque, defaultdict

import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ------------------------- CONFIG -------------------------
TIMEOUT = 10
HEADERS = {"User-Agent": "Mozilla/5.0 (RankingBot/1.0)"}

# ------------------------- HELPERS -------------------------
def normalize_url(base, link):
    if not link:
        return None
    # remove anchors/fragments; build absolute
    url = urljoin(base, link)
    url, _ = urldefrag(url)
    # keep only http(s)
    if not (url.startswith("http://") or url.startswith("https://")):
        return None
    return url

def same_site(seed, candidate):
    s1 = tldextract.extract(seed)
    s2 = tldextract.extract(candidate)
    return (s1.registered_domain == s2.registered_domain)

def get_html(url):
    try:
        r = requests.get(url, headers=HEADERS, timeout=TIMEOUT)
        ctype = r.headers.get("Content-Type","").lower()
        if r.status_code == 200 and "text/html" in ctype:
            return r.text, r.headers
    except requests.RequestException:
        return None, {}
    return None, {}

def parse_page(url, html):
    soup = BeautifulSoup(html, "html.parser")
    title = (soup.title.string.strip() if soup.title and soup.title.string else "")
    # text content
    for bad in soup(["script","style","noscript"]):
        bad.extract()
    text = re.sub(r"\s+", " ", soup.get_text(" ", strip=True))
    # headers
    headers = " ".join(h.get_text(" ", strip=True) for h in soup.find_all(re.compile("^h[1-3]$")))
    # links
    links = [a.get("href") for a in soup.find_all("a", href=True)]
    return title, text, headers, links

def pagerank(graph, damping=0.85, max_iter=40, tol=1.0e-6):
    nodes = list(graph.keys())
    N = len(nodes)
    if N == 0:
        return {}
    idx = {n:i for i,n in enumerate(nodes)}
    outdeg = {n: len([t for t in graph[n] if t in idx]) for n in nodes}
    rank = {n: 1.0/N for n in nodes}

    for _ in range(max_iter):
        new = dict.fromkeys(nodes, (1.0 - damping)/N)
        for n in nodes:
            if outdeg[n] == 0:
                # distribute dangling
                share = damping * (rank[n]/N)
                for m in nodes:
                    new[m] += share
            else:
                share = damping * (rank[n]/outdeg[n])
                for t in graph[n]:
                    if t in idx:
                        new[t] += share
        delta = sum(abs(new[n]-rank[n]) for n in nodes)
        rank = new
        if delta < tol:
            break
    return rank

def safe_len(x):
    return len(x) if isinstance(x, (list, set, dict, str)) else 0

def try_last_modified(headers):
    lm = headers.get("Last-Modified")
    if not lm:
        return None
    return lm  # just bubble raw header; optional scoring fallback below

def normalize_series(s):
    if s.empty:
        return s
    mn, mx = s.min(), s.max()
    if math.isclose(mx, mn):
        return pd.Series([0.5]*len(s), index=s.index)
    return (s - mn) / (mx - mn)

# ------------------------- STREAMLIT UI -------------------------
st.set_page_config(page_title="Automated Page Ranking", layout="wide")
st.title("🔎 Automated Identification & Ranking of Website Pages")
st.caption("Ranks pages by structure, content, and relevance signals")

with st.sidebar:
    seed_url = st.text_input("Seed URL (e.g., https://www.egain.com)", "https://www.egain.com")
    topic = st.text_input("Relevance Topic / Query (comma-separated keywords)", "knowledge, customer service, ai")
    max_pages = st.number_input("Max pages to crawl", 20, 1000, 150, step=10)
    per_domain_only = st.checkbox("Restrict to same registered domain", True)
    crawl_btn = st.button("Crawl & Rank")

# ------------------------- CRAWL -------------------------
if crawl_btn:
    t0 = time.time()
    st.info("Crawling… this may take a minute depending on site & depth.")

    visited = set()
    q = deque([seed_url])
    graph = defaultdict(set)        # adjacency list
    inlinks = defaultdict(set)      # reverse links
    pages = {}                      # url -> {title,text,headers,links,lastmod}
    order = []                      # discovered order

    while q and len(visited) < max_pages:
        url = q.popleft()
        if url in visited:
            continue
        if per_domain_only and not same_site(seed_url, url):
            continue

        html, headers = get_html(url)
        visited.add(url)
        if not html:
            continue

        title, text, headers_text, links = parse_page(url, html)
        lastmod = try_last_modified(headers)
        order.append(url)
        pages[url] = {
            "title": title,
            "text": text,
            "headers": headers_text,
            "links": [],
            "lastmod": lastmod
        }

        # normalize and keep internal links first
        normalized_links = []
        for lk in links:
            nu = normalize_url(url, lk)
            if not nu:
                continue
            if per_domain_only and not same_site(seed_url, nu):
                continue
            normalized_links.append(nu)

        # Deduplicate but keep order
        seen = set()
        cleaned = []
        for lk in normalized_links:
            if lk not in seen:
                seen.add(lk)
                cleaned.append(lk)

        pages[url]["links"] = cleaned
        for lk in cleaned:
            graph[url].add(lk)
            inlinks[lk].add(url)
            if lk not in visited and len(visited) + len(q) < max_pages:
                q.append(lk)

    # make sure all nodes exist in graph even if no outlinks
    for u in list(pages.keys()):
        graph.setdefault(u, set())

    # ------------------------- SIGNALS -------------------------
    # Structure: PageRank, in-degree, out-degree, crawl-depth (discovery order)
    pr = pagerank(graph)
    in_deg = {u: safe_len(inlinks.get(u, set())) for u in pages}
    out_deg = {u: safe_len(graph.get(u, set())) for u in pages}
    depth_idx = {u: i for i,u in enumerate(order)}  # earlier discovery ~ shallower

    # Content: TF-IDF relevance vs topic; Title/Header boosts; Content length
    docs = [pages[u]["text"] for u in pages]
    urls = list(pages.keys())

    # Build TF-IDF on corpus + query terms
    vectorizer = TfidfVectorizer(stop_words="english", max_features=20000)
    X = vectorizer.fit_transform(docs + [topic])
    corpus_X, query_x = X[:-1], X[-1]
    cos = cosine_similarity(corpus_X, query_x)
    relevance = {urls[i]: float(cos[i][0]) for i in range(len(urls))}

    # simple keyword boost for title/headers
    topic_terms = [t.strip().lower() for t in re.split(r"[,\s]+", topic) if t.strip()]
    def kw_hits(s):
        s = (s or "").lower()
        return sum(s.count(k) for k in topic_terms) if s else 0

    title_hits = {u: kw_hits(pages[u]["title"]) for u in pages}
    header_hits = {u: kw_hits(pages[u]["headers"]) for u in pages}
    content_len = {u: len((pages[u]["text"] or "")) for u in pages}

    # Freshness: use presence of Last-Modified header as a weak signal
    freshness = {u: 1.0 if pages[u]["lastmod"] else 0.0 for u in pages}

    # ------------------------- SCORING -------------------------
    df = pd.DataFrame({
        "url": urls,
        "title": [pages[u]["title"] for u in urls],
        "in_degree": [in_deg[u] for u in urls],
        "out_degree": [out_deg[u] for u in urls],
        "pagerank": [pr.get(u, 0.0) for u in urls],
        "tfidf_relevance": [relevance[u] for u in urls],
        "title_hits": [title_hits[u] for u in urls],
        "header_hits": [header_hits[u] for u in urls],
        "content_length": [content_len[u] for u in urls],
        "freshness_hint": [freshness[u] for u in urls],
        "discovery_order": [depth_idx[u] for u in urls],
    })

    # Normalize and combine (weights adjustable in UI)
    with st.sidebar:
        st.markdown("---")
        st.subheader("Weights")
        w_pr = st.slider("Structure: PageRank", 0.0, 1.0, 0.30, 0.05)
        w_in = st.slider("Structure: In-degree", 0.0, 1.0, 0.20, 0.05)
        w_out = st.slider("Structure: Out-degree", 0.0, 1.0, 0.05, 0.05)
        w_rel = st.slider("Content: TF-IDF relevance", 0.0, 1.0, 0.35, 0.05)
        w_th = st.slider("Content: Title/Headers boost", 0.0, 1.0, 0.05, 0.05)
        w_len = st.slider("Content: Length", 0.0, 1.0, 0.03, 0.01)
        w_fr = st.slider("Freshness hint", 0.0, 1.0, 0.02, 0.01)

    df["n_pr"] = normalize_series(df["pagerank"])
    df["n_in"] = normalize_series(df["in_degree"])
    df["n_out"] = normalize_series(df["out_degree"])
    df["n_rel"] = normalize_series(df["tfidf_relevance"])
    df["n_th"] = normalize_series(df["title_hits"] + df["header_hits"])
    df["n_len"] = normalize_series(df["content_length"])
    df["n_fr"] = df["freshness_hint"]  # already 0/1

    df["score"] = (
        w_pr*df["n_pr"] +
        w_in*df["n_in"] +
        w_out*df["n_out"] +
        w_rel*df["n_rel"] +
        w_th*df["n_th"] +
        w_len*df["n_len"] +
        w_fr*df["n_fr"]
    )

    df = df.sort_values("score", ascending=False).reset_index(drop=True)

    # ------------------------- UI OUTPUT -------------------------
    st.success(f"Crawled {len(df)} pages in {time.time()-t0:.1f}s")

    st.subheader("🏆 Top Ranked Pages")
    st.dataframe(
        df[["url","title","score","pagerank","in_degree","out_degree","tfidf_relevance","title_hits","header_hits","freshness_hint"]]
        .head(30),
        use_container_width=True
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**Structure vs Relevance**")
        st.scatter_chart(df[["n_pr","n_in","n_rel"]].rename(columns={"n_pr":"PageRank","n_in":"InDeg","n_rel":"Relevance"}))
    with c2:
        st.markdown("**Distribution of Scores**")
        st.bar_chart(df["score"])
    with c3:
        st.markdown("**Top Outdegree Pages**")
        st.bar_chart(df.sort_values("out_degree",ascending=False).head(10)[["out_degree"]])

    st.markdown("### Raw Results (CSV)")
    st.download_button(
        "Download Rankings CSV",
        df.to_csv(index=False).encode("utf-8"),
        file_name="page_rankings.csv",
        mime="text/csv"
    )

    with st.expander("Explainability: How score is computed"):
        st.write("""
        **score =** w_pr·PageRank + w_in·In-degree + w_out·Out-degree +
        w_rel·TF-IDF relevance + w_th·(Title+Header keyword hits) +
        w_len·Content length + w_fr·Freshness hint

        All inputs are min–max normalized per crawl (0–1) except freshness (0/1).
        We show top pages, plus structural & content diagnostics.
        """)
