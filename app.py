"""
arXiv XAI & Reasoning semantic search — Streamlit app.

Loads the prebuilt Chroma database in ./chromadb (created by build_db.py)
and the same embedding model used to build it. Nothing is re-embedded at
startup except the user's query.

Run locally:  streamlit run app.py
"""

from collections import Counter
from pathlib import Path

import chromadb
import streamlit as st
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"  # must match build_db.py
DB_DIR = Path(__file__).parent / "chromadb"
COLLECTION_NAME = "arxiv_xai"

EXAMPLES = [
    "How can we tell whether a model's explanation is faithful to its actual reasoning?",
    "Finding circuits inside transformers that implement specific behaviors",
    "Does chain-of-thought prompting actually improve reasoning or just look like it?",
    "Counterfactual explanations for decisions made by credit models",
    "Interpretable models for medical diagnosis that doctors can trust",
]

st.set_page_config(page_title="arXiv XAI Semantic Search", page_icon="🔎")


# --- Load once per server process (cached across reruns and users) ---------
@st.cache_resource(show_spinner="Loading embedding model...")
def load_model():
    return SentenceTransformer(MODEL_NAME)


@st.cache_resource(show_spinner="Opening vector database...")
def load_collection():
    client = chromadb.PersistentClient(path=str(DB_DIR),
                                       settings=Settings(anonymized_telemetry=False))
    collection = client.get_collection(COLLECTION_NAME)
    metas = collection.get(include=["metadatas"])["metadatas"]
    # One (category, year) entry per paper, used to count filter matches.
    papers = {m["paper_id"]: (m["primary_category"], m["year"]) for m in metas}
    cat_counts = Counter(cat for cat, _ in papers.values())
    stats = {
        "n_chunks": len(metas),
        "n_papers": len(papers),
        "papers": list(papers.values()),
        "cat_counts": cat_counts,
        "categories": sorted(cat_counts, key=lambda c: -cat_counts[c]),
        "years": sorted({year for _, year in papers.values()}),
    }
    return collection, stats


model = load_model()
collection, stats = load_collection()
MIN_YEAR, MAX_YEAR = stats["years"][0], stats["years"][-1]


def count_matching(category: str, min_year: int) -> int:
    return sum(1 for cat, year in stats["papers"]
               if (category == "All" or cat == category) and year >= min_year)


def describe_filters(category: str, min_year: int) -> str:
    parts = []
    if category != "All":
        parts.append(f"category **{category}**")
    if min_year > MIN_YEAR:
        parts.append(f"published in or after **{min_year}**")
    return " and ".join(parts)


def explain_no_matches(category: str, min_year: int) -> str:
    """Say which filter rules everything out and what would fix it."""
    msg = f"No papers in the dataset match {describe_filters(category, min_year)}."
    if category != "All":
        cat_years = [year for cat, year in stats["papers"] if cat == category]
        msg += (f" The newest {category} paper is from {max(cat_years)}, so lower "
                "the year or choose another category.")
    return msg


# --- Search ------------------------------------------------------------------
def build_filter(category: str, min_year: int):
    clauses = []
    if category != "All":
        clauses.append({"primary_category": category})
    if min_year > MIN_YEAR:
        clauses.append({"year": {"$gte": int(min_year)}})
    if not clauses:
        return None
    return clauses[0] if len(clauses) == 1 else {"$and": clauses}


def semantic_search(query: str, k: int, category: str, min_year: int):
    """Return up to k (document, metadata, similarity) tuples, one per paper."""
    q_emb = model.encode([query], normalize_embeddings=True).tolist()
    # Over-fetch chunks, then keep the best-scoring chunk per paper so a paper
    # split into several chunks can't take up multiple result slots.
    res = collection.query(
        query_embeddings=q_emb,
        n_results=min(k * 4, stats["n_chunks"]),
        where=build_filter(category, min_year),
        include=["documents", "metadatas", "distances"],
    )
    seen, hits = set(), []
    for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0],
                               res["distances"][0]):
        if meta["paper_id"] in seen:
            continue
        seen.add(meta["paper_id"])
        hits.append((doc, meta, 1 - dist))  # cosine distance -> similarity
        if len(hits) == k:
            break
    return hits


# --- UI ----------------------------------------------------------------------
st.title("🔎 arXiv XAI & Reasoning Search")
st.markdown(
    f"Semantic search over **{stats['n_papers']:,} arXiv papers** "
    f"({stats['n_chunks']:,} chunks) on explainable AI, interpretability, and LLM "
    "reasoning. Describe what you're looking for in plain language; results are "
    "ranked by meaning, not keywords."
)
st.caption(f"Embeddings: `{MODEL_NAME}` · Vector store: Chroma (cosine)")

FILTER_DEFAULTS = {"k": 5, "category": "All", "min_year": MIN_YEAR}


def reset_filters():
    for key, value in FILTER_DEFAULTS.items():
        st.session_state[key] = value


# Seed widget state once; widgets then read from session_state (no value= args),
# which lets the reset button change them without Streamlit warnings.
for key, value in FILTER_DEFAULTS.items():
    st.session_state.setdefault(key, value)

with st.sidebar:
    st.header("Options")
    st.caption("Changes apply to the results right away.")
    k = st.slider("Number of papers", 1, 15, key="k")
    category = st.selectbox(
        "Primary category", ["All"] + stats["categories"], key="category",
        format_func=lambda c: (f"All ({stats['n_papers']:,} papers)" if c == "All"
                               else f"{c} ({stats['cat_counts'][c]:,})"),
    )
    if MAX_YEAR > MIN_YEAR:
        min_year = st.slider("Published in or after", MIN_YEAR, MAX_YEAR,
                             key="min_year")
    else:
        min_year = MIN_YEAR

    n_match = count_matching(category, min_year)
    if n_match == 0:
        st.warning("No papers match these filters.")
    else:
        st.markdown(f"**{n_match:,}** of {stats['n_papers']:,} papers match these filters.")
    st.button("Reset filters", on_click=reset_filters, width="stretch")

    st.header("Try an example")
    for ex in EXAMPLES:
        if st.button(ex, width="stretch"):
            st.session_state.query = ex

with st.form("search", border=False):
    col1, col2 = st.columns([5, 1], vertical_alignment="bottom")
    query = col1.text_input("Search query", key="query",
                            placeholder="e.g. attributing predictions to input features in vision models")
    submitted = col2.form_submit_button("Search", type="primary", width="stretch")

query = (query or "").strip()
filters = describe_filters(category, min_year)

if not query:
    if submitted:
        st.warning("Type a search query first.")
    else:
        st.info("Type a question or topic above, or pick an example from the sidebar.")
elif n_match == 0:
    st.warning(explain_no_matches(category, min_year))
else:
    hits = semantic_search(query, k, category, min_year)
    summary = f"Top {len(hits)} of {n_match:,} papers for **“{query}”**"
    st.markdown(summary + (f", filtered to {filters}." if filters else "."))
    if n_match < k:
        st.caption(f"Only {n_match} paper{'s' if n_match != 1 else ''} match these "
                   f"filters, so fewer than {k} results are shown.")
    for rank, (doc, m, score) in enumerate(hits, 1):
        title_prefix = f"{m['title']}. "
        passage = doc[len(title_prefix):] if doc.startswith(title_prefix) else doc
        part = (f" · passage {m['chunk_index'] + 1}/{m['n_chunks']}"
                if m["n_chunks"] > 1 else "")
        with st.container(border=True):
            st.markdown(f"#### {rank}. [{m['title']}]({m['url']})")
            st.markdown(f"**{m['authors']}** · {m['published']} · "
                        f"`{m['primary_category']}` · similarity **{score:.3f}**{part}")
            st.write(passage)