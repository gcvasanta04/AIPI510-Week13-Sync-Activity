"""
Build the Chroma vector database for the arXiv semantic search app.

Run this ONCE on your own machine (or in Colab) before deploying:

    pip install -r requirements-build.txt
    python build_db.py                 # fetch papers, chunk, embed, save to ./chromadb
    python build_db.py --refresh       # re-download papers instead of using the cache
    python build_db.py --max-per-query 300   # smaller/faster build

Pipeline
--------
1. Fetch: pull abstracts from the arXiv API for several XAI / reasoning queries
   restricted to cs.AI, cs.LG and cs.CL. Results are cached to data/papers.jsonl
   so rebuilding the index doesn't re-hit the API.
2. Chunk: each chunk = "title + a window of abstract sentences". Most abstracts
   fit in one chunk; long ones are split into overlapping sentence windows so no
   text gets silently truncated by the embedding model's 256-token limit.
3. Embed: sentence-transformers/all-MiniLM-L6-v2, L2-normalized.
4. Store: persistent Chroma collection using cosine distance, with paper
   metadata (title, authors, date, category, URL) attached to every chunk.
"""

import argparse
import json
import re
import shutil
import time
from pathlib import Path

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DB_DIR = Path("chromadb")
CACHE_PATH = Path("data/papers.jsonl")
COLLECTION_NAME = "arxiv_xai"

# Token budget per chunk. MiniLM's max_seq_length is 256 word-pieces; staying
# under ~200 leaves headroom for the title prefix and special tokens.
MAX_CHUNK_TOKENS = 200
OVERLAP_SENTENCES = 1

CATEGORIES = "(cat:cs.AI OR cat:cs.LG OR cat:cs.CL)"
TOPIC_QUERIES = [
    'abs:"explainable AI"',
    'abs:"explainability"',
    'abs:"interpretability"',
    'abs:"mechanistic interpretability"',
    'abs:"feature attribution"',
    'abs:"counterfactual explanation"',
    'abs:"concept bottleneck"',
    'abs:"chain-of-thought"',
    'abs:"reasoning" AND abs:"language model"',
    'abs:"faithfulness" AND abs:"explanation"',
]


# --------------------------------------------------------------------------- #
# 1. Fetch
# --------------------------------------------------------------------------- #
def fetch_papers(max_per_query: int) -> list[dict]:
    import arxiv  # only needed for fetching

    # arXiv asks for >= 3 s between requests; the client handles paging + delay.
    client = arxiv.Client(page_size=200, delay_seconds=3.0, num_retries=5)
    papers: dict[str, dict] = {}

    for topic in TOPIC_QUERIES:
        query = f"{CATEGORIES} AND ({topic})"
        search = arxiv.Search(
            query=query,
            max_results=max_per_query,
            sort_by=arxiv.SortCriterion.SubmittedDate,
        )
        before = len(papers)
        try:
            for r in client.results(search):
                pid = re.sub(r"v\d+$", "", r.get_short_id())  # drop version suffix
                if pid in papers:
                    continue
                authors = [a.name for a in r.authors]
                papers[pid] = {
                    "id": pid,
                    "title": clean(r.title),
                    "abstract": clean(r.summary),
                    "authors": authors,
                    "published": r.published.strftime("%Y-%m-%d"),
                    "primary_category": r.primary_category,
                    "url": f"https://arxiv.org/abs/{pid}",
                }
        except Exception as e:  # keep what we have if one query fails
            print(f"  ! query failed ({e!r}); continuing")
        print(f"  {topic:<45} +{len(papers) - before:>4}  (total {len(papers)})")
        time.sleep(3)

    return list(papers.values())


def load_or_fetch(max_per_query: int, refresh: bool) -> list[dict]:
    if CACHE_PATH.exists() and not refresh:
        with CACHE_PATH.open() as f:
            papers = [json.loads(line) for line in f]
        print(f"Loaded {len(papers)} cached papers from {CACHE_PATH}")
        return papers

    print("Fetching papers from the arXiv API (takes a few minutes)...")
    papers = fetch_papers(max_per_query)
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with CACHE_PATH.open("w") as f:
        for p in papers:
            f.write(json.dumps(p) + "\n")
    print(f"Saved {len(papers)} papers to {CACHE_PATH}")
    return papers


def clean(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


# --------------------------------------------------------------------------- #
# 2. Chunk
# --------------------------------------------------------------------------- #
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9(\[])")


def chunk_paper(paper: dict, count_tokens) -> list[str]:
    """Title-prefixed sentence windows that fit within MAX_CHUNK_TOKENS."""
    prefix = f"{paper['title']}. "
    budget = MAX_CHUNK_TOKENS - count_tokens(prefix)
    sentences = [s for s in _SENT_SPLIT.split(paper["abstract"]) if s]

    # Fast path: whole abstract fits.
    if count_tokens(paper["abstract"]) <= budget:
        return [prefix + paper["abstract"]]

    chunks, window, window_tokens = [], [], 0
    for sent in sentences:
        n = count_tokens(sent)
        if window and window_tokens + n > budget:
            chunks.append(prefix + " ".join(window))
            window = window[-OVERLAP_SENTENCES:] if OVERLAP_SENTENCES else []
            window_tokens = sum(count_tokens(s) for s in window)
        window.append(sent)
        window_tokens += n
    if window:
        chunks.append(prefix + " ".join(window))
    return chunks


def build_chunks(papers: list[dict], tokenizer) -> tuple[list, list, list]:
    def count_tokens(text: str) -> int:
        return len(tokenizer.tokenize(text))

    ids, docs, metas = [], [], []
    for p in papers:
        chunks = chunk_paper(p, count_tokens)
        authors = ", ".join(p["authors"][:3]) + (" et al." if len(p["authors"]) > 3 else "")
        for i, chunk in enumerate(chunks):
            ids.append(f"{p['id']}::{i}")
            docs.append(chunk)
            metas.append({
                "paper_id": p["id"],
                "title": p["title"],
                "authors": authors,
                "published": p["published"],
                "year": int(p["published"][:4]),
                "primary_category": p["primary_category"],
                "url": p["url"],
                "chunk_index": i,
                "n_chunks": len(chunks),
            })
    return ids, docs, metas


# --------------------------------------------------------------------------- #
# 3 + 4. Embed and store
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-per-query", type=int, default=600,
                    help="max papers fetched per topic query (default 600)")
    ap.add_argument("--refresh", action="store_true",
                    help="ignore data/papers.jsonl and re-fetch from arXiv")
    args = ap.parse_args()

    papers = load_or_fetch(args.max_per_query, args.refresh)
    if not papers:
        raise SystemExit("No papers fetched; check your network connection.")

    model = SentenceTransformer(MODEL_NAME)
    ids, docs, metas = build_chunks(papers, model.tokenizer)
    multi = sum(1 for m in metas if m["chunk_index"] == 1)
    print(f"{len(papers)} papers -> {len(docs)} chunks "
          f"({multi} papers needed more than one chunk)")

    print("Embedding...")
    embeddings = model.encode(docs, batch_size=64, show_progress_bar=True,
                              normalize_embeddings=True)

    # Start from a clean directory so stale index files never get committed.
    if DB_DIR.exists():
        shutil.rmtree(DB_DIR)
    client = chromadb.PersistentClient(path=str(DB_DIR),
                                       settings=Settings(anonymized_telemetry=False))
    collection = client.create_collection(
        COLLECTION_NAME,
        configuration={"hnsw": {"space": "cosine"}},
        metadata={"embedding_model": MODEL_NAME, "n_papers": len(papers)},
    )

    batch = min(1000, client.get_max_batch_size())
    for start in range(0, len(ids), batch):
        end = start + batch
        collection.add(
            ids=ids[start:end],
            documents=docs[start:end],
            embeddings=embeddings[start:end].tolist(),
            metadatas=metas[start:end],
        )
    print(f"Stored {collection.count()} chunks in {DB_DIR}/ "
          f"(collection '{COLLECTION_NAME}')")

    # Sanity check: one query straight against the saved DB.
    q = model.encode(["how do we know if a model's explanation is faithful?"],
                     normalize_embeddings=True)
    res = collection.query(query_embeddings=q.tolist(), n_results=3)
    print("\nSanity check query results:")
    for meta, dist in zip(res["metadatas"][0], res["distances"][0]):
        print(f"  {1 - dist:.3f}  {meta['title'][:90]}")


if __name__ == "__main__":
    main()
