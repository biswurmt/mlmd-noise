#!/usr/bin/env python3
"""
build_vector_db_chroma.py — Build and query a local ChromaDB vector database
from filtered medical guidelines using local MedEmbed / MedEIR embeddings
(sentence-transformers, no API key required).

Index usage:
    python build_vector_db_chroma.py                    # full combined dataset
    python build_vector_db_chroma.py --source cma       # gauge run: CMA only (~86 docs)
    python build_vector_db_chroma.py --parquet PATH     # pre-filtered parquet

Query usage:
    python build_vector_db_chroma.py --query "ST elevation myocardial infarction" --n 5
    python build_vector_db_chroma.py --query "scrotal pain sudden onset" --test testicular_ultrasound

Model options (set MEDEMBED_MODEL env var):
    abhinand/MedEmbed-small-v0.1    384-dim  ~340 MB  fastest
    abhinand/MedEmbed-base-v0.1     768-dim  ~440 MB  balanced
    abhinand/MedEmbed-large-v0.1   1024-dim  ~1.3 GB  best quality (default)
    Thakkar-AI/MedEIR               768-dim            IR-tuned alternative
"""

import argparse
import json
import os
import time
import uuid
from pathlib import Path

from dotenv import load_dotenv

# Load .env from knowledge-graphs/ first, then repo root as fallback
load_dotenv(Path(__file__).parent.parent / ".env", override=False)
load_dotenv(Path(__file__).parent.parent.parent / ".env", override=False)

# ── Constants ─────────────────────────────────────────────────────────────────
MIN_TEXT_LENGTH     = 200    # chars — drop stub/empty docs
CHUNK_SIZE          = 2000   # chars ≈ 500 tokens
CHUNK_OVERLAP       = 200    # chars overlap between consecutive chunks
COLLECTION_NAME     = "guidelines"
EMBED_BATCH_SIZE    = 32     # sentences per model.encode() call
CHROMA_UPSERT_BATCH = 100    # chunks per collection.upsert() call

MEDEMBED_MODEL = os.environ.get("MEDEMBED_MODEL", "abhinand/MedEmbed-large-v0.1")
CHROMA_PATH    = os.environ.get("CHROMA_PATH", str(Path(__file__).parent / "chroma_db"))

DATA_PATH = Path(__file__).parent.parent / "data" / "guidelines_filtered" / "combined"

# ── Lazy-initialized singletons ───────────────────────────────────────────────
_embed_model       = None   # sentence_transformers.SentenceTransformer
_chroma_collection = None   # chromadb.Collection


def _get_embed_model():
    global _embed_model
    if _embed_model is None:
        import torch
        from sentence_transformers import SentenceTransformer

        if torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"

        print(f"Loading embedding model '{MEDEMBED_MODEL}' on {device} ...")
        _embed_model = SentenceTransformer(MEDEMBED_MODEL, device=device)
        dim = _embed_model.get_sentence_embedding_dimension()
        print(f"  Model loaded. Output dim: {dim}")
    return _embed_model


def _get_collection():
    global _chroma_collection
    if _chroma_collection is None:
        import chromadb

        client = chromadb.PersistentClient(
            path=CHROMA_PATH,
            settings=chromadb.Settings(anonymized_telemetry=False),
        )
        _chroma_collection = client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        print(
            f"ChromaDB collection '{COLLECTION_NAME}' ready at '{CHROMA_PATH}' "
            f"({_chroma_collection.count()} existing points)"
        )
    return _chroma_collection


# ── Embedding helpers (symmetric — no passage/query prefix) ───────────────────
def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed a list of texts using the local model.

    MedEmbed and MedEIR are symmetric bi-encoders: passages and queries are
    encoded identically (no instruction prefix needed unlike e5-mistral).
    Pre-normalizing ensures ChromaDB cosine distance = 1 − cos_sim exactly.
    """
    model = _get_embed_model()
    vecs = model.encode(
        texts,
        batch_size=EMBED_BATCH_SIZE,
        show_progress_bar=len(texts) > EMBED_BATCH_SIZE,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return vecs.tolist()


def embed_passages(texts: list[str]) -> list[list[float]]:
    """Embed document chunks at index time — symmetric model, same as embed_texts."""
    return embed_texts(texts)


def embed_query(query_text: str) -> list[float]:
    """Embed a single query string — symmetric model, no instruction prefix."""
    return embed_texts([query_text])[0]


# ── Chunking ──────────────────────────────────────────────────────────────────
def chunk_text(text: str) -> list[str]:
    """Split text into overlapping character-based chunks."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + CHUNK_SIZE
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break
        start += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks


# ── ChromaDB helpers ──────────────────────────────────────────────────────────
def _chunk_id(chunk_key: str) -> str:
    """Deterministic string ID from a chunk key — enables idempotent upserts."""
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk_key))


def _sanitize_metadata(raw: dict) -> dict:
    """Coerce all values to ChromaDB-safe scalar types (str/int/float/bool).

    ChromaDB rejects None, lists, dicts, and other complex types in metadata.
    """
    safe = {}
    for k, v in raw.items():
        if isinstance(v, (str, int, float, bool)):
            safe[k] = v
        elif v is None:
            safe[k] = ""
        else:
            safe[k] = json.dumps(v)
    return safe


def _get_indexed_doc_ids() -> set[str]:
    """Return the set of doc_ids already present in the ChromaDB collection."""
    collection = _get_collection()
    if collection.count() == 0:
        return set()
    print("  Fetching existing doc_ids for resume detection ...")
    result = collection.get(include=["metadatas"])
    return {m["doc_id"] for m in result["metadatas"] if "doc_id" in m}


# ── Query ─────────────────────────────────────────────────────────────────────
def _fetch_surrounding_chunks(doc_id: str, chunk_index: int, window: int = 1) -> str:
    """Fetch chunks ±window around a matched chunk from the same document.

    Replaces Qdrant's scroll() + Range filter. ChromaDB's $and compound
    filter handles the doc_id equality + chunk_index range conditions.
    """
    collection = _get_collection()
    low  = max(0, chunk_index - window)
    high = chunk_index + window

    result = collection.get(
        where={
            "$and": [
                {"doc_id":      {"$eq":  doc_id}},
                {"chunk_index": {"$gte": low}},
                {"chunk_index": {"$lte": high}},
            ]
        },
        include=["metadatas", "documents"],
    )

    pairs = sorted(
        zip(result["metadatas"], result["documents"]),
        key=lambda x: x[0].get("chunk_index", 0),
    )
    return " ".join(doc for _, doc in pairs)


def query_guidelines(
    query_text: str,
    n_results: int = 5,
    diagnostic_test: str = None,
    context_window: int = 1,
) -> list[dict]:
    """Return top-N guideline chunks semantically relevant to query_text.

    Each result is expanded with surrounding chunks for fuller context.
    Identical public signature to build_vector_db.query_guidelines.

    Args:
        query_text:      Natural-language clinical query.
        n_results:       Number of results to return.
        diagnostic_test: Optional substring filter on the diagnostic_tests field
                         (e.g. 'ecg', 'testicular_ultrasound'). ChromaDB has no
                         native substring operator, so results are oversampled
                         and post-filtered in Python.
        context_window:  Chunks before/after each hit to include (default: 1).

    Returns:
        List of dicts: score, matched_text, context, title, source, url,
                       diagnostic_tests.
    """
    collection = _get_collection()
    total = collection.count()
    if total == 0:
        return []

    # Oversample when filtering so post-filter has enough candidates
    fetch_n = min(n_results * 4 if diagnostic_test else n_results, total)

    response = collection.query(
        query_embeddings=[embed_query(query_text)],
        n_results=fetch_n,
        include=["metadatas", "documents", "distances"],
    )

    results = []
    # ChromaDB returns lists-of-lists (batch API); unwrap single-query response
    for meta, doc, dist in zip(
        response["metadatas"][0],
        response["documents"][0],
        response["distances"][0],
    ):
        # ChromaDB cosine distance = 1 − cosine_similarity → convert to similarity
        score       = 1.0 - dist
        doc_id      = meta.get("doc_id", "")
        chunk_index = meta.get("chunk_index", 0)
        context     = _fetch_surrounding_chunks(doc_id, chunk_index, window=context_window)
        results.append(
            {
                "score":            score,
                "matched_text":     doc,
                "context":          context,
                "title":            meta.get("title", ""),
                "source":           meta.get("source", ""),
                "url":              meta.get("url", ""),
                "diagnostic_tests": meta.get("diagnostic_tests", ""),
            }
        )

    # Post-filter by diagnostic_test substring (ChromaDB has no MatchText operator)
    if diagnostic_test:
        results = [
            r for r in results
            if diagnostic_test.lower() in r["diagnostic_tests"].lower()
        ]

    return results[:n_results]


# ── Indexing ──────────────────────────────────────────────────────────────────
def build_index(source_filter: str = None, parquet_path: str = None):
    """Load, filter, chunk, embed, and upsert guidelines into ChromaDB.

    Args:
        source_filter: Restrict to a single source (e.g. 'cma'). Only used
                       when loading from the full Arrow dataset.
        parquet_path:  Path to a pre-filtered parquet file. Skips the Arrow
                       dataset load and the min-length/dedup filters.
    """
    import pandas as pd

    # 1. Load dataset
    if parquet_path:
        print(f"Loading pre-filtered dataset from {parquet_path} ...")
        df = pd.read_parquet(parquet_path)
        print(f"Loaded {len(df)} docs (pre-filtered)")
    else:
        from datasets import load_from_disk
        print(f"Loading dataset from {DATA_PATH} ...")
        ds = load_from_disk(str(DATA_PATH))
        df = ds.to_pandas()
        print(f"Loaded {len(df)} docs")

        # 2. Filter: minimum text length
        before = len(df)
        df = df[df["clean_text"].str.len() >= MIN_TEXT_LENGTH]
        print(f"Min length filter ({MIN_TEXT_LENGTH} chars): {before} → {len(df)} docs")

        # 3. Filter: deduplicate by title
        before = len(df)
        df = df.drop_duplicates(subset=["title"], keep="first")
        print(f"Dedup by title: {before} → {len(df)} docs")

    # 4. Optional source filter for gauge runs
    if source_filter:
        before = len(df)
        df = df[df["source"].str.lower() == source_filter.lower()]
        print(f"Source filter '{source_filter}': {before} → {len(df)} docs")

    if df.empty:
        print("No documents after filtering — nothing to index.")
        return

    # 5. Fetch already-indexed doc_ids to enable safe resume
    indexed_doc_ids = _get_indexed_doc_ids()
    if indexed_doc_ids:
        print(f"  {len(indexed_doc_ids)} doc(s) already indexed — will skip.")

    # 6. Stream through docs: chunk → embed → upsert without holding all in memory
    print(f"Streaming {len(df)} docs → chunking, embedding, upserting ...")
    collection = _get_collection()
    indexed = 0
    t_start = time.time()
    pending: list[tuple[str, str, dict]] = []  # (chunk_id, chunk_text, payload)

    def _flush(batch: list[tuple[str, str, dict]], total_so_far: int) -> int:
        chunk_ids, chunk_texts, payloads = zip(*batch)
        vectors       = embed_passages(list(chunk_texts))
        safe_payloads = [_sanitize_metadata(p) for p in payloads]
        collection.upsert(
            ids=list(chunk_ids),
            embeddings=vectors,
            documents=list(chunk_texts),
            metadatas=safe_payloads,
        )
        n       = len(chunk_ids)
        elapsed = time.time() - t_start
        rate    = (total_so_far + n) / elapsed if elapsed > 0 else 0
        print(
            f"  [{total_so_far + n}] upserted — "
            f"{elapsed:.1f}s elapsed, {rate:.1f} chunks/s",
            flush=True,
        )
        return n

    for doc_num, (_, row) in enumerate(df.iterrows(), 1):
        doc_id = str(row["id"])
        if doc_id in indexed_doc_ids:
            continue  # already fully indexed — skip

        text_chunks = chunk_text(str(row["clean_text"]))
        for i, chunk in enumerate(text_chunks):
            payload = {
                "doc_id":           doc_id,
                "title":            str(row.get("title",            ""))[:500],
                "source":           str(row.get("source",           ""))[:200],
                "url":              str(row.get("url",               ""))[:500],
                "overview":         str(row.get("overview",         ""))[:1000],
                "diagnostic_tests": str(row.get("diagnostic_tests", "")),
                "chunk_index":      i,
                # Note: chunk text is stored in ChromaDB's native documents field,
                # not duplicated here in metadata.
            }
            pending.append((_chunk_id(f"{doc_id}_chunk_{i}"), chunk, payload))

        if len(pending) >= CHROMA_UPSERT_BATCH:
            indexed += _flush(pending[:CHROMA_UPSERT_BATCH], indexed)
            pending = pending[CHROMA_UPSERT_BATCH:]

        if doc_num % 100 == 0:
            print(f"  docs processed: {doc_num}/{len(df)}", flush=True)

    # Flush remainder
    while pending:
        batch, pending = pending[:CHROMA_UPSERT_BATCH], pending[CHROMA_UPSERT_BATCH:]
        indexed += _flush(batch, indexed)

    print(f"\nDone. {indexed} chunks indexed in {time.time() - t_start:.1f}s", flush=True)


# ── CLI ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Build or query the guidelines vector DB (ChromaDB + local MedEmbed)"
    )
    parser.add_argument(
        "--source",
        metavar="SOURCE",
        help="Restrict indexing to one source (e.g. cma, nice, pubmed). Default: all.",
    )
    parser.add_argument(
        "--query",
        metavar="TEXT",
        help="Query the DB instead of building it.",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=5,
        metavar="N",
        help="Number of query results to return (default: 5).",
    )
    parser.add_argument(
        "--test",
        metavar="TEST",
        help="Filter query results by diagnostic test (e.g. ecg, arm_xray).",
    )
    parser.add_argument(
        "--parquet",
        metavar="PATH",
        help="Path to a pre-filtered parquet file to index (skips full dataset load).",
    )
    args = parser.parse_args()

    if args.query:
        results = query_guidelines(
            args.query, n_results=args.n, diagnostic_test=args.test
        )
        print(f"\nTop {len(results)} results for: '{args.query}'\n{'─' * 60}")
        for i, r in enumerate(results, 1):
            print(
                f"\n[{i}] score={r['score']:.4f}  "
                f"source={r['source']}  tests={r['diagnostic_tests']}"
            )
            print(f"    title:   {r['title'][:80]}")
            print(f"    url:     {r['url'][:80]}")
            print(f"    matched: {r['matched_text'][:200]}...")
            print(f"    context: {r['context'][:500]}...")
    else:
        build_index(source_filter=args.source, parquet_path=args.parquet)


if __name__ == "__main__":
    main()
