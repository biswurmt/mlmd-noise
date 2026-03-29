#!/usr/bin/env python3
"""
build_vector_db.py — Build and query a Qdrant Cloud vector database from
filtered medical guidelines using Nebius e5-mistral-7b-instruct embeddings.

Index usage:
    python build_vector_db.py                    # full combined dataset
    python build_vector_db.py --source cma       # gauge run: CMA only (~86 docs)

Query usage:
    python build_vector_db.py --query "ST elevation myocardial infarction" --n 5
    python build_vector_db.py --query "scrotal pain sudden onset" --test testicular_ultrasound
"""

import argparse
import os
import time
import uuid
from pathlib import Path

from dotenv import load_dotenv

# Load .env from knowledge-graphs/ first, then repo root as fallback
load_dotenv(Path(__file__).parent.parent / ".env", override=False)
load_dotenv(Path(__file__).parent.parent.parent / ".env", override=False)

# ── Constants ─────────────────────────────────────────────────────────────────
MIN_TEXT_LENGTH = 200          # chars — drop stub/empty docs
CHUNK_SIZE = 2000              # chars ≈ 500 tokens
CHUNK_OVERLAP = 200            # chars overlap between consecutive chunks
COLLECTION_NAME = "guidelines"
NEBIUS_EMBEDDING_MODEL = os.environ.get(
    "NEBIUS_EMBEDDING_MODEL", "intfloat/e5-mistral-7b-instruct"
)
VECTOR_SIZE = 4096             # e5-mistral-7b-instruct output dimension
NEBIUS_EMBED_BATCH = 16        # small batches — e5-mistral is a large model
QDRANT_UPSERT_BATCH = 100      # PointStructs per upsert call
QUERY_INSTRUCTION = (
    "Instruct: Retrieve highly technical passages that answer the following query.\nQuery: "
)
DATA_PATH = Path(__file__).parent.parent / "data" / "guidelines_filtered" / "combined"
CHECKPOINT_PATH = Path(__file__).parent / ".index_checkpoint.json"

# ── Lazy-initialized clients ──────────────────────────────────────────────────
_embed_client = None
_qdrant = None


def _get_embed_client():
    global _embed_client
    if _embed_client is None:
        from openai import OpenAI
        _embed_client = OpenAI(
            base_url="https://api.studio.nebius.ai/v1/",
            api_key=os.environ["NEBIUS_API_KEY"],
        )
    return _embed_client


def _get_qdrant():
    global _qdrant
    if _qdrant is None:
        from qdrant_client import QdrantClient
        _qdrant = QdrantClient(
            url=os.environ["QDRANT_URL"],
            api_key=os.environ["QDRANT_API_KEY"],
        )
    return _qdrant


# ── Embedding helpers ─────────────────────────────────────────────────────────
def embed_passages(texts: list[str]) -> list[list[float]]:
    """Embed document chunks at index time — prepends 'passage: ' for e5-mistral."""
    client = _get_embed_client()
    prefixed = ["passage: " + t for t in texts]
    results = []
    for i in range(0, len(prefixed), NEBIUS_EMBED_BATCH):
        batch = prefixed[i : i + NEBIUS_EMBED_BATCH]
        resp = client.embeddings.create(model=NEBIUS_EMBEDDING_MODEL, input=batch)
        results.extend([item.embedding for item in resp.data])
    return results


def embed_query(query_text: str) -> list[float]:
    """Embed a query at retrieval time — prepends the task instruction for e5-mistral."""
    client = _get_embed_client()
    resp = client.embeddings.create(
        model=NEBIUS_EMBEDDING_MODEL,
        input=[QUERY_INSTRUCTION + query_text],
    )
    return resp.data[0].embedding


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


# ── Qdrant helpers ────────────────────────────────────────────────────────────
def _chunk_uuid(chunk_id: str) -> str:
    """Deterministic UUID from a chunk string ID — enables idempotent upserts."""
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk_id))


def ensure_collection():
    """Create the Qdrant collection and payload indexes if they don't already exist."""
    from qdrant_client.models import Distance, PayloadSchemaType, VectorParams
    qdrant = _get_qdrant()
    if not qdrant.collection_exists(COLLECTION_NAME):
        qdrant.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
        )
        print(f"Created collection '{COLLECTION_NAME}' (cosine, dim={VECTOR_SIZE})")
    else:
        info = qdrant.get_collection(COLLECTION_NAME)
        print(f"Collection '{COLLECTION_NAME}' already exists — {info.points_count} points")

    # Payload indexes required for scroll filters used in context expansion
    qdrant.create_payload_index(
        collection_name=COLLECTION_NAME,
        field_name="doc_id",
        field_schema=PayloadSchemaType.KEYWORD,
    )
    qdrant.create_payload_index(
        collection_name=COLLECTION_NAME,
        field_name="chunk_index",
        field_schema=PayloadSchemaType.INTEGER,
    )


# ── Query ─────────────────────────────────────────────────────────────────────
def _fetch_surrounding_chunks(doc_id: str, chunk_index: int, window: int = 1) -> str:
    """
    Fetch the chunk before and after a matched chunk from the same document,
    then stitch them into a single context string.
    """
    from qdrant_client.models import FieldCondition, Filter, MatchValue, Range

    qdrant = _get_qdrant()
    points, _ = qdrant.scroll(
        collection_name=COLLECTION_NAME,
        scroll_filter=Filter(
            must=[
                FieldCondition(key="doc_id", match=MatchValue(value=doc_id)),
                FieldCondition(
                    key="chunk_index",
                    range=Range(
                        gte=max(0, chunk_index - window),
                        lte=chunk_index + window,
                    ),
                ),
            ]
        ),
        with_payload=True,
        limit=window * 2 + 1,
    )
    points.sort(key=lambda p: p.payload.get("chunk_index", 0))
    return " ".join(p.payload.get("text", "") for p in points)


def query_guidelines(
    query_text: str,
    n_results: int = 5,
    diagnostic_test: str = None,
    context_window: int = 1,
) -> list[dict]:
    """
    Return top-N guideline chunks semantically relevant to query_text,
    each expanded with surrounding chunks for fuller context.

    Args:
        query_text:      Natural-language clinical query.
        n_results:       Number of results to return.
        diagnostic_test: Optional filter, e.g. 'ecg', 'arm_xray'.
        context_window:  Number of chunks before/after each hit to include (default: 1).

    Returns:
        List of dicts with keys: score, matched_text, context, title, source, url, diagnostic_tests.
    """
    from qdrant_client.models import FieldCondition, Filter, MatchText

    qdrant = _get_qdrant()
    query_vec = embed_query(query_text)

    search_filter = None
    if diagnostic_test:
        search_filter = Filter(
            must=[
                FieldCondition(
                    key="diagnostic_tests",
                    match=MatchText(text=diagnostic_test),
                )
            ]
        )

    response = qdrant.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vec,
        query_filter=search_filter,
        limit=n_results,
        with_payload=True,
    )

    results = []
    for hit in response.points:
        doc_id = hit.payload.get("doc_id", "")
        chunk_index = hit.payload.get("chunk_index", 0)
        context = _fetch_surrounding_chunks(doc_id, chunk_index, window=context_window)
        results.append(
            {
                "score": hit.score,
                "matched_text": hit.payload.get("text", ""),
                "context": context,
                "title": hit.payload.get("title", ""),
                "source": hit.payload.get("source", ""),
                "url": hit.payload.get("url", ""),
                "diagnostic_tests": hit.payload.get("diagnostic_tests", ""),
            }
        )
    return results


# ── Indexing ──────────────────────────────────────────────────────────────────
def build_index(source_filter: str = None, parquet_path: str = None):
    """Load, filter, chunk, embed, and upsert guidelines into Qdrant.

    Args:
        source_filter: Restrict to a single source (e.g. 'cma'). Only used
                       when loading from the full Arrow dataset.
        parquet_path:  Path to a pre-filtered parquet file (e.g. from a
                       GitHub release). Skips the Arrow dataset load and the
                       min-length/dedup filters — the parquet is already clean.
    """
    import pandas as pd
    from qdrant_client.models import PointStruct

    # 1. Load dataset — parquet (CI/resume) or full Arrow dataset (local)
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

    # 5. Ensure Qdrant collection exists
    ensure_collection()

    # 6. Fetch already-indexed doc_ids from Qdrant to enable safe resume
    print("Fetching indexed doc_ids from Qdrant for resume detection ...")
    indexed_doc_ids: set[str] = set()
    offset = None
    while True:
        points, offset = _get_qdrant().scroll(
            collection_name=COLLECTION_NAME,
            with_payload=["doc_id"],
            with_vectors=False,
            limit=1000,
            offset=offset,
        )
        for p in points:
            if p.payload and "doc_id" in p.payload:
                indexed_doc_ids.add(p.payload["doc_id"])
        if offset is None:
            break
    if indexed_doc_ids:
        print(f"  {len(indexed_doc_ids)} doc(s) already indexed — will skip.")

    # 7. Stream through docs: chunk → embed → upsert without holding all chunks in memory
    print(f"Streaming {len(df)} docs → chunking, embedding, upserting ...")
    qdrant = _get_qdrant()
    indexed = 0
    t_start = time.time()
    pending: list[tuple[str, str, dict]] = []  # (chunk_id, chunk_text, payload)

    def _flush(batch: list[tuple[str, str, dict]], total_so_far: int) -> int:
        chunk_ids, chunk_texts, payloads = zip(*batch)
        vectors = embed_passages(list(chunk_texts))
        points = [
            PointStruct(id=_chunk_uuid(cid), vector=vec, payload=payload)
            for cid, vec, payload in zip(chunk_ids, vectors, payloads)
        ]
        qdrant.upsert(collection_name=COLLECTION_NAME, points=points)
        n = len(points)
        elapsed = time.time() - t_start
        rate = (total_so_far + n) / elapsed if elapsed > 0 else 0
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
                "doc_id": doc_id,
                "title": str(row.get("title", ""))[:500],
                "source": str(row.get("source", ""))[:200],
                "url": str(row.get("url", ""))[:500],
                "overview": str(row.get("overview", ""))[:1000],
                "diagnostic_tests": str(row.get("diagnostic_tests", "")),
                "chunk_index": i,
                "text": chunk,
            }
            pending.append((f"{doc_id}_chunk_{i}", chunk, payload))

        if len(pending) >= QDRANT_UPSERT_BATCH:
            indexed += _flush(pending[:QDRANT_UPSERT_BATCH], indexed)
            pending = pending[QDRANT_UPSERT_BATCH:]

        if doc_num % 100 == 0:
            print(f"  docs processed: {doc_num}/{len(df)}", flush=True)

    # flush remainder
    while pending:
        batch, pending = pending[:QDRANT_UPSERT_BATCH], pending[QDRANT_UPSERT_BATCH:]
        indexed += _flush(batch, indexed)

    print(f"\nDone. {indexed} chunks indexed in {time.time() - t_start:.1f}s", flush=True)


# ── CLI ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Build or query the guidelines vector DB (Qdrant Cloud + Nebius embeddings)"
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
