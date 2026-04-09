#!/usr/bin/env python3
"""
build_vector_db_chroma.py — Embed KG nodes from triage_knowledge_graph_enriched.pkl
into a local ChromaDB collection using local MedEmbed / MedEIR embeddings
(sentence-transformers, no API key required).

Each node is embedded as a short natural-language passage, enabling semantic
entity-linking from free-text clinical terms to KG nodes.

Index usage:
    python build_vector_db_chroma.py                        # default PKL path
    python build_vector_db_chroma.py --graph-pkl path.pkl   # custom graph
    python build_vector_db_chroma.py --force-rebuild        # drop and re-index
    python build_vector_db_chroma.py --dry-run              # print texts only

Query usage:
    python build_vector_db_chroma.py --query "chest pain diaphoresis"
    python build_vector_db_chroma.py --query "wrist fracture" --type Symptom --n 5

Model options (set MEDEMBED_MODEL env var):
    abhinand/MedEmbed-small-v0.1    384-dim  ~340 MB  fastest
    abhinand/MedEmbed-base-v0.1     768-dim  ~440 MB  balanced
    abhinand/MedEmbed-large-v0.1   1024-dim  ~1.3 GB  best quality (default)
    Thakkar-AI/MedEIR               768-dim            IR-tuned alternative
"""

import argparse
import os
import pickle
import uuid
from pathlib import Path

from dotenv import load_dotenv

# Load .env from knowledge-graphs/ first, then repo root as fallback
load_dotenv(Path(__file__).parent.parent / ".env", override=False)
load_dotenv(Path(__file__).parent.parent.parent / ".env", override=False)

# ── Constants ─────────────────────────────────────────────────────────────────
KG_COLLECTION_NAME = "kg_nodes"
EMBED_BATCH_SIZE   = 32    # sentences per model.encode() call
CHROMA_UPSERT_BATCH = 100  # nodes per collection.upsert() call

MEDEMBED_MODEL = os.environ.get("MEDEMBED_MODEL", "abhinand/MedEmbed-large-v0.1")

# Resolve CHROMA_PATH against the script's own directory so the DB location is
# stable regardless of the caller's working directory.  A relative path in the
# env var (e.g. "chroma_db" or "./chroma_db") is anchored to this file's parent;
# an absolute path is used as-is.
_raw_chroma_path = os.environ.get("CHROMA_PATH", "")
if _raw_chroma_path:
    _p = Path(_raw_chroma_path)
    CHROMA_PATH = str(_p if _p.is_absolute() else Path(__file__).parent / _p)
else:
    CHROMA_PATH = str(Path(__file__).parent / "chroma_db")

_KG_DIR      = Path(__file__).parent.parent
_KG_PKL_FILE = os.environ.get("KG_PKL_FILE", "triage_knowledge_graph_enriched.pkl")
DEFAULT_PKL  = str(_KG_DIR / _KG_PKL_FILE)

# Node-type prefix mapping — mirrors csv_mapping.py _PREFIX_TO_TYPE
_PREFIX_TO_TYPE = {
    "Condition":   "Condition",
    "Symptom":     "Symptom",
    "Vital":       "Vital_Sign_Threshold",
    "Demographic": "Demographic_Factor",
    "Risk Factor": "Risk_Factor",
    "Attribute":   "Clinical_Attribute",
    "MOI":         "Mechanism_of_Injury",
}
_EMBEDDABLE_PREFIXES = set(_PREFIX_TO_TYPE.keys())

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
            name=KG_COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        print(
            f"ChromaDB collection '{KG_COLLECTION_NAME}' ready at '{CHROMA_PATH}' "
            f"({_chroma_collection.count()} existing points)"
        )
    return _chroma_collection


# ── Text construction ─────────────────────────────────────────────────────────
def build_node_text(node_id: str, attrs: dict) -> str:
    """Construct a natural-language passage to embed for a KG node.

    The node type is intentionally excluded from the embedded text and stored
    only in ChromaDB metadata.  Including the type prefix (e.g. "Symptom:")
    shifts the embedding away from the pure clinical concept, which suppresses
    cosine similarity when queries are plain clinical terms (e.g. "fever"
    against "Symptom: Fever" scores ~0.75 instead of ~1.0).

    Format: "{Label}. Also known as: {synonyms}. Associated tests: {tests}. SNOMED: {code}."
    """
    _, label = node_id.split(": ", 1)

    parts = [label]

    synonyms = [s for s in (attrs.get("synonyms") or []) if s]
    if synonyms:
        parts.append("Also known as: " + ", ".join(synonyms))

    te = attrs.get("test_evidence") or []
    test_names = [t["test"] for t in te if isinstance(t, dict) and "test" in t]
    if test_names:
        parts.append("Associated tests: " + ", ".join(test_names))

    for code_key, code_label in [
        ("snomed_ca_code", "SNOMED"),
        ("icd10_code",     "ICD-10"),
        ("icd10ca_code",   "ICD-10-CA"),
        ("ebi_open_code",  "EBI"),
    ]:
        val = attrs.get(code_key) or ""
        if val:
            parts.append(f"{code_label}: {val}")

    return ". ".join(parts) + "."


# ── Embedding ─────────────────────────────────────────────────────────────────
def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed texts with the local MedEmbed model (symmetric — no prefix needed)."""
    model = _get_embed_model()
    vecs = model.encode(
        texts,
        batch_size=EMBED_BATCH_SIZE,
        show_progress_bar=len(texts) > EMBED_BATCH_SIZE,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return vecs.tolist()


def embed_query(query_text: str) -> list[float]:
    return embed_texts([query_text])[0]


# ── ChromaDB helpers ──────────────────────────────────────────────────────────
def _node_id_to_chroma_id(node_id: str) -> str:
    """Deterministic string ID from a node_id — enables idempotent upserts."""
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, node_id))


def _get_indexed_node_ids() -> set[str]:
    """Return the set of node_ids already present in the ChromaDB collection."""
    collection = _get_collection()
    if collection.count() == 0:
        return set()
    print("  Fetching existing node_ids for resume detection ...")
    result = collection.get(include=["metadatas"])
    return {m["node_id"] for m in result["metadatas"] if "node_id" in m}


# ── Query ─────────────────────────────────────────────────────────────────────
def query_kg_nodes(
    query_text: str,
    n_results: int = 5,
    node_type: str = None,
) -> list[dict]:
    """Return KG nodes semantically closest to query_text.

    Args:
        query_text: Free-text clinical term or phrase to match.
        n_results:  Number of results to return.
        node_type:  Optional exact filter on node_type
                    (e.g. 'Symptom', 'Condition', 'Vital_Sign_Threshold').

    Returns:
        List of dicts: score, node_id, label, node_type, text,
                       snomed_ca_code, icd10_code, icd10ca_code, ebi_open_code.
    """
    collection = _get_collection()
    total = collection.count()
    if total == 0:
        return []

    query_kwargs = dict(
        query_embeddings=[embed_query(query_text)],
        n_results=min(n_results, total),
        include=["metadatas", "documents", "distances"],
    )
    if node_type:
        query_kwargs["where"] = {"node_type": {"$eq": node_type}}

    response = collection.query(**query_kwargs)

    results = []
    for meta, doc, dist in zip(
        response["metadatas"][0],
        response["documents"][0],
        response["distances"][0],
    ):
        results.append(
            {
                "score":          1.0 - dist,   # cosine distance → similarity
                "node_id":        meta.get("node_id", ""),
                "label":          meta.get("label", ""),
                "node_type":      meta.get("node_type", ""),
                "text":           doc,
                "snomed_ca_code": meta.get("snomed_ca_code", ""),
                "icd10_code":     meta.get("icd10_code", ""),
                "icd10ca_code":   meta.get("icd10ca_code", ""),
                "ebi_open_code":  meta.get("ebi_open_code", ""),
            }
        )
    return results


# ── Indexing ──────────────────────────────────────────────────────────────────
def build_kg_index(
    graph_pkl_path: str = DEFAULT_PKL,
    force_rebuild: bool = False,
    dry_run: bool = False,
):
    """Load the KG PKL, embed all embeddable nodes, and upsert into ChromaDB.

    Args:
        graph_pkl_path: Path to the serialized NetworkX DiGraph PKL.
        force_rebuild:  Drop the existing collection and re-index from scratch.
        dry_run:        Print the first 10 node texts and exit — no DB calls.
    """
    print(f"Loading Knowledge Graph from: {graph_pkl_path}")
    with open(graph_pkl_path, "rb") as f:
        kg = pickle.load(f)
    print(f"Graph loaded: {kg.number_of_nodes()} nodes, {kg.number_of_edges()} edges.")

    # Filter to embeddable node types
    embeddable: list[tuple[str, dict]] = []
    for node_id in kg.nodes:
        node_str = str(node_id)
        for prefix in _EMBEDDABLE_PREFIXES:
            if node_str.startswith(f"{prefix}: "):
                embeddable.append((node_str, dict(kg.nodes[node_id])))
                break

    print(f"Embeddable nodes: {len(embeddable)} / {kg.number_of_nodes()}")

    if dry_run:
        print("\n--- Dry run: sample node texts ---")
        for node_id, attrs in embeddable[:10]:
            print(f"  {node_id!r}")
            print(f"    → {build_node_text(node_id, attrs)}\n")
        if len(embeddable) > 10:
            print(f"  ... and {len(embeddable) - 10} more nodes.")
        return

    # Force rebuild: drop and recreate the collection
    if force_rebuild:
        import chromadb
        client = chromadb.PersistentClient(
            path=CHROMA_PATH,
            settings=chromadb.Settings(anonymized_telemetry=False),
        )
        try:
            client.delete_collection(KG_COLLECTION_NAME)
            print(f"Deleted existing collection '{KG_COLLECTION_NAME}' for rebuild.")
        except Exception:
            pass
        global _chroma_collection
        _chroma_collection = None  # force re-init

    collection = _get_collection()

    # Resume: skip already-indexed nodes
    already_indexed = set() if force_rebuild else _get_indexed_node_ids()
    if already_indexed:
        print(f"  {len(already_indexed)} node(s) already indexed — will skip.")

    # Build list of (chroma_id, text, payload) for nodes not yet indexed
    to_index: list[tuple[str, str, dict]] = []
    for node_id, attrs in embeddable:
        if node_id in already_indexed:
            continue
        prefix = node_id.split(": ", 1)[0]
        label = node_id[len(prefix) + 2:]
        # Prefer the 'type' attribute stored on the node (authoritative); fall
        # back to the prefix mapping for nodes that don't carry a type attr.
        node_type = attrs.get("type") or _PREFIX_TO_TYPE.get(prefix, prefix)
        text = build_node_text(node_id, attrs)
        payload = {
            "node_id":        node_id,
            "label":          label,
            "node_type":      node_type,
            "snomed_ca_code": str(attrs.get("snomed_ca_code") or ""),
            "icd10_code":     str(attrs.get("icd10_code")     or ""),
            "icd10ca_code":   str(attrs.get("icd10ca_code")   or ""),
            "ebi_open_code":  str(attrs.get("ebi_open_code")  or ""),
        }
        to_index.append((_node_id_to_chroma_id(node_id), text, payload))

    if not to_index:
        print("Nothing to index — all nodes already present.")
        return

    print(f"Embedding and upserting {len(to_index)} nodes ...")
    indexed = 0
    for batch_start in range(0, len(to_index), CHROMA_UPSERT_BATCH):
        batch = to_index[batch_start : batch_start + CHROMA_UPSERT_BATCH]
        chroma_ids, texts, payloads = zip(*batch)
        vectors = embed_texts(list(texts))
        collection.upsert(
            ids=list(chroma_ids),
            embeddings=vectors,
            documents=list(texts),
            metadatas=list(payloads),
        )
        indexed += len(batch)
        print(f"  [{indexed}/{len(to_index)}] upserted", flush=True)

    print(f"\nDone. {indexed} nodes indexed in '{KG_COLLECTION_NAME}'.")


# ── CLI ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Embed KG nodes into ChromaDB using local MedEmbed / MedEIR."
    )
    parser.add_argument(
        "--graph-pkl",
        default=DEFAULT_PKL,
        metavar="PATH",
        help=f"Path to the KG pickle file (default: {_KG_PKL_FILE}).",
    )
    parser.add_argument(
        "--force-rebuild",
        action="store_true",
        help="Delete and re-index the kg_nodes collection from scratch.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the first 10 node texts and exit without writing to ChromaDB.",
    )
    parser.add_argument(
        "--query",
        metavar="TEXT",
        help="Query the collection instead of building it.",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=5,
        metavar="N",
        help="Number of query results to return (default: 5).",
    )
    parser.add_argument(
        "--type",
        metavar="NODE_TYPE",
        help="Filter query results by node_type (e.g. Symptom, Condition).",
    )
    args = parser.parse_args()

    if args.query:
        results = query_kg_nodes(args.query, n_results=args.n, node_type=args.type)
        print(f"\nTop {len(results)} results for: '{args.query}'\n{'─' * 60}")
        for i, r in enumerate(results, 1):
            print(f"\n[{i}] score={r['score']:.4f}  type={r['node_type']}")
            print(f"    node_id: {r['node_id']}")
            print(f"    text:    {r['text']}")
            if r["snomed_ca_code"]:
                print(f"    SNOMED:  {r['snomed_ca_code']}")
    else:
        build_kg_index(
            graph_pkl_path=args.graph_pkl,
            force_rebuild=args.force_rebuild,
            dry_run=args.dry_run,
        )


if __name__ == "__main__":
    main()
