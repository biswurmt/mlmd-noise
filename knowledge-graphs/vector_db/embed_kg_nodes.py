#!/usr/bin/env python3
"""
embed_kg_nodes.py — Embed KG nodes using Nebius e5-mistral-7b-instruct and
store them in a Qdrant Cloud collection named 'kg_nodes'.

Each node is embedded as a short descriptive passage, enabling semantic
entity-linking from free-text diagnoses to KG nodes in csv_mapping workflows.

Index usage:
    python embed_kg_nodes.py                        # default PKL from KG_PKL_FILE env
    python embed_kg_nodes.py --graph-pkl path.pkl   # custom graph
    python embed_kg_nodes.py --force-rebuild        # drop and re-index
    python embed_kg_nodes.py --dry-run              # print texts only, no Qdrant calls
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

# ── Constants ──────────────────────────────────────────────────────────────────
KG_COLLECTION_NAME = "kg_nodes"
NEBIUS_EMBEDDING_MODEL = os.environ.get(
    "NEBIUS_EMBEDDING_MODEL", "intfloat/e5-mistral-7b-instruct"
)
VECTOR_SIZE = 4096          # e5-mistral-7b-instruct output dimension
NEBIUS_EMBED_BATCH = 16     # small batches — e5-mistral is a large model
QDRANT_UPSERT_BATCH = 100   # PointStructs per upsert call

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

# ── Lazy-initialized clients ───────────────────────────────────────────────────
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


# ── Text construction ──────────────────────────────────────────────────────────
def build_node_text(node_id: str, attrs: dict) -> str:
    """Construct a natural-language passage to embed for a KG node.

    Format: "{NodeType}: {Label}. Associated tests: {tests}. SNOMED: {code}."
    test_evidence is the richest signal available (synonyms are currently empty).
    """
    prefix, label = node_id.split(": ", 1)
    node_type_raw = attrs.get("type", prefix)
    node_type_human = node_type_raw.replace("_", " ")

    parts = [f"{node_type_human}: {label}"]

    # Associated tests from test_evidence (primary enrichment signal)
    te = attrs.get("test_evidence") or []
    test_names = [t["test"] for t in te if isinstance(t, dict) and "test" in t]
    if test_names:
        parts.append("Associated tests: " + ", ".join(test_names))

    # Ontology codes (only when non-empty)
    for code_key, code_label in [
        ("snomed_ca_code", "SNOMED"),
        ("icd10_code",     "ICD-10"),
        ("icd10ca_code",   "ICD-10-CA"),
        ("ebi_open_code",  "EBI"),
    ]:
        val = attrs.get(code_key) or ""
        if val:
            parts.append(f"{code_label}: {val}")

    # Synonyms (empty on current nodes, but future-proofed)
    synonyms = [s for s in (attrs.get("synonyms") or []) if s]
    if synonyms:
        parts.append("Also known as: " + ", ".join(synonyms))

    return ". ".join(parts) + "."


# ── Embedding ──────────────────────────────────────────────────────────────────
def embed_passages(texts: list[str]) -> list[list[float]]:
    """Embed node texts at index time — prepends 'passage: ' for e5-mistral."""
    client = _get_embed_client()
    prefixed = ["passage: " + t for t in texts]
    results = []
    for i in range(0, len(prefixed), NEBIUS_EMBED_BATCH):
        batch = prefixed[i : i + NEBIUS_EMBED_BATCH]
        resp = client.embeddings.create(model=NEBIUS_EMBEDDING_MODEL, input=batch)
        results.extend([item.embedding for item in resp.data])
    return results


# ── Qdrant helpers ─────────────────────────────────────────────────────────────
def _node_uuid(node_id: str) -> str:
    """Deterministic UUID from a node_id string — enables idempotent upserts."""
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, node_id))


def ensure_kg_collection():
    """Create the kg_nodes Qdrant collection and payload indexes if needed."""
    from qdrant_client.models import Distance, PayloadSchemaType, VectorParams

    qdrant = _get_qdrant()
    if not qdrant.collection_exists(KG_COLLECTION_NAME):
        qdrant.create_collection(
            collection_name=KG_COLLECTION_NAME,
            vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
        )
        print(f"Created collection '{KG_COLLECTION_NAME}' (cosine, dim={VECTOR_SIZE})")
    else:
        info = qdrant.get_collection(KG_COLLECTION_NAME)
        print(f"Collection '{KG_COLLECTION_NAME}' already exists — {info.points_count} points")

    # Payload indexes for filtered semantic search
    for field, schema in [
        ("node_type",      PayloadSchemaType.KEYWORD),
        ("node_id",        PayloadSchemaType.KEYWORD),
        ("snomed_ca_code", PayloadSchemaType.KEYWORD),
    ]:
        qdrant.create_payload_index(
            collection_name=KG_COLLECTION_NAME,
            field_name=field,
            field_schema=schema,
        )


# ── Indexing ───────────────────────────────────────────────────────────────────
def build_kg_index(graph_pkl_path: str, force_rebuild: bool = False, dry_run: bool = False):
    """Load PKL, embed all embeddable nodes, and upsert into kg_nodes collection.

    Args:
        graph_pkl_path: Path to the serialized NetworkX DiGraph PKL file.
        force_rebuild:  If True, delete the existing collection and re-index all nodes.
        dry_run:        If True, print node texts but make no Qdrant API calls.
    """
    from qdrant_client.models import PointStruct

    print(f"Loading Knowledge Graph from: {graph_pkl_path}")
    with open(graph_pkl_path, "rb") as f:
        kg = pickle.load(f)
    print(f"Graph loaded: {kg.number_of_nodes()} nodes, {kg.number_of_edges()} edges.")

    # Filter to embeddable node types
    embeddable_nodes: list[tuple[str, dict]] = []
    for node_id in kg.nodes:
        node_str = str(node_id)
        for prefix in _EMBEDDABLE_PREFIXES:
            if node_str.startswith(f"{prefix}: "):
                embeddable_nodes.append((node_str, dict(kg.nodes[node_id])))
                break

    print(f"Embeddable nodes: {len(embeddable_nodes)} / {kg.number_of_nodes()}")

    if dry_run:
        print("\n--- Dry run: sample node texts ---")
        for node_id, attrs in embeddable_nodes[:10]:
            print(f"  {node_id!r}")
            print(f"    → {build_node_text(node_id, attrs)}\n")
        if len(embeddable_nodes) > 10:
            print(f"  ... and {len(embeddable_nodes) - 10} more nodes.")
        return

    # Force rebuild: drop existing collection
    qdrant = _get_qdrant()
    if force_rebuild and qdrant.collection_exists(KG_COLLECTION_NAME):
        qdrant.delete_collection(KG_COLLECTION_NAME)
        print(f"Deleted existing collection '{KG_COLLECTION_NAME}' for rebuild.")

    ensure_kg_collection()

    # Resume: fetch already-indexed node_ids to skip
    already_indexed: set[str] = set()
    if not force_rebuild:
        print("Fetching already-indexed node_ids for resume detection ...")
        offset = None
        while True:
            points, offset = qdrant.scroll(
                collection_name=KG_COLLECTION_NAME,
                with_payload=["node_id"],
                with_vectors=False,
                limit=1000,
                offset=offset,
            )
            for p in points:
                if p.payload and "node_id" in p.payload:
                    already_indexed.add(p.payload["node_id"])
            if offset is None:
                break
        if already_indexed:
            print(f"  {len(already_indexed)} node(s) already indexed — will skip.")

    # Build (node_id, text, payload) tuples for remaining nodes
    to_index: list[tuple[str, str, dict]] = []
    for node_id, attrs in embeddable_nodes:
        if node_id in already_indexed:
            continue
        prefix = node_id.split(": ", 1)[0]
        label = node_id[len(prefix) + 2:]
        node_type = _PREFIX_TO_TYPE.get(prefix, prefix)
        text = build_node_text(node_id, attrs)
        payload = {
            "node_id":       node_id,
            "label":         label,
            "node_type":     node_type,
            "text":          text,
            "snomed_ca_code": str(attrs.get("snomed_ca_code") or ""),
            "icd10_code":    str(attrs.get("icd10_code") or ""),
            "icd10ca_code":  str(attrs.get("icd10ca_code") or ""),
            "ebi_open_code": str(attrs.get("ebi_open_code") or ""),
        }
        to_index.append((node_id, text, payload))

    if not to_index:
        print("Nothing to index — all nodes already present.")
        return

    print(f"Embedding and upserting {len(to_index)} nodes ...")

    indexed = 0
    for batch_start in range(0, len(to_index), QDRANT_UPSERT_BATCH):
        batch = to_index[batch_start : batch_start + QDRANT_UPSERT_BATCH]
        node_ids, texts, payloads = zip(*batch)
        vectors = embed_passages(list(texts))
        points = [
            PointStruct(id=_node_uuid(nid), vector=vec, payload=pl)
            for nid, vec, pl in zip(node_ids, vectors, payloads)
        ]
        qdrant.upsert(collection_name=KG_COLLECTION_NAME, points=points)
        indexed += len(points)
        print(f"  [{indexed}/{len(to_index)}] upserted", flush=True)

    print(f"\nDone. {indexed} nodes indexed in '{KG_COLLECTION_NAME}'.")


# ── CLI ────────────────────────────────────────────────────────────────────────
def main():
    _KG_DIR = Path(__file__).parent.parent
    _KG_PKL_FILE = os.environ.get("KG_PKL_FILE", "triage_knowledge_graph_enriched.pkl")
    _DEFAULT_PKL = str(_KG_DIR / _KG_PKL_FILE)

    parser = argparse.ArgumentParser(
        description="Embed KG nodes into Qdrant Cloud using Nebius e5-mistral-7b-instruct."
    )
    parser.add_argument(
        "--graph-pkl",
        default=_DEFAULT_PKL,
        metavar="PATH",
        help=f"Path to the KG pickle file (default: {_KG_PKL_FILE} from KG_PKL_FILE env).",
    )
    parser.add_argument(
        "--force-rebuild",
        action="store_true",
        help="Delete and re-index the kg_nodes collection from scratch.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the first 10 node texts and exit without calling Qdrant.",
    )
    args = parser.parse_args()

    build_kg_index(
        graph_pkl_path=args.graph_pkl,
        force_rebuild=args.force_rebuild,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
