#!/usr/bin/env python3
"""
semantic_csv_mapping.py — Map patient diagnoses (dx column) to KG nodes using
semantic search against a Qdrant 'kg_nodes' collection built by
knowledge-graphs/vector_db/embed_kg_nodes.py.

This is a drop-in replacement for csv_mapping.py's entity-linking step:
the matching logic uses Nebius e5-mistral-7b-instruct embeddings instead of
substring matching, while graph traversal and validation are unchanged.

Usage:
    python semantic_csv_mapping.py --input-csv <path> [options]

Options:
    --input-csv         PATH   Input CSV with a 'dx' column (required)
    --output-csv        PATH   Enriched output CSV (default: patient_diagnoses_semantic.csv)
    --graph-pkl         PATH   Knowledge graph PKL (default: from KG_PKL_FILE env)
    --limit             N      Process only first N rows
    --resume                   Skip diagnoses already present in output CSV
    --min-nodes         N      Min matched nodes to include a test (default: 1)
    --semantic-top-k    K      Max KG nodes from semantic search (default: 5)
    --semantic-threshold F     Min cosine similarity 0–1 (default: 0.50)
"""

import argparse
import os
import pickle
import sys
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):  # silent no-op fallback
        return iterable

# Load .env from knowledge-graphs/ (same path as csv_mapping.py)
_current_dir = os.path.dirname(__file__)
_env_path = os.path.abspath(os.path.join(_current_dir, "..", "knowledge-graphs", ".env"))
load_dotenv(_env_path, override=True)

# Import shared helpers from csv_mapping (in the same directory)
sys.path.insert(0, _current_dir)
from kg_substring_csv_mapping import (
    _PREFIX_TO_TYPE,
    TEST_COLUMN_MAP,
    _run_validation,
    get_tests_for_node,
)

# ── Constants ──────────────────────────────────────────────────────────────────
KG_COLLECTION_NAME = "kg_nodes"
NEBIUS_EMBEDDING_MODEL = os.environ.get(
    "NEBIUS_EMBEDDING_MODEL", "intfloat/e5-mistral-7b-instruct"
)
_SEMANTIC_QUERY_INSTR = (
    "Instruct: Retrieve relevant medical knowledge graph nodes for the following "
    "clinical diagnosis.\nQuery: "
)
_SEMANTIC_TOP_K = 5
_SEMANTIC_MIN_SIM = 0.50

# ── Lazy-initialized clients ───────────────────────────────────────────────────
_embed_client = None
_qdrant_client = None


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
    global _qdrant_client
    if _qdrant_client is None:
        from qdrant_client import QdrantClient
        _qdrant_client = QdrantClient(
            url=os.environ["QDRANT_URL"],
            api_key=os.environ["QDRANT_API_KEY"],
        )
    return _qdrant_client


# ── Semantic entity linking ────────────────────────────────────────────────────
def get_graph_matches_semantic(
    raw_dx: str,
    top_k: int = _SEMANTIC_TOP_K,
    min_score: float = _SEMANTIC_MIN_SIM,
) -> list[dict]:
    """Semantically search kg_nodes for KG nodes relevant to raw_dx.

    Returns a list of {"label", "node_type", "confidence", "score"} dicts
    sorted by descending similarity score.

    Returns [] on any Qdrant/API error (soft failure — caller logs no-match).
    """
    try:
        client = _get_embed_client()
        vec = client.embeddings.create(
            model=NEBIUS_EMBEDDING_MODEL,
            input=[_SEMANTIC_QUERY_INSTR + raw_dx],
        ).data[0].embedding

        qdrant = _get_qdrant()
        hits = qdrant.query_points(
            collection_name=KG_COLLECTION_NAME,
            query=vec,
            limit=top_k,
            with_payload=True,
            score_threshold=min_score,
        ).points

        results = []
        seen: set[str] = set()
        for hit in hits:
            p = hit.payload
            key = f"{p['node_type']}:{p['label']}"
            if key not in seen:
                seen.add(key)
                results.append({
                    "label":      p["label"],
                    "node_type":  p["node_type"],
                    "confidence": round(hit.score * 100, 1),
                    "score":      hit.score,
                })
        return results

    except Exception as e:
        print(
            f"  [semantic] Qdrant unavailable ({type(e).__name__}: {e}) — "
            "returning no matches."
        )
        return []


# ── Output formatting ──────────────────────────────────────────────────────────
def _format_matched_nodes(matches: list[dict]) -> str:
    if not matches:
        return ""
    return " | ".join(
        f"{m['label']} ({m['node_type']}) [{m.get('confidence', '?')}%]"
        for m in matches
    )


def _format_potential_tests(matches: list[dict], min_nodes: int) -> str:
    test_vote_count: dict[str, int] = {}
    for m in matches:
        for t in m.get("tests", []):
            test_vote_count[t] = test_vote_count.get(t, 0) + 1
    tests = [t for t, count in test_vote_count.items() if count >= min_nodes]
    return ", ".join(tests) if tests else "No linked tests found"


# ── Main pipeline ──────────────────────────────────────────────────────────────
def _save_partial(df: pd.DataFrame, mapping_dictionary: dict, min_nodes: int, path: str) -> None:
    """Write current mapping results to disk without blocking the main loop."""
    df = df.copy()
    df["matched_graph_nodes"] = df["dx"].map(
        lambda dx: _format_matched_nodes(mapping_dictionary.get(dx, []))
    )
    df["potential_tests"] = df["dx"].map(
        lambda dx: _format_potential_tests(mapping_dictionary.get(dx, []), min_nodes)
    )
    df.to_csv(path, index=False)


def map_diagnoses_semantic(
    csv_input_path: str,
    graph_pkl_path: str,
    csv_output_path: str,
    row_limit: int | None = None,
    resume: bool = False,
    min_nodes: int = 1,
    semantic_top_k: int = _SEMANTIC_TOP_K,
    semantic_threshold: float = _SEMANTIC_MIN_SIM,
    save_every: int = 10,
):
    # Load KG
    print("Loading Knowledge Graph...")
    with open(graph_pkl_path, "rb") as f:
        kg = pickle.load(f)
    print(f"Graph loaded: {kg.number_of_nodes()} nodes, {kg.number_of_edges()} edges.")

    # Load CSV
    df = pd.read_csv(csv_input_path)
    print(list(df.columns))
    if "dx" not in df.columns:
        raise ValueError("CSV must contain a 'dx' column.")

    if row_limit is not None:
        print(f"  [--limit] Restricting to first {row_limit} rows for testing.")
        df = df.head(row_limit)

    # Resume: seed from existing output
    already_mapped: set[str] = set()
    mapping_dictionary: dict[str, list[dict]] = {}

    if resume and os.path.exists(csv_output_path):
        existing_df = pd.read_csv(csv_output_path)
        if "matched_graph_nodes" in existing_df.columns and "potential_tests" in existing_df.columns:
            for _, row in existing_df.dropna(subset=["matched_graph_nodes"]).iterrows():
                dx_val = row["dx"]
                if pd.isna(dx_val) or str(row.get("matched_graph_nodes", "")).strip() == "":
                    continue
                nodes_str = str(row["matched_graph_nodes"])
                tests_str = str(row["potential_tests"]) if not pd.isna(row.get("potential_tests")) else ""
                tests_list = [
                    t.strip()
                    for t in tests_str.split(",")
                    if t.strip() and t.strip() != "No linked tests found"
                ]
                enriched_resume = []
                for part in nodes_str.split(" | "):
                    part = part.strip()
                    # Strip trailing [score%] bracket added by _format_matched_nodes
                    if part.endswith("]") and "[" in part:
                        part = part[: part.rfind("[")].strip()
                    if "(" in part and part.endswith(")"):
                        lbl = part[: part.rfind("(")].strip()
                        ntype = part[part.rfind("(") + 1 : -1].strip()
                        enriched_resume.append(
                            {"label": lbl, "node_type": ntype, "tests": tests_list, "confidence": "?"}
                        )
                mapping_dictionary[dx_val] = enriched_resume
                already_mapped.add(dx_val)
            print(
                f"  [resume] Loaded {len(already_mapped)} already-mapped diagnoses "
                f"from '{csv_output_path}'."
            )
        else:
            print("  [resume] Output CSV found but missing expected columns — starting fresh.")

    if not resume and os.path.exists(csv_output_path):
        print(
            f"  [warning] Output file '{csv_output_path}' already exists but --resume was not set.\n"
            f"            Re-run with --resume to continue from the last checkpoint.\n"
            f"            Proceeding will OVERWRITE existing results."
        )

    unique_raw_diagnoses = df["dx"].dropna().unique()
    remaining = [d for d in unique_raw_diagnoses if d not in already_mapped]
    print(
        f"Found {len(unique_raw_diagnoses)} unique raw diagnoses "
        f"({len(already_mapped)} already mapped, {len(remaining)} to process)."
    )

    # Semantic matching
    print(
        f"Starting semantic matching (top_k={semantic_top_k}, "
        f"threshold={semantic_threshold}, save_every={save_every}) ..."
    )
    semantic_matched = 0
    semantic_unmatched = 0

    progress = tqdm(
        remaining,
        total=len(remaining),
        unit="dx",
        desc="Mapping",
        dynamic_ncols=True,
    )

    try:
        for i, raw_diag in enumerate(progress, 1):
            terms = [t.strip() for t in str(raw_diag).split(";") if t.strip()]

            seen_labels: set[str] = set()
            enriched: list[dict] = []

            for term in terms:
                term_matches = get_graph_matches_semantic(
                    term, top_k=semantic_top_k, min_score=semantic_threshold
                )
                if term_matches:
                    semantic_matched += 1
                else:
                    semantic_unmatched += 1

                for m in term_matches:
                    key = f"{m['node_type']}:{m['label']}"
                    if key not in seen_labels:
                        seen_labels.add(key)
                        tests = get_tests_for_node(kg, m["label"], m["node_type"])
                        enriched.append({**m, "tests": tests})

                labels = (
                    " | ".join(
                        f"{m['label']} ({m['node_type']}) [{m['confidence']:.1f}%]"
                        for m in term_matches
                    )
                    or "no match"
                )
                tqdm.write(f"  [semantic] '{term}' -> {labels}")

            mapping_dictionary[raw_diag] = enriched

            # Update progress bar suffix with live match rate
            total_so_far = semantic_matched + semantic_unmatched
            match_pct = semantic_matched / total_so_far * 100 if total_so_far else 0
            progress.set_postfix(matched=f"{match_pct:.1f}%", refresh=False)

            # Incremental save every save_every unique diagnoses
            if save_every > 0 and i % save_every == 0:
                _save_partial(df, mapping_dictionary, min_nodes, csv_output_path)
                tqdm.write(f"  [checkpoint] Saved {i}/{len(remaining)} diagnoses → '{csv_output_path}'")
    finally:
        # Always flush current state on any exit (crash, KeyboardInterrupt, etc.)
        if mapping_dictionary:
            _save_partial(df, mapping_dictionary, min_nodes, csv_output_path)
            tqdm.write(f"  [checkpoint] Final flush: {len(mapping_dictionary)} diagnoses saved → '{csv_output_path}'")

    total_terms = semantic_matched + semantic_unmatched
    print("\n--- Matching Summary ---")
    if total_terms:
        print(
            f"  Matched   : {semantic_matched} / {total_terms} terms "
            f"({semantic_matched / total_terms * 100:.1f}%)"
        )
        print(
            f"  Unmatched : {semantic_unmatched} / {total_terms} terms "
            f"({semantic_unmatched / total_terms * 100:.1f}%)"
        )
    else:
        print("  Matched   : 0")

    # Map back to DataFrame
    df["matched_graph_nodes"] = df["dx"].map(
        lambda dx: _format_matched_nodes(mapping_dictionary.get(dx, []))
    )
    df["potential_tests"] = df["dx"].map(
        lambda dx: _format_potential_tests(mapping_dictionary.get(dx, []), min_nodes)
    )

    # Save output
    df.to_csv(csv_output_path, index=False)
    print(
        f"\nSuccess! Saved enriched data to '{csv_output_path}' "
        f"(min_nodes={min_nodes})"
    )

    # Validation (runs when ground-truth columns are present)
    missing = [col for col in TEST_COLUMN_MAP.values() if col not in df.columns]
    if missing:
        print(f"\n[!] Validation skipped: ground-truth column(s) not found: {missing}")
    else:
        metrics_df = _run_validation(df)
        validation_path = csv_output_path.replace(".csv", "_validation.csv")
        metrics_df.to_csv(validation_path, index=False)
        print(f"\nValidation report saved to '{validation_path}'")
        df.to_csv(csv_output_path, index=False)

    return df


# ── CLI ────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    _KG_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "knowledge-graphs"))
    _KG_PKL_FILE = os.environ.get("KG_PKL_FILE", "triage_knowledge_graph_enriched.pkl")
    _DEFAULT_PKL = os.path.join(_KG_DIR, _KG_PKL_FILE)

    parser = argparse.ArgumentParser(
        description=(
            "Semantically map patient diagnoses (dx column) to knowledge-graph "
            "nodes using Nebius embeddings + Qdrant vector search."
        )
    )
    parser.add_argument(
        "--input-csv", required=True,
        help="Path to the input CSV file containing a 'dx' column.",
    )
    parser.add_argument(
        "--output-csv", default="patient_diagnoses_semantic.csv",
        help="Path for the enriched output CSV (default: patient_diagnoses_semantic.csv).",
    )
    parser.add_argument(
        "--graph-pkl",
        default=_DEFAULT_PKL,
        help=f"Path to the knowledge graph PKL file (default: {_KG_PKL_FILE} from KG_PKL_FILE env).",
    )
    parser.add_argument(
        "--limit", type=int, default=None, metavar="N",
        help="Process only the first N rows of the CSV.",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Skip diagnoses already present in the output CSV.",
    )
    parser.add_argument(
        "--min-nodes", type=int, default=1, metavar="N",
        help=(
            "Minimum number of matched KG nodes recommending a test before it is "
            "included in potential_tests. Default: 1."
        ),
    )
    parser.add_argument(
        "--semantic-top-k", type=int, default=_SEMANTIC_TOP_K, metavar="K",
        help=f"Max KG nodes to retrieve via semantic search (default: {_SEMANTIC_TOP_K}).",
    )
    parser.add_argument(
        "--semantic-threshold", type=float, default=_SEMANTIC_MIN_SIM, metavar="F",
        help=f"Min cosine similarity 0–1 for a match to be accepted (default: {_SEMANTIC_MIN_SIM}).",
    )
    parser.add_argument(
        "--save-every", type=int, default=10, metavar="N",
        help=(
            "Save partial results to the output CSV every N unique diagnoses processed. "
            "Enables --resume to pick up from the last checkpoint. Set 0 to disable. Default: 10."
        ),
    )
    args = parser.parse_args()

    result_df = map_diagnoses_semantic(
        csv_input_path=args.input_csv,
        graph_pkl_path=args.graph_pkl,
        csv_output_path=args.output_csv,
        row_limit=args.limit,
        resume=args.resume,
        min_nodes=args.min_nodes,
        semantic_top_k=args.semantic_top_k,
        semantic_threshold=args.semantic_threshold,
        save_every=args.save_every,
    )

    print("\n--- Output Preview ---")
    print(result_df[["dx", "matched_graph_nodes", "potential_tests"]].head())

#!/usr/bin/env python3
"""
semantic_csv_mapping.py — Map patient diagnoses (dx column) to KG nodes using
semantic search against a Qdrant 'kg_nodes' collection built by
knowledge-graphs/vector_db/embed_kg_nodes.py.

This is a drop-in replacement for csv_mapping.py's entity-linking step:
the matching logic uses Nebius e5-mistral-7b-instruct embeddings instead of
substring matching, while graph traversal and validation are unchanged.

Usage:
    python semantic_csv_mapping.py --input-csv <path> [options]

Options:
    --input-csv         PATH   Input CSV with a 'dx' column (required)
    --output-csv        PATH   Enriched output CSV (default: patient_diagnoses_semantic.csv)
    --graph-pkl         PATH   Knowledge graph PKL (default: from KG_PKL_FILE env)
    --limit             N      Process only first N rows
    --resume                   Skip diagnoses already present in output CSV
    --min-nodes         N      Min matched nodes to include a test (default: 1)
    --semantic-top-k    K      Max KG nodes from semantic search (default: 5)
    --semantic-threshold F     Min cosine similarity 0–1 (default: 0.50)
"""

import argparse
import os
import pickle
import sys
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):  # silent no-op fallback
        return iterable

# Load .env from knowledge-graphs/ (same path as csv_mapping.py)
_current_dir = os.path.dirname(__file__)
_env_path = os.path.abspath(os.path.join(_current_dir, "..", "knowledge-graphs", ".env"))
load_dotenv(_env_path, override=True)

# Import shared helpers from csv_mapping (in the same directory)
sys.path.insert(0, _current_dir)
from csv_mapping import (
    _PREFIX_TO_TYPE,
    TEST_COLUMN_MAP,
    _run_validation,
    get_tests_for_node,
)

# ── Constants ──────────────────────────────────────────────────────────────────
KG_COLLECTION_NAME = "kg_nodes"
NEBIUS_EMBEDDING_MODEL = os.environ.get(
    "NEBIUS_EMBEDDING_MODEL", "intfloat/e5-mistral-7b-instruct"
)
_SEMANTIC_QUERY_INSTR = (
    "Instruct: Retrieve relevant medical knowledge graph nodes for the following "
    "clinical diagnosis.\nQuery: "
)
_SEMANTIC_TOP_K = 5
_SEMANTIC_MIN_SIM = 0.50

# ── Lazy-initialized clients ───────────────────────────────────────────────────
_embed_client = None
_qdrant_client = None


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
    global _qdrant_client
    if _qdrant_client is None:
        from qdrant_client import QdrantClient
        _qdrant_client = QdrantClient(
            url=os.environ["QDRANT_URL"],
            api_key=os.environ["QDRANT_API_KEY"],
        )
    return _qdrant_client


# ── Semantic entity linking ────────────────────────────────────────────────────
def get_graph_matches_semantic(
    raw_dx: str,
    top_k: int = _SEMANTIC_TOP_K,
    min_score: float = _SEMANTIC_MIN_SIM,
) -> list[dict]:
    """Semantically search kg_nodes for KG nodes relevant to raw_dx.

    Returns a list of {"label", "node_type", "confidence", "score"} dicts
    sorted by descending similarity score.

    Returns [] on any Qdrant/API error (soft failure — caller logs no-match).
    """
    try:
        client = _get_embed_client()
        vec = client.embeddings.create(
            model=NEBIUS_EMBEDDING_MODEL,
            input=[_SEMANTIC_QUERY_INSTR + raw_dx],
        ).data[0].embedding

        qdrant = _get_qdrant()
        hits = qdrant.query_points(
            collection_name=KG_COLLECTION_NAME,
            query=vec,
            limit=top_k,
            with_payload=True,
            score_threshold=min_score,
        ).points

        results = []
        seen: set[str] = set()
        for hit in hits:
            p = hit.payload
            key = f"{p['node_type']}:{p['label']}"
            if key not in seen:
                seen.add(key)
                results.append({
                    "label":      p["label"],
                    "node_type":  p["node_type"],
                    "confidence": round(hit.score * 100, 1),
                    "score":      hit.score,
                })
        return results

    except Exception as e:
        print(
            f"  [semantic] Qdrant unavailable ({type(e).__name__}: {e}) — "
            "returning no matches."
        )
        return []


# ── Output formatting ──────────────────────────────────────────────────────────
def _format_matched_nodes(matches: list[dict]) -> str:
    if not matches:
        return ""
    return " | ".join(
        f"{m['label']} ({m['node_type']}) [{m.get('confidence', '?')}%]"
        for m in matches
    )


def _format_potential_tests(matches: list[dict], min_nodes: int) -> str:
    test_vote_count: dict[str, int] = {}
    for m in matches:
        for t in m.get("tests", []):
            test_vote_count[t] = test_vote_count.get(t, 0) + 1
    tests = [t for t, count in test_vote_count.items() if count >= min_nodes]
    return ", ".join(tests) if tests else "No linked tests found"


# ── Main pipeline ──────────────────────────────────────────────────────────────
def _save_partial(df: pd.DataFrame, mapping_dictionary: dict, min_nodes: int, path: str) -> None:
    """Write current mapping results to disk without blocking the main loop."""
    df = df.copy()
    df["matched_graph_nodes"] = df["dx"].map(
        lambda dx: _format_matched_nodes(mapping_dictionary.get(dx, []))
    )
    df["potential_tests"] = df["dx"].map(
        lambda dx: _format_potential_tests(mapping_dictionary.get(dx, []), min_nodes)
    )
    df.to_csv(path, index=False)


def map_diagnoses_semantic(
    csv_input_path: str,
    graph_pkl_path: str,
    csv_output_path: str,
    row_limit: int | None = None,
    resume: bool = False,
    min_nodes: int = 1,
    semantic_top_k: int = _SEMANTIC_TOP_K,
    semantic_threshold: float = _SEMANTIC_MIN_SIM,
    save_every: int = 10,
):
    # Load KG
    print("Loading Knowledge Graph...")
    with open(graph_pkl_path, "rb") as f:
        kg = pickle.load(f)
    print(f"Graph loaded: {kg.number_of_nodes()} nodes, {kg.number_of_edges()} edges.")

    # Load CSV
    df = pd.read_csv(csv_input_path)
    print(list(df.columns))
    if "dx" not in df.columns:
        raise ValueError("CSV must contain a 'dx' column.")

    if row_limit is not None:
        print(f"  [--limit] Restricting to first {row_limit} rows for testing.")
        df = df.head(row_limit)

    # Resume: seed from existing output
    already_mapped: set[str] = set()
    mapping_dictionary: dict[str, list[dict]] = {}

    if resume and os.path.exists(csv_output_path):
        existing_df = pd.read_csv(csv_output_path)
        if "matched_graph_nodes" in existing_df.columns and "potential_tests" in existing_df.columns:
            for _, row in existing_df.dropna(subset=["matched_graph_nodes"]).iterrows():
                dx_val = row["dx"]
                if pd.isna(dx_val) or str(row.get("matched_graph_nodes", "")).strip() == "":
                    continue
                nodes_str = str(row["matched_graph_nodes"])
                tests_str = str(row["potential_tests"]) if not pd.isna(row.get("potential_tests")) else ""
                tests_list = [
                    t.strip()
                    for t in tests_str.split(",")
                    if t.strip() and t.strip() != "No linked tests found"
                ]
                enriched_resume = []
                for part in nodes_str.split(" | "):
                    part = part.strip()
                    # Strip trailing [score%] bracket added by _format_matched_nodes
                    if part.endswith("]") and "[" in part:
                        part = part[: part.rfind("[")].strip()
                    if "(" in part and part.endswith(")"):
                        lbl = part[: part.rfind("(")].strip()
                        ntype = part[part.rfind("(") + 1 : -1].strip()
                        enriched_resume.append(
                            {"label": lbl, "node_type": ntype, "tests": tests_list, "confidence": "?"}
                        )
                mapping_dictionary[dx_val] = enriched_resume
                already_mapped.add(dx_val)
            print(
                f"  [resume] Loaded {len(already_mapped)} already-mapped diagnoses "
                f"from '{csv_output_path}'."
            )
        else:
            print("  [resume] Output CSV found but missing expected columns — starting fresh.")

    if not resume and os.path.exists(csv_output_path):
        print(
            f"  [warning] Output file '{csv_output_path}' already exists but --resume was not set.\n"
            f"            Re-run with --resume to continue from the last checkpoint.\n"
            f"            Proceeding will OVERWRITE existing results."
        )

    unique_raw_diagnoses = df["dx"].dropna().unique()
    remaining = [d for d in unique_raw_diagnoses if d not in already_mapped]
    print(
        f"Found {len(unique_raw_diagnoses)} unique raw diagnoses "
        f"({len(already_mapped)} already mapped, {len(remaining)} to process)."
    )

    # Semantic matching
    print(
        f"Starting semantic matching (top_k={semantic_top_k}, "
        f"threshold={semantic_threshold}, save_every={save_every}) ..."
    )
    semantic_matched = 0
    semantic_unmatched = 0

    progress = tqdm(
        remaining,
        total=len(remaining),
        unit="dx",
        desc="Mapping",
        dynamic_ncols=True,
    )

    try:
        for i, raw_diag in enumerate(progress, 1):
            terms = [t.strip() for t in str(raw_diag).split(";") if t.strip()]

            seen_labels: set[str] = set()
            enriched: list[dict] = []

            for term in terms:
                term_matches = get_graph_matches_semantic(
                    term, top_k=semantic_top_k, min_score=semantic_threshold
                )
                if term_matches:
                    semantic_matched += 1
                else:
                    semantic_unmatched += 1

                for m in term_matches:
                    key = f"{m['node_type']}:{m['label']}"
                    if key not in seen_labels:
                        seen_labels.add(key)
                        tests = get_tests_for_node(kg, m["label"], m["node_type"])
                        enriched.append({**m, "tests": tests})

                labels = (
                    " | ".join(
                        f"{m['label']} ({m['node_type']}) [{m['confidence']:.1f}%]"
                        for m in term_matches
                    )
                    or "no match"
                )
                tqdm.write(f"  [semantic] '{term}' -> {labels}")

            mapping_dictionary[raw_diag] = enriched

            # Update progress bar suffix with live match rate
            total_so_far = semantic_matched + semantic_unmatched
            match_pct = semantic_matched / total_so_far * 100 if total_so_far else 0
            progress.set_postfix(matched=f"{match_pct:.1f}%", refresh=False)

            # Incremental save every save_every unique diagnoses
            if save_every > 0 and i % save_every == 0:
                _save_partial(df, mapping_dictionary, min_nodes, csv_output_path)
                tqdm.write(f"  [checkpoint] Saved {i}/{len(remaining)} diagnoses → '{csv_output_path}'")
    finally:
        # Always flush current state on any exit (crash, KeyboardInterrupt, etc.)
        if mapping_dictionary:
            _save_partial(df, mapping_dictionary, min_nodes, csv_output_path)
            tqdm.write(f"  [checkpoint] Final flush: {len(mapping_dictionary)} diagnoses saved → '{csv_output_path}'")

    total_terms = semantic_matched + semantic_unmatched
    print("\n--- Matching Summary ---")
    if total_terms:
        print(
            f"  Matched   : {semantic_matched} / {total_terms} terms "
            f"({semantic_matched / total_terms * 100:.1f}%)"
        )
        print(
            f"  Unmatched : {semantic_unmatched} / {total_terms} terms "
            f"({semantic_unmatched / total_terms * 100:.1f}%)"
        )
    else:
        print("  Matched   : 0")

    # Map back to DataFrame
    df["matched_graph_nodes"] = df["dx"].map(
        lambda dx: _format_matched_nodes(mapping_dictionary.get(dx, []))
    )
    df["potential_tests"] = df["dx"].map(
        lambda dx: _format_potential_tests(mapping_dictionary.get(dx, []), min_nodes)
    )

    # Save output
    df.to_csv(csv_output_path, index=False)
    print(
        f"\nSuccess! Saved enriched data to '{csv_output_path}' "
        f"(min_nodes={min_nodes})"
    )

    # Validation (runs when ground-truth columns are present)
    missing = [col for col in TEST_COLUMN_MAP.values() if col not in df.columns]
    if missing:
        print(f"\n[!] Validation skipped: ground-truth column(s) not found: {missing}")
    else:
        metrics_df = _run_validation(df)
        validation_path = csv_output_path.replace(".csv", "_validation.csv")
        metrics_df.to_csv(validation_path, index=False)
        print(f"\nValidation report saved to '{validation_path}'")
        df.to_csv(csv_output_path, index=False)

    return df


# ── CLI ────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    _KG_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "knowledge-graphs"))
    _KG_PKL_FILE = os.environ.get("KG_PKL_FILE", "triage_knowledge_graph_enriched.pkl")
    _DEFAULT_PKL = os.path.join(_KG_DIR, _KG_PKL_FILE)

    parser = argparse.ArgumentParser(
        description=(
            "Semantically map patient diagnoses (dx column) to knowledge-graph "
            "nodes using Nebius embeddings + Qdrant vector search."
        )
    )
    parser.add_argument(
        "--input-csv", required=True,
        help="Path to the input CSV file containing a 'dx' column.",
    )
    parser.add_argument(
        "--output-csv", default="patient_diagnoses_semantic.csv",
        help="Path for the enriched output CSV (default: patient_diagnoses_semantic.csv).",
    )
    parser.add_argument(
        "--graph-pkl",
        default=_DEFAULT_PKL,
        help=f"Path to the knowledge graph PKL file (default: {_KG_PKL_FILE} from KG_PKL_FILE env).",
    )
    parser.add_argument(
        "--limit", type=int, default=None, metavar="N",
        help="Process only the first N rows of the CSV.",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Skip diagnoses already present in the output CSV.",
    )
    parser.add_argument(
        "--min-nodes", type=int, default=1, metavar="N",
        help=(
            "Minimum number of matched KG nodes recommending a test before it is "
            "included in potential_tests. Default: 1."
        ),
    )
    parser.add_argument(
        "--semantic-top-k", type=int, default=_SEMANTIC_TOP_K, metavar="K",
        help=f"Max KG nodes to retrieve via semantic search (default: {_SEMANTIC_TOP_K}).",
    )
    parser.add_argument(
        "--semantic-threshold", type=float, default=_SEMANTIC_MIN_SIM, metavar="F",
        help=f"Min cosine similarity 0–1 for a match to be accepted (default: {_SEMANTIC_MIN_SIM}).",
    )
    parser.add_argument(
        "--save-every", type=int, default=10, metavar="N",
        help=(
            "Save partial results to the output CSV every N unique diagnoses processed. "
            "Enables --resume to pick up from the last checkpoint. Set 0 to disable. Default: 10."
        ),
    )
    args = parser.parse_args()

    result_df = map_diagnoses_semantic(
        csv_input_path=args.input_csv,
        graph_pkl_path=args.graph_pkl,
        csv_output_path=args.output_csv,
        row_limit=args.limit,
        resume=args.resume,
        min_nodes=args.min_nodes,
        semantic_top_k=args.semantic_top_k,
        semantic_threshold=args.semantic_threshold,
        save_every=args.save_every,
    )

    print("\n--- Output Preview ---")
    print(result_df[["dx", "matched_graph_nodes", "potential_tests"]].head())
