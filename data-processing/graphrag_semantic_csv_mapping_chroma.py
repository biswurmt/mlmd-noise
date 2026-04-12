#!/usr/bin/env python3
"""
chroma_semantic_csv_mapping.py — Map patient diagnoses (dx column) to KG nodes
using semantic search against the local ChromaDB 'kg_nodes' collection built by
knowledge-graphs/vector_db/build_vector_db_chroma.py.

Drop-in replacement for graphrag_semantic_csv_mapping.py:
  - Nebius cloud embeddings  →  local MedEmbed / MedEIR (sentence-transformers)
  - Qdrant Cloud             →  ChromaDB local persistent store
  - Graph traversal, validation, and CSV schema are unchanged.

New output columns compared to the Qdrant version:
  match_scores   Pipe-separated "Label: 0.XX" scores for every matched node,
                 preserving the raw cosine similarity above the threshold for
                 downstream probability-aware use.
  test_scores    Pipe-separated "Test: aggregate_score" showing the sum of
                 cosine similarities across all nodes that recommend each test.
                 Tests are ordered by descending aggregate score.

Negation handling:
  Matched nodes are discarded when the node label appears in the dx string
  preceded by a negation word (no, not, denies, without, absent, negative for,
  rules out) within a 6-token window.
  Example: "no fever" → Fever node is NOT matched.

Usage:
    python chroma_semantic_csv_mapping.py --input-csv <path> [options]

Options:
    --input-csv          PATH   Input CSV with a 'dx' column (required)
    --output-csv         PATH   Enriched output CSV (default: patient_diagnoses_chroma.csv)
    --graph-pkl          PATH   Knowledge graph PKL (default: from KG_PKL_FILE env)
    --limit              N      Process only first N rows
    --resume                    Skip diagnoses already present in output CSV
    --min-score-sum      F      Min aggregate cosine score for a test to appear in
                                potential_tests (default: 0.5). Replaces --min-nodes.
    --semantic-top-k     K      Max KG nodes from ChromaDB query (default: 5)
    --semantic-threshold F      Min cosine similarity 0–1 to accept a match (default: 0.50)
    --save-every         N      Checkpoint to disk every N unique diagnoses (default: 10)
"""

import argparse
import os
import pickle
import re
import sys
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
import json

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable

# ── Path setup ────────────────────────────────────────────────────────────────
_current_dir = os.path.dirname(os.path.abspath(__file__))
_kg_dir      = os.path.abspath(os.path.join(_current_dir, "..", "knowledge-graphs"))
_vector_dir  = os.path.join(_kg_dir, "vector_db")

# Load .env from knowledge-graphs/ (same as csv_mapping.py)
load_dotenv(os.path.join(_kg_dir, ".env"), override=True)

# Shared helpers from the substring mapping script (same directory)
sys.path.insert(0, _current_dir)
from kg_substring_csv_mapping import (
    TEST_COLUMN_MAP,
    _run_validation,
    get_tests_for_node,
)

# ChromaDB collection + MedEmbed model from the vector_db package
sys.path.insert(0, _vector_dir)
from build_vector_db_chroma import (
    _get_collection,
    embed_query,
)

# ── Constants ─────────────────────────────────────────────────────────────────
KG_COLLECTION_NAME  = "kg_nodes"
_SEMANTIC_TOP_K     = 5
_SEMANTIC_MIN_SIM   = 0.50
_MIN_SCORE_SUM      = 0.5   # min aggregate cosine score for a test to appear in potential_tests

# ── Negation detection ────────────────────────────────────────────────────────
_NEGATION_RE = re.compile(
    r"\b(no|not|denies|denying|without|absent|negative\s+for|rules?\s+out|ruled?\s+out)\b",
    re.I,
)


def _is_negated(query_text: str, label: str, window: int = 6) -> bool:
    """Return True if `label` appears in `query_text` preceded by a negation word
    within `window` tokens — indicating the matched entity should be excluded.

    Examples:
        _is_negated("no chest pain", "Chest Pain")          → True
        _is_negated("chest pain", "Chest Pain")             → False
        _is_negated("chest pain, no diaphoresis", "Chest Pain") → False
        _is_negated("denies fever and nausea", "Fever")     → True
    """
    tokens = re.split(r"\s+", query_text.lower())
    label_tokens = label.lower().split()
    n = len(label_tokens)
    for i in range(len(tokens) - n + 1):
        if tokens[i : i + n] == label_tokens:
            window_text = " ".join(tokens[max(0, i - window) : i])
            if _NEGATION_RE.search(window_text):
                return True
    return False

_KG_PKL_FILE = os.environ.get("KG_PKL_FILE", "triage_knowledge_graph_enriched.pkl")
_DEFAULT_PKL = os.path.join(_kg_dir, _KG_PKL_FILE)


# ── Semantic entity linking ───────────────────────────────────────────────────
def get_graph_matches_semantic(
    raw_dx: str,
    top_k: int = _SEMANTIC_TOP_K,
    min_score: float = _SEMANTIC_MIN_SIM,
) -> list[dict]:
    """Query ChromaDB kg_nodes for KG nodes semantically relevant to raw_dx.

    ChromaDB returns cosine distance (0 = identical); converted to similarity
    as score = 1.0 − distance. Results below min_score are discarded.

    Returns a list of dicts, each with:
        label       str    KG node label
        node_type   str    e.g. 'Symptom', 'Condition'
        score       float  cosine similarity in [0, 1]
        confidence  float  score × 100, rounded to 1 dp  (percentage)

    Returns [] on any error (soft failure — caller logs no-match).
    """
    try:
        collection = _get_collection()
        total = collection.count()
        if total == 0:
            return []

        query_vec = embed_query(raw_dx)

        # Oversample slightly then post-filter — ChromaDB has no score_threshold param
        response = collection.query(
            query_embeddings=[query_vec],
            n_results=min(top_k * 2, total),
            include=["metadatas", "distances"],
        )

        results = []
        seen: set[str] = set()

        for meta, dist in zip(
            response["metadatas"][0],
            response["distances"][0],
        ):
            score = 1.0 - dist   # cosine distance → cosine similarity
            if score < min_score:
                continue

            # Skip nodes whose label is negated in the query text
            # (e.g. "no fever" should not match the Fever node)
            if _is_negated(raw_dx, meta["label"]):
                continue

            key = f"{meta['node_type']}:{meta['label']}"
            if key not in seen:
                seen.add(key)
                results.append(
                    {
                        "label":      meta["label"],
                        "node_type":  meta["node_type"],
                        "score":      round(score, 4),
                        "confidence": round(score * 100, 1),
                    }
                )
            if len(results) >= top_k:
                break

        # Sort by descending similarity (ChromaDB already returns nearest-first,
        # but post-filtering may reorder after the distance-to-similarity flip)
        results.sort(key=lambda x: x["score"], reverse=True)
        return results

    except Exception as e:
        print(
            f"  [semantic] ChromaDB query failed ({type(e).__name__}: {e}) — "
            "returning no matches."
        )
        return []


# ── Output formatting ─────────────────────────────────────────────────────────
def _format_matched_nodes(matches: list[dict]) -> str:
    """Human-readable pipe-separated node list with confidence percentages."""
    if not matches:
        return ""
    return " | ".join(
        f"{m['label']} ({m['node_type']}) [{m['confidence']}%]"
        for m in matches
    )


def _format_match_scores(matches: list[dict]) -> str:
    """Pipe-separated 'Label: score' pairs for every node above the threshold.

    Stores the raw cosine similarity (0–1) so downstream code can apply its
    own probability thresholds without re-running the embedding pipeline.
    Example: "Chest Pain: 0.8712 | Diaphoresis: 0.8234"
    """
    if not matches:
        return ""
    return " | ".join(
        f"{m['label']}: {m['score']:.4f}"
        for m in matches
    )


def _aggregate_test_scores(matches: list[dict]) -> dict[str, float]:
    """Compute per-test aggregate cosine similarity across all matched nodes."""
    test_score_sum: dict[str, float] = {}
    for m in matches:
        for t in m.get("tests", []):
            test_score_sum[t] = test_score_sum.get(t, 0.0) + m.get("score", 0.0)
    return test_score_sum


def _format_potential_tests(matches: list[dict], min_score_sum: float = _MIN_SCORE_SUM) -> str:
    """Return tests whose aggregate cosine score meets the threshold, ordered by score.

    A score-weighted approach: a node with similarity 0.95 contributes nearly
    twice as much as one at 0.51, reducing noise from borderline matches.
    `min_score_sum` defaults to 0.5, equivalent to one moderate-confidence match.
    """
    test_score_sum = _aggregate_test_scores(matches)
    tests = sorted(
        [t for t, s in test_score_sum.items() if s >= min_score_sum],
        key=lambda t: -test_score_sum[t],
    )
    return ", ".join(tests) if tests else "No linked tests found"


def _format_test_scores(matches: list[dict]) -> str:
    """Pipe-separated 'Test: aggregate_score' for every recommended test.

    Stores the sum of cosine similarities across all nodes that recommend
    each test, enabling downstream threshold tuning without re-running embeddings.
    Example: "ECG: 1.4946 | Arm X-Ray: 0.6234"
    """
    test_score_sum = _aggregate_test_scores(matches)
    return " | ".join(
        f"{t}: {s:.4f}"
        for t, s in sorted(test_score_sum.items(), key=lambda x: -x[1])
    )


# ── Checkpoint helpers ────────────────────────────────────────────────────────
def _save_partial(
    df: pd.DataFrame,
    mapping_dictionary: dict,
    min_score_sum: float,
    path: str,
) -> None:
    df = df.copy()
    df["matched_graph_nodes"] = df["dx"].map(
        lambda dx: _format_matched_nodes(mapping_dictionary.get(dx, []))
    )
    df["match_scores"] = df["dx"].map(
        lambda dx: _format_match_scores(mapping_dictionary.get(dx, []))
    )
    df["potential_tests"] = df["dx"].map(
        lambda dx: _format_potential_tests(mapping_dictionary.get(dx, []), min_score_sum)
    )
    df["test_scores"] = df["dx"].map(
        lambda dx: _format_test_scores(mapping_dictionary.get(dx, []))
    )

    # Instead of your custom pipe format, dump the data as JSON
    df["matches_json"] = df["dx"].map(lambda dx: json.dumps(mapping_dictionary.get(dx, [])))

    df.to_csv(path, index=False)


# ── Main pipeline ─────────────────────────────────────────────────────────────
def map_diagnoses_semantic(
    csv_input_path: str,
    graph_pkl_path: str,
    csv_output_path: str,
    row_limit: int | None = None,
    resume: bool = False,
    min_score_sum: float = _MIN_SCORE_SUM,
    semantic_top_k: int = _SEMANTIC_TOP_K,
    semantic_threshold: float = _SEMANTIC_MIN_SIM,
    save_every: int = 10,
):
    # Load KG for graph traversal (get_tests_for_node)
    print("Loading Knowledge Graph...")
    with open(graph_pkl_path, "rb") as f:
        kg = pickle.load(f)
    print(f"Graph loaded: {kg.number_of_nodes()} nodes, {kg.number_of_edges()} edges.")

    # Load input CSV
    df = pd.read_csv(csv_input_path)
    print(f"Columns: {list(df.columns)}")
    if "dx" not in df.columns:
        raise ValueError("Input CSV must contain a 'dx' column.")

    if row_limit is not None:
        print(f"  [--limit] Restricting to first {row_limit} rows.")
        df = df.head(row_limit)

    # Resume: reload already-mapped diagnoses from the existing output CSV
    already_mapped: set[str] = set()
    mapping_dictionary: dict[str, list[dict]] = {}

    if resume and os.path.exists(csv_output_path):
        existing_df = pd.read_csv(csv_output_path)
        required = {"matched_graph_nodes", "match_scores", "potential_tests", "test_scores"}
        if required.issubset(existing_df.columns):
            for _, row in existing_df.dropna(subset=["matched_graph_nodes"]).iterrows():
                dx_val = row["dx"]
                if pd.isna(dx_val) or str(row.get("matched_graph_nodes", "")).strip() == "":
                    continue

                # Reconstruct match dicts from the stored score string
                scores_str  = str(row.get("match_scores", "")) if not pd.isna(row.get("match_scores")) else ""
                nodes_str   = str(row["matched_graph_nodes"])
                tests_str   = str(row["potential_tests"]) if not pd.isna(row.get("potential_tests")) else ""
                tests_list  = [
                    t.strip()
                    for t in tests_str.split(",")
                    if t.strip() and t.strip() != "No linked tests found"
                ]

                # Build a score lookup from match_scores column: "Label: 0.XXXX"
                score_lookup: dict[str, float] = {}
                for part in scores_str.split(" | "):
                    part = part.strip()
                    if ": " in part:
                        lbl, _, val = part.rpartition(": ")
                        try:
                            score_lookup[lbl.strip()] = float(val)
                        except ValueError:
                            pass

                enriched_resume = []
                for part in nodes_str.split(" | "):
                    part = part.strip()
                    if part.endswith("]") and "[" in part:
                        part = part[: part.rfind("[")].strip()
                    if "(" in part and part.endswith(")"):
                        lbl   = part[: part.rfind("(")].strip()
                        ntype = part[part.rfind("(") + 1 : -1].strip()
                        sc    = score_lookup.get(lbl, 0.0)
                        enriched_resume.append(
                            {
                                "label":      lbl,
                                "node_type":  ntype,
                                "score":      sc,
                                "confidence": round(sc * 100, 1),
                                "tests":      tests_list,
                            }
                        )
                mapping_dictionary[dx_val] = enriched_resume
                already_mapped.add(dx_val)

            print(
                f"  [resume] Loaded {len(already_mapped)} already-mapped diagnoses "
                f"from '{csv_output_path}'."
            )
        else:
            missing = required - set(existing_df.columns)
            print(f"  [resume] Output CSV missing columns {missing} — starting fresh.")

    if not resume and os.path.exists(csv_output_path):
        print(
            f"  [warning] '{csv_output_path}' already exists but --resume was not set.\n"
            f"            Proceeding will OVERWRITE existing results.\n"
            f"            Re-run with --resume to continue from the last checkpoint."
        )

    unique_raw_diagnoses = df["dx"].dropna().unique()
    remaining = [d for d in unique_raw_diagnoses if d not in already_mapped]
    print(
        f"Found {len(unique_raw_diagnoses)} unique diagnoses "
        f"({len(already_mapped)} already mapped, {len(remaining)} to process)."
    )

    print(
        f"Starting semantic matching "
        f"(top_k={semantic_top_k}, threshold={semantic_threshold}, "
        f"min_score_sum={min_score_sum}, save_every={save_every}) ..."
    )

    semantic_matched   = 0
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
            # Split on ";" to handle multi-term dx fields
            terms = [t.strip() for t in str(raw_diag).split(";") if t.strip()]

            seen_keys: set[str] = set()
            enriched:  list[dict] = []

            for term in terms:
                term_matches = get_graph_matches_semantic(
                    term,
                    top_k=semantic_top_k,
                    min_score=semantic_threshold,
                )
                if term_matches:
                    semantic_matched += 1
                else:
                    semantic_unmatched += 1

                for m in term_matches:
                    key = f"{m['node_type']}:{m['label']}"
                    if key not in seen_keys:
                        seen_keys.add(key)
                        tests = get_tests_for_node(kg, m["label"], m["node_type"])
                        enriched.append({**m, "tests": tests})

                label_str = (
                    " | ".join(
                        f"{m['label']} ({m['node_type']}) [{m['confidence']}%]"
                        for m in term_matches
                    )
                    or "no match"
                )
                tqdm.write(f"  [semantic] '{term}' -> {label_str}")

            mapping_dictionary[raw_diag] = enriched

            total_so_far = semantic_matched + semantic_unmatched
            match_pct = semantic_matched / total_so_far * 100 if total_so_far else 0
            progress.set_postfix(matched=f"{match_pct:.1f}%", refresh=False)

            if save_every > 0 and i % save_every == 0:
                _save_partial(df, mapping_dictionary, min_score_sum, csv_output_path)
                tqdm.write(
                    f"  [checkpoint] Saved {i}/{len(remaining)} diagnoses → '{csv_output_path}'"
                )

    finally:
        if mapping_dictionary:
            _save_partial(df, mapping_dictionary, min_score_sum, csv_output_path)
            tqdm.write(
                f"  [checkpoint] Final flush: {len(mapping_dictionary)} diagnoses → '{csv_output_path}'"
            )

    # Summary
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

    # Final DataFrame columns
    df["matched_graph_nodes"] = df["dx"].map(
        lambda dx: _format_matched_nodes(mapping_dictionary.get(dx, []))
    )
    df["match_scores"] = df["dx"].map(
        lambda dx: _format_match_scores(mapping_dictionary.get(dx, []))
    )
    df["potential_tests"] = df["dx"].map(
        lambda dx: _format_potential_tests(mapping_dictionary.get(dx, []), min_score_sum)
    )
    df["test_scores"] = df["dx"].map(
        lambda dx: _format_test_scores(mapping_dictionary.get(dx, []))
    )

    # ADD THIS LINE HERE:
    df["matches_json"] = df["dx"].map(
        lambda dx: json.dumps(mapping_dictionary.get(dx, []))
    )
    
    df.to_csv(csv_output_path, index=False)
    print(f"\nSaved enriched data to '{csv_output_path}' (min_score_sum={min_score_sum})")

    # Validation (only when ground-truth columns are present)
    missing_gt = [col for col in TEST_COLUMN_MAP.values() if col not in df.columns]
    if missing_gt:
        print(f"\n[!] Validation skipped: ground-truth column(s) not found: {missing_gt}")
    else:
        metrics_df = _run_validation(df)
        validation_path = csv_output_path.replace(".csv", "_validation.csv")
        metrics_df.to_csv(validation_path, index=False)
        print(f"\nValidation report saved to '{validation_path}'")
        df.to_csv(csv_output_path, index=False)

    return df


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Semantically map patient diagnoses (dx column) to KG nodes "
            "using local ChromaDB + MedEmbed embeddings."
        )
    )
    parser.add_argument(
        "--input-csv", required=True,
        help="Path to the input CSV file containing a 'dx' column.",
    )
    parser.add_argument(
        "--output-csv", default="patient_diagnoses_chroma.csv",
        help="Path for the enriched output CSV (default: patient_diagnoses_chroma.csv).",
    )
    parser.add_argument(
        "--graph-pkl", default=_DEFAULT_PKL,
        help=f"Path to the KG pickle file (default: {_KG_PKL_FILE} from KG_PKL_FILE env).",
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
        "--min-score-sum", type=float, default=_MIN_SCORE_SUM, metavar="F",
        help=(
            "Minimum aggregate cosine similarity score (sum across all matched nodes) "
            "for a test to appear in potential_tests. Default: 0.5 "
            "(≈ one moderate-confidence match). Lower = more tests surfaced."
        ),
    )
    parser.add_argument(
        "--semantic-top-k", type=int, default=_SEMANTIC_TOP_K, metavar="K",
        help=f"Max KG nodes to retrieve per query (default: {_SEMANTIC_TOP_K}).",
    )
    parser.add_argument(
        "--semantic-threshold", type=float, default=_SEMANTIC_MIN_SIM, metavar="F",
        help=f"Min cosine similarity 0–1 to accept a match (default: {_SEMANTIC_MIN_SIM}).",
    )
    parser.add_argument(
        "--save-every", type=int, default=10, metavar="N",
        help="Checkpoint to disk every N unique diagnoses. Set 0 to disable. Default: 10.",
    )
    args = parser.parse_args()

    result_df = map_diagnoses_semantic(
        csv_input_path=args.input_csv,
        graph_pkl_path=args.graph_pkl,
        csv_output_path=args.output_csv,
        row_limit=args.limit,
        resume=args.resume,
        min_score_sum=args.min_score_sum,
        semantic_top_k=args.semantic_top_k,
        semantic_threshold=args.semantic_threshold,
        save_every=args.save_every,
    )

    print("\n--- Output Preview ---")
    print(result_df[["dx", "matched_graph_nodes", "match_scores", "potential_tests", "test_scores"]].head())
