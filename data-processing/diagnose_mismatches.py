#!/usr/bin/env python3
"""
diagnose_mismatches.py — Investigate false positives and false negatives
from a semantic_csv_mapping run.

For each FP/FN case, re-queries Qdrant at a sweep of similarity thresholds
to show why a dx matched (or failed to match) and what KG nodes are nearby.

Usage:
    python diagnose_mismatches.py \\
        --enriched-csv  patient_diagnoses_semantic.csv \\
        --graph-pkl     ../knowledge-graphs/triage_knowledge_graph_enriched.pkl

    # Focus on specific tests only
    python diagnose_mismatches.py \\
        --enriched-csv  patient_diagnoses_semantic.csv \\
        --tests         xray_arm us_testes
"""

import argparse
import os
import pickle
import sys
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

_current_dir = os.path.dirname(__file__)
_env_path = os.path.abspath(os.path.join(_current_dir, "..", "knowledge-graphs", ".env"))
load_dotenv(_env_path, override=True)

sys.path.insert(0, _current_dir)
from csv_mapping import get_tests_for_node, _PREFIX_TO_TYPE

# ── Constants ──────────────────────────────────────────────────────────────────
KG_COLLECTION_NAME = "kg_nodes"
NEBIUS_EMBEDDING_MODEL = os.environ.get(
    "NEBIUS_EMBEDDING_MODEL", "intfloat/e5-mistral-7b-instruct"
)
_SEMANTIC_QUERY_INSTR = (
    "Instruct: Retrieve relevant medical knowledge graph nodes for the following "
    "clinical diagnosis.\nQuery: "
)

# Maps short test name → (KG test label, gt column, kg prediction column)
_TEST_META = {
    "ecg":       ("ECG",                   "ecg_dx",       "ecg_kg"),
    "xray_arm":  ("Arm X-Ray",             "xray_arm_dx",  "xray_arm_kg"),
    "us_app":    ("Appendix Ultrasound",   "us_app_dx",    "us_app_kg"),
    "us_testes": ("Testicular Ultrasound", "us_testes_dx", "us_testes_kg"),
}

# Thresholds to probe during the sweep
_THRESHOLD_SWEEP = [0.50, 0.55, 0.60, 0.65, 0.68, 0.70, 0.73, 0.75, 0.80]

# ── Lazy clients (same pattern as semantic_csv_mapping.py) ────────────────────
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


# ── Embedding / retrieval ──────────────────────────────────────────────────────
def _embed_query(text: str) -> list[float]:
    client = _get_embed_client()
    resp = client.embeddings.create(
        model=NEBIUS_EMBEDDING_MODEL,
        input=[_SEMANTIC_QUERY_INSTR + text],
    )
    return resp.data[0].embedding


def _retrieve_all(dx: str, top_k: int = 20) -> list[dict]:
    """Return the top_k nearest KG nodes for dx with NO score threshold,
    so we can see the full similarity neighbourhood."""
    vec = _embed_query(dx)
    qdrant = _get_qdrant()
    hits = qdrant.query_points(
        collection_name=KG_COLLECTION_NAME,
        query=vec,
        limit=top_k,
        with_payload=True,
    ).points
    return [
        {
            "score":     hit.score,
            "node_id":   hit.payload["node_id"],
            "label":     hit.payload["label"],
            "node_type": hit.payload["node_type"],
        }
        for hit in hits
    ]


# ── KG traversal ──────────────────────────────────────────────────────────────
def _tests_for_hits(hits: list[dict], kg) -> set[str]:
    """Return the set of test names reachable from the given hits."""
    tests: set[str] = set()
    for h in hits:
        for t in get_tests_for_node(kg, h["label"], h["node_type"]):
            tests.add(t)
    return tests


# ── Threshold sweep ────────────────────────────────────────────────────────────
def _threshold_sweep(hits: list[dict], kg, target_test: str) -> None:
    """Print a table showing which thresholds would / wouldn't trigger target_test."""
    print(f"\n  Threshold sweep for '{target_test}':")
    print(f"  {'Threshold':>10}  {'Nodes matched':>13}  {'Triggers test?':>14}  Matched nodes")
    print(f"  {'-'*10}  {'-'*13}  {'-'*14}  {'-'*40}")
    for thr in _THRESHOLD_SWEEP:
        filtered = [h for h in hits if h["score"] >= thr]
        tests = _tests_for_hits(filtered, kg)
        triggered = "YES ✓" if target_test in tests else "no"
        node_summary = ", ".join(
            f"{h['label']} [{h['score']:.2f}]" for h in filtered
        ) or "—"
        print(f"  {thr:>10.2f}  {len(filtered):>13}  {triggered:>14}  {node_summary[:80]}")


# ── Case report ───────────────────────────────────────────────────────────────
def _report_case(dx: str, case_type: str, test_label: str, kg, row: pd.Series) -> None:
    """Print a full diagnostic report for one FP or FN case."""
    print(f"\n{'═'*70}")
    print(f"  {case_type}  |  Test: {test_label}")
    print(f"  dx         : {dx}")
    if "matched_graph_nodes" in row.index:
        print(f"  KG match   : {row.get('matched_graph_nodes', '—')}")
    if "potential_tests" in row.index:
        print(f"  Predicted  : {row.get('potential_tests', '—')}")
    print(f"{'─'*70}")

    print(f"\n  Top-20 nearest KG nodes (no threshold):")
    hits = _retrieve_all(dx, top_k=20)
    for i, h in enumerate(hits, 1):
        print(f"    [{i:>2}] {h['score']:.4f}  {h['node_type']:<25}  {h['label']}")

    _threshold_sweep(hits, kg, test_label)
    print()


# ── Main ───────────────────────────────────────────────────────────────────────
def diagnose(
    enriched_csv: str,
    graph_pkl: str,
    tests_to_check: list[str],
) -> None:
    # Load enriched CSV
    df = pd.read_csv(enriched_csv)

    # Load KG
    print(f"Loading KG from {graph_pkl} ...")
    with open(graph_pkl, "rb") as f:
        kg = pickle.load(f)
    print(f"Graph loaded: {kg.number_of_nodes()} nodes.\n")

    for test_key in tests_to_check:
        if test_key not in _TEST_META:
            print(f"[!] Unknown test key '{test_key}'. Valid: {list(_TEST_META)}")
            continue

        test_label, gt_col, kg_col = _TEST_META[test_key]

        # Check required columns exist
        missing = [c for c in [gt_col, "potential_tests"] if c not in df.columns]
        if missing:
            print(f"[!] Skipping {test_key}: columns not found: {missing}")
            continue

        # Derive prediction column from potential_tests if *_kg not present
        if kg_col in df.columns:
            pred = df[kg_col].astype(int)
        else:
            pred = df["potential_tests"].apply(
                lambda v: int(test_label in str(v)) if pd.notna(v) else 0
            )

        gt = df[gt_col].astype(int)

        fp_rows = df[(pred == 1) & (gt == 0)]
        fn_rows = df[(pred == 0) & (gt == 1)]

        print(f"\n{'━'*70}")
        print(f"  TEST: {test_label}   |   FP={len(fp_rows)}   FN={len(fn_rows)}")
        print(f"{'━'*70}")

        if fp_rows.empty and fn_rows.empty:
            print("  No FP or FN cases found for this test.\n")
            continue

        # ── False Positives ───────────────────────────────────────────────────
        seen_dx: set[str] = set()
        for _, row in fp_rows.iterrows():
            dx = str(row["dx"])
            if dx in seen_dx:
                continue
            seen_dx.add(dx)
            _report_case(dx, "FALSE POSITIVE", test_label, kg, row)

        # ── False Negatives ───────────────────────────────────────────────────
        seen_dx = set()
        for _, row in fn_rows.iterrows():
            dx = str(row["dx"])
            if dx in seen_dx:
                continue
            seen_dx.add(dx)
            _report_case(dx, "FALSE NEGATIVE", test_label, kg, row)


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    _KG_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "knowledge-graphs"))
    _KG_PKL_FILE = os.environ.get("KG_PKL_FILE", "triage_knowledge_graph_enriched.pkl")
    _DEFAULT_PKL = os.path.join(_KG_DIR, _KG_PKL_FILE)

    parser = argparse.ArgumentParser(
        description="Investigate FP/FN cases from a semantic_csv_mapping run."
    )
    parser.add_argument(
        "--enriched-csv", required=True, metavar="PATH",
        help="Output CSV from semantic_csv_mapping.py (must contain potential_tests column).",
    )
    parser.add_argument(
        "--graph-pkl", default=_DEFAULT_PKL, metavar="PATH",
        help=f"Knowledge graph PKL (default: {_KG_PKL_FILE}).",
    )
    parser.add_argument(
        "--tests", nargs="+",
        default=list(_TEST_META.keys()),
        metavar="TEST",
        help=(
            f"Which tests to investigate. Choices: {list(_TEST_META)}. "
            "Default: all four."
        ),
    )
    args = parser.parse_args()

    diagnose(
        enriched_csv=args.enriched_csv,
        graph_pkl=args.graph_pkl,
        tests_to_check=args.tests,
    )
