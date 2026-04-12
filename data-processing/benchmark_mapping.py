#!/usr/bin/env python3
"""
benchmark_mapping.py — Compare timing of substring vs semantic KG mapping.

Randomly samples --sample rows from an input CSV (dx column), then runs
three measurement phases per diagnosis:

  Phase 1 — Match-only hot-path (--iterations reps each):
      Substring : get_graph_matches(dx, kg)
      Semantic  : get_graph_matches_semantic(dx, top_k, min_score)

  Phase 2 — Full single-row pipeline (--iterations // 2 reps, or --no-full-row to skip):
      Substring : match → get_tests_for_node → format output strings
      Semantic  : same, plus score aggregation / formatting

  Phase 3 — Cold-start overhead (measured once before Phase 1):
      Substring : time to load KG from PKL
      Semantic  : time to initialise ChromaDB + embed the first query

Usage:
    python benchmark_mapping.py --input-csv <path> [options]

Options:
    --input-csv     PATH   Input CSV with a 'dx' column (required)
    --sample        N      Rows to randomly select (default: 10)
    --seed          S      Random seed for reproducibility (default: 42)
    --iterations    N      Timing reps per diagnosis in Phase 1 (default: 50)
                           Phase 2 uses N // 2 reps
    --no-full-row          Skip Phase 2
    --output-csv    PATH   Raw-timing CSV export (default: benchmark_results_<ts>.csv)
    --top-k         K      Semantic ChromaDB top_k (default: 5)
    --min-score     F      Semantic cosine similarity threshold (default: 0.50)
    --graph-pkl     PATH   Override KG pickle path (default: from KG_PKL_FILE env)
"""

import argparse
import gc
import json
import os
import pickle
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

# ── Path / env setup ──────────────────────────────────────────────────────────
_script_dir = Path(__file__).resolve().parent
_kg_dir     = _script_dir.parent / "knowledge-graphs"
_vector_dir = _kg_dir / "vector_db"

load_dotenv(_kg_dir / ".env", override=True)

sys.path.insert(0, str(_script_dir))
sys.path.insert(0, str(_vector_dir))

# ── Lazy imports (so cold-start timing can be measured below) ─────────────────
# Substring helpers — imported now (no heavy model loading)
from kg_substring_csv_mapping import (
    get_graph_matches,
    get_tests_for_node,
)

# Semantic helpers — importing the module is cheap; the ChromaDB client and
# the embedding model are only initialised on the *first* call to those fns.
from graphrag_semantic_csv_mapping_chroma import (
    get_graph_matches_semantic,
    _format_matched_nodes,
    _format_match_scores,
    _format_potential_tests,
    _format_test_scores,
    _SEMANTIC_TOP_K,
    _SEMANTIC_MIN_SIM,
    _MIN_SCORE_SUM,
)

# ── Constants / defaults ──────────────────────────────────────────────────────
_DEFAULT_SAMPLE     = 10
_DEFAULT_SEED       = 42
_DEFAULT_ITERATIONS = 50
_DEFAULT_TOP_K      = _SEMANTIC_TOP_K
_DEFAULT_MIN_SCORE  = _SEMANTIC_MIN_SIM

_KG_PKL_FILE = os.environ.get("KG_PKL_FILE", "triage_knowledge_graph_enriched.pkl")
_DEFAULT_PKL = str(_kg_dir / _KG_PKL_FILE)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _hr(label: str, width: int = 72) -> None:
    print(f"\n{'─' * width}")
    print(f"  {label}")
    print('─' * width)


def _mean(vals):
    return sum(vals) / len(vals) if vals else float("nan")


def _std(vals):
    if len(vals) < 2:
        return 0.0
    m = _mean(vals)
    return (sum((v - m) ** 2 for v in vals) / (len(vals) - 1)) ** 0.5


def _ms(seconds: float) -> str:
    """Format seconds as milliseconds string."""
    return f"{seconds * 1000:.3f} ms"


def _us(seconds: float) -> str:
    """Format seconds as microseconds string."""
    return f"{seconds * 1_000_000:.1f} µs"


def _fmt_matches(matches: list[dict]) -> str:
    if not matches:
        return "(no match)"
    parts = []
    for m in matches:
        score_str = f" [{m['confidence']}%]" if "confidence" in m else ""
        parts.append(f"{m['label']} ({m['node_type']}){score_str}")
    return " | ".join(parts)


# ─────────────────────────────────────────────────────────────────────────────
# Phase 3 — Cold-start measurement
# ─────────────────────────────────────────────────────────────────────────────

def measure_cold_start(graph_pkl_path: str, warm_dx: str, top_k: int, min_score: float):
    """
    Measure one-time init costs separately from per-query latency.

    Returns dict with:
        kg_load_s       float   seconds to load the KG pickle
        semantic_init_s float   seconds for first ChromaDB query (model load included)
    """
    _hr("PHASE 3 — Cold-start overhead (measured once)")

    # Substring cold start: PKL load
    print("  [substring] Loading KG from PKL …", end=" ", flush=True)
    t0 = time.perf_counter()
    with open(graph_pkl_path, "rb") as f:
        kg = pickle.load(f)
    kg_load_s = time.perf_counter() - t0
    print(f"{_ms(kg_load_s)}")
    print(f"    → {kg.number_of_nodes()} nodes, {kg.number_of_edges()} edges")

    # Semantic cold start: first embedding + ChromaDB query
    print("  [semantic]  First query (model + ChromaDB init) …", end=" ", flush=True)
    t0 = time.perf_counter()
    _ = get_graph_matches_semantic(warm_dx, top_k=top_k, min_score=min_score)
    semantic_init_s = time.perf_counter() - t0
    print(f"{_ms(semantic_init_s)}")

    return {"kg": kg, "kg_load_s": kg_load_s, "semantic_init_s": semantic_init_s}


# ─────────────────────────────────────────────────────────────────────────────
# Phase 1 — Match-only timing
# ─────────────────────────────────────────────────────────────────────────────

def time_match_only(
    dx: str,
    kg,
    n_iter: int,
    top_k: int,
    min_score: float,
) -> dict:
    """
    Return per-diagnosis timing stats for the matching hot-path only.

    dict keys:
        sub_times_s     list[float]
        sem_times_s     list[float]
        sub_matches     last match result (list[dict])
        sem_matches     last match result (list[dict])
    """
    sub_times: list[float] = []
    sem_times: list[float] = []
    sub_matches: list[dict] = []
    sem_matches: list[dict] = []

    # Substring iterations
    for _ in range(n_iter):
        gc.collect()
        t0 = time.perf_counter()
        result = get_graph_matches(dx, kg)
        sub_times.append(time.perf_counter() - t0)
    sub_matches = result  # last result

    # Semantic iterations (model already warm after cold-start phase)
    for _ in range(n_iter):
        gc.collect()
        t0 = time.perf_counter()
        result = get_graph_matches_semantic(dx, top_k=top_k, min_score=min_score)
        sem_times.append(time.perf_counter() - t0)
    sem_matches = result

    return {
        "sub_times_s": sub_times,
        "sem_times_s": sem_times,
        "sub_matches": sub_matches,
        "sem_matches": sem_matches,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Phase 2 — Full single-row pipeline
# ─────────────────────────────────────────────────────────────────────────────

def _run_substring_row(dx: str, kg, min_nodes: int = 1) -> dict:
    """Replicate substring pipeline logic for one dx string (no CSV I/O)."""
    terms = [t.strip() for t in dx.split(";") if t.strip()]
    seen_labels: set[str] = set()
    enriched: list[dict] = []

    for term in terms:
        for m in get_graph_matches(term, kg):
            key = f"{m['node_type']}:{m['label']}"
            if key not in seen_labels:
                seen_labels.add(key)
                tests = get_tests_for_node(kg, m["label"], m["node_type"])
                enriched.append({**m, "tests": tests})

    # Format output strings (mirrors map_diagnoses_with_llm)
    matched_nodes_str = " | ".join(
        f"{m['label']} ({m['node_type']})" for m in enriched
    ) or ""

    test_vote: dict[str, int] = {}
    for m in enriched:
        for t in m.get("tests", []):
            test_vote[t] = test_vote.get(t, 0) + 1
    tests = [t for t, c in test_vote.items() if c >= min_nodes]
    potential_tests_str = ", ".join(tests) or "No linked tests found"

    return {
        "matched_graph_nodes": matched_nodes_str,
        "potential_tests":     potential_tests_str,
        "enriched":            enriched,
    }


def _run_semantic_row(
    dx: str,
    kg,
    top_k: int = _DEFAULT_TOP_K,
    min_score: float = _DEFAULT_MIN_SCORE,
    min_score_sum: float = _MIN_SCORE_SUM,
) -> dict:
    """Replicate semantic pipeline logic for one dx string (no CSV I/O)."""
    terms = [t.strip() for t in dx.split(";") if t.strip()]
    seen_keys: set[str] = set()
    enriched: list[dict] = []

    for term in terms:
        for m in get_graph_matches_semantic(term, top_k=top_k, min_score=min_score):
            key = f"{m['node_type']}:{m['label']}"
            if key not in seen_keys:
                seen_keys.add(key)
                tests = get_tests_for_node(kg, m["label"], m["node_type"])
                enriched.append({**m, "tests": tests})

    matched_nodes_str  = _format_matched_nodes(enriched)
    match_scores_str   = _format_match_scores(enriched)
    potential_tests_str = _format_potential_tests(enriched, min_score_sum)
    test_scores_str    = _format_test_scores(enriched)

    return {
        "matched_graph_nodes": matched_nodes_str,
        "match_scores":        match_scores_str,
        "potential_tests":     potential_tests_str,
        "test_scores":         test_scores_str,
        "enriched":            enriched,
    }


def time_full_row(
    dx: str,
    kg,
    n_iter: int,
    top_k: int,
    min_score: float,
) -> dict:
    """Return per-diagnosis timing stats for the full single-row pipeline."""
    sub_times: list[float] = []
    sem_times: list[float] = []
    sub_result: dict = {}
    sem_result: dict = {}

    for _ in range(n_iter):
        gc.collect()
        t0 = time.perf_counter()
        sub_result = _run_substring_row(dx, kg)
        sub_times.append(time.perf_counter() - t0)

    for _ in range(n_iter):
        gc.collect()
        t0 = time.perf_counter()
        sem_result = _run_semantic_row(dx, kg, top_k=top_k, min_score=min_score)
        sem_times.append(time.perf_counter() - t0)

    return {
        "sub_times_s": sub_times,
        "sem_times_s": sem_times,
        "sub_result":  sub_result,
        "sem_result":  sem_result,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Output / reporting
# ─────────────────────────────────────────────────────────────────────────────

def _stats_row(times: list[float]) -> dict:
    return {
        "mean_ms":  _mean(times) * 1000,
        "std_ms":   _std(times)  * 1000,
        "min_ms":   min(times)   * 1000,
        "max_ms":   max(times)   * 1000,
        "n":        len(times),
    }


def print_phase1_table(results: list[dict]) -> None:
    _hr("PHASE 1 — Match-only timing summary")
    col_w = 35
    hdr = (
        f"{'Diagnosis':<{col_w}} {'Method':<10}"
        f" {'Mean':>9} {'Std':>9} {'Min':>9} {'Max':>9}"
        f"  {'Speedup':>8}"
    )
    sep = "─" * len(hdr)
    print(hdr)
    print(sep)
    for row in results:
        dx_short = row["dx"][:col_w - 1]
        sub = row["phase1_sub"]
        sem = row["phase1_sem"]
        speedup = sem["mean_ms"] / sub["mean_ms"] if sub["mean_ms"] > 0 else float("nan")
        print(
            f"{dx_short:<{col_w}} {'substring':<10}"
            f" {sub['mean_ms']:>8.3f}ms {sub['std_ms']:>8.3f}ms"
            f" {sub['min_ms']:>8.3f}ms {sub['max_ms']:>8.3f}ms"
        )
        print(
            f"{'':<{col_w}} {'semantic':<10}"
            f" {sem['mean_ms']:>8.3f}ms {sem['std_ms']:>8.3f}ms"
            f" {sem['min_ms']:>8.3f}ms {sem['max_ms']:>8.3f}ms"
            f"  {speedup:>6.1f}×"
        )
        print(sep)


def print_phase2_table(results: list[dict]) -> None:
    _hr("PHASE 2 — Full single-row pipeline timing summary")
    col_w = 35
    hdr = (
        f"{'Diagnosis':<{col_w}} {'Method':<10}"
        f" {'Mean':>9} {'Std':>9} {'Min':>9} {'Max':>9}"
        f"  {'Speedup':>8}"
    )
    sep = "─" * len(hdr)
    print(hdr)
    print(sep)
    for row in results:
        if row.get("phase2_sub") is None:
            continue
        dx_short = row["dx"][:col_w - 1]
        sub = row["phase2_sub"]
        sem = row["phase2_sem"]
        speedup = sem["mean_ms"] / sub["mean_ms"] if sub["mean_ms"] > 0 else float("nan")
        print(
            f"{dx_short:<{col_w}} {'substring':<10}"
            f" {sub['mean_ms']:>8.3f}ms {sub['std_ms']:>8.3f}ms"
            f" {sub['min_ms']:>8.3f}ms {sub['max_ms']:>8.3f}ms"
        )
        print(
            f"{'':<{col_w}} {'semantic':<10}"
            f" {sem['mean_ms']:>8.3f}ms {sem['std_ms']:>8.3f}ms"
            f" {sem['min_ms']:>8.3f}ms {sem['max_ms']:>8.3f}ms"
            f"  {speedup:>6.1f}×"
        )
        print(sep)


def print_output_comparison(results: list[dict]) -> None:
    _hr("OUTPUT COMPARISON — matched nodes & recommended tests")
    for row in results:
        dx = row["dx"]
        print(f"\n  dx: {dx!r}")

        # Phase 1 matches
        sub_m = _fmt_matches(row["sub_matches"])
        sem_m = _fmt_matches(row["sem_matches"])
        print(f"    [substring] nodes  : {sub_m}")
        print(f"    [semantic]  nodes  : {sem_m}")

        # Phase 2 tests (if available)
        if row.get("sub_result"):
            sub_t = row["sub_result"].get("potential_tests", "—")
            sem_t = row["sem_result"].get("potential_tests", "—")
            sub_ts = row["sem_result"].get("test_scores", "")
            print(f"    [substring] tests  : {sub_t}")
            print(f"    [semantic]  tests  : {sem_t}")
            if sub_ts:
                print(f"    [semantic]  scores : {sub_ts}")

        match_diff = set(_fmt_matches(row["sub_matches"]).split(" | ")) != set(
            _fmt_matches(row["sem_matches"]).split(" | ")
        )
        if match_diff:
            print(f"    ⚑ Methods returned DIFFERENT matched nodes for this dx")


def print_aggregate_summary(results: list[dict], cold: dict) -> None:
    _hr("AGGREGATE SUMMARY")
    all_sub_p1 = [t for r in results for t in r["phase1_sub_raw"]]
    all_sem_p1 = [t for r in results for t in r["phase1_sem_raw"]]
    mean_sub   = _mean(all_sub_p1) * 1000
    mean_sem   = _mean(all_sem_p1) * 1000
    overall_speedup = mean_sem / mean_sub if mean_sub > 0 else float("nan")

    print(f"  Substring mean (Phase 1) : {mean_sub:.3f} ms")
    print(f"  Semantic  mean (Phase 1) : {mean_sem:.3f} ms")
    print(f"  Semantic / Substring     : {overall_speedup:.1f}× slower on average\n")

    print(f"  Cold-start overhead:")
    print(f"    KG pickle load         : {_ms(cold['kg_load_s'])}")
    print(f"    Semantic model init    : {_ms(cold['semantic_init_s'])}")

    if any(r.get("phase2_sub") for r in results):
        all_sub_p2 = [t for r in results if r.get("phase2_sub_raw") for t in r["phase2_sub_raw"]]
        all_sem_p2 = [t for r in results if r.get("phase2_sem_raw") for t in r["phase2_sem_raw"]]
        if all_sub_p2 and all_sem_p2:
            mean_sub_p2 = _mean(all_sub_p2) * 1000
            mean_sem_p2 = _mean(all_sem_p2) * 1000
            speedup_p2  = mean_sem_p2 / mean_sub_p2 if mean_sub_p2 > 0 else float("nan")
            print(f"\n  Full-row (Phase 2):")
            print(f"    Substring mean         : {mean_sub_p2:.3f} ms")
            print(f"    Semantic  mean         : {mean_sem_p2:.3f} ms")
            print(f"    Semantic / Substring   : {speedup_p2:.1f}× slower on average")


def export_csv(results: list[dict], cold: dict, output_path: str) -> None:
    rows = []
    for r in results:
        for phase, sub_key, sem_key in [
            ("phase1", "phase1_sub_raw", "phase1_sem_raw"),
            ("phase2", "phase2_sub_raw", "phase2_sem_raw"),
        ]:
            if not r.get(sub_key):
                continue
            for i, (s, se) in enumerate(zip(r[sub_key], r[sem_key])):
                rows.append({
                    "dx":            r["dx"],
                    "phase":         phase,
                    "iteration":     i + 1,
                    "substring_s":   s,
                    "semantic_s":    se,
                    "speedup":       se / s if s > 0 else None,
                })

    df = pd.DataFrame(rows)
    # Prepend cold-start row
    cold_row = pd.DataFrame([{
        "dx":           "__cold_start__",
        "phase":        "cold",
        "iteration":    1,
        "substring_s":  cold["kg_load_s"],
        "semantic_s":   cold["semantic_init_s"],
        "speedup":      cold["semantic_init_s"] / cold["kg_load_s"],
    }])
    df = pd.concat([cold_row, df], ignore_index=True)
    df.to_csv(output_path, index=False)
    print(f"\n  Raw timings exported to: {output_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark substring vs semantic KG diagnosis mapping."
    )
    parser.add_argument("--input-csv",  required=True,
                        help="CSV with a 'dx' column.")
    parser.add_argument("--sample",     type=int, default=_DEFAULT_SAMPLE, metavar="N",
                        help=f"Rows to randomly sample (default: {_DEFAULT_SAMPLE}).")
    parser.add_argument("--seed",       type=int, default=_DEFAULT_SEED, metavar="S",
                        help=f"Random seed (default: {_DEFAULT_SEED}).")
    parser.add_argument("--iterations", type=int, default=_DEFAULT_ITERATIONS, metavar="N",
                        help=f"Timing reps per diagnosis in Phase 1 (default: {_DEFAULT_ITERATIONS}). "
                             "Phase 2 uses N // 2.")
    parser.add_argument("--no-full-row", action="store_true",
                        help="Skip Phase 2 (full single-row pipeline timing).")
    parser.add_argument("--output-csv",
                        default=f"benchmark_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        help="Path for raw-timing CSV export.")
    parser.add_argument("--top-k",    type=int,   default=_DEFAULT_TOP_K,    metavar="K",
                        help=f"Semantic ChromaDB top_k (default: {_DEFAULT_TOP_K}).")
    parser.add_argument("--min-score", type=float, default=_DEFAULT_MIN_SCORE, metavar="F",
                        help=f"Semantic cosine threshold (default: {_DEFAULT_MIN_SCORE}).")
    parser.add_argument("--graph-pkl", default=_DEFAULT_PKL,
                        help=f"KG pickle path (default: {_KG_PKL_FILE} from KG_PKL_FILE env).")
    args = parser.parse_args()

    # ── Load input CSV + sample ───────────────────────────────────────────────
    _hr("Setup")
    print(f"  Input CSV    : {args.input_csv}")
    input_df = pd.read_csv(args.input_csv)
    if "dx" not in input_df.columns:
        sys.exit("  ERROR: input CSV must contain a 'dx' column.")

    input_df = input_df.dropna(subset=["dx"])
    n_sample = min(args.sample, len(input_df))
    sample_df = input_df.sample(n=n_sample, random_state=args.seed)
    diagnoses = sample_df["dx"].tolist()

    print(f"  Total rows   : {len(input_df)}")
    print(f"  Sampled      : {n_sample}  (seed={args.seed})")
    print(f"  Iterations   : {args.iterations} (Phase 1) / "
          f"{args.iterations // 2} (Phase 2)")
    print(f"  Graph PKL    : {args.graph_pkl}")
    print(f"  top_k        : {args.top_k}   min_score: {args.min_score}")
    print(f"  Selected diagnoses:")
    for i, dx in enumerate(diagnoses, 1):
        print(f"    {i:>2}. {dx!r}")

    # ── Phase 3: Cold-start measurement ──────────────────────────────────────
    cold = measure_cold_start(
        args.graph_pkl,
        warm_dx=diagnoses[0],
        top_k=args.top_k,
        min_score=args.min_score,
    )
    kg = cold["kg"]

    # ── Phase 1 + Phase 2: per-diagnosis ─────────────────────────────────────
    _hr("PHASE 1 — Match-only timing  (warm model)")
    print(f"  Running {args.iterations} iterations per diagnosis …\n")

    results: list[dict] = []

    for i, dx in enumerate(diagnoses, 1):
        print(f"  [{i}/{n_sample}] {dx!r}")
        p1 = time_match_only(dx, kg, args.iterations, args.top_k, args.min_score)

        row: dict = {
            "dx":             dx,
            "sub_matches":    p1["sub_matches"],
            "sem_matches":    p1["sem_matches"],
            "phase1_sub":     _stats_row(p1["sub_times_s"]),
            "phase1_sem":     _stats_row(p1["sem_times_s"]),
            "phase1_sub_raw": p1["sub_times_s"],
            "phase1_sem_raw": p1["sem_times_s"],
            # Phase 2 slots (filled below if not skipped)
            "phase2_sub":     None,
            "phase2_sem":     None,
            "phase2_sub_raw": [],
            "phase2_sem_raw": [],
            "sub_result":     {},
            "sem_result":     {},
        }

        if not args.no_full_row:
            n_full = max(1, args.iterations // 2)
            p2 = time_full_row(dx, kg, n_full, args.top_k, args.min_score)
            row["phase2_sub"]     = _stats_row(p2["sub_times_s"])
            row["phase2_sem"]     = _stats_row(p2["sem_times_s"])
            row["phase2_sub_raw"] = p2["sub_times_s"]
            row["phase2_sem_raw"] = p2["sem_times_s"]
            row["sub_result"]     = p2["sub_result"]
            row["sem_result"]     = p2["sem_result"]

        results.append(row)

    # ── Print reports ─────────────────────────────────────────────────────────
    print_phase1_table(results)
    if not args.no_full_row:
        print_phase2_table(results)
    print_output_comparison(results)
    print_aggregate_summary(results, cold)

    # ── Export raw timings CSV ────────────────────────────────────────────────
    export_csv(results, cold, args.output_csv)

    print("\nDone.\n")


if __name__ == "__main__":
    main()
