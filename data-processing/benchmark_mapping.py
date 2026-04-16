#!/usr/bin/env python3
"""
benchmark_mapping.py — Compare timing of substring vs semantic KG mapping vs regex.

Randomly samples --sample *unique* diagnoses from an input CSV (dx column)
to form a pool, then runs three measurement phases. In Phases 1 and 2
each timing iteration randomly draws one diagnosis from that pool, so
timings are spread across all inputs rather than repeating the same string:

  Phase 1 — Match-only hot-path (--iterations reps each):
      Substring : get_graph_matches(dx, kg)
      Semantic  : get_graph_matches_semantic(dx, top_k, min_score)
      Regex     : match dx against 4 compiled regex lists → test labels

  Phase 2 — Full single-row pipeline (--iterations // 2 reps, or --no-full-row to skip):
      Substring : match → get_tests_for_node → format output strings
      Semantic  : same, plus score aggregation / formatting
      Regex     : same match step (no KG lookup needed)

  Phase 3 — Cold-start overhead (measured once before Phase 1):
      Substring : time to load KG from PKL
      Semantic  : time to initialise ChromaDB + embed the first query
      Regex     : time to compile all regex patterns

Usage:
    python benchmark_mapping.py --input-csv <path> [options]

Options:
    --input-csv       PATH   Input CSV with a 'dx' column (required)
    --sample          N      Rows to randomly select (default: 10)
    --seed            S      Random seed for reproducibility (default: 42)
    --iterations      N      Timing reps per diagnosis in Phase 1 (default: 50)
                             Phase 2 uses N // 2 reps
    --no-full-row            Skip Phase 2
    --output-csv      PATH   Raw-timing CSV export (default: benchmark_results_<ts>.csv)
    --top-k           K      Semantic ChromaDB top_k (default: 5)
    --min-score       F      Semantic cosine similarity threshold (default: 0.50)
    --graph-pkl       PATH   Override KG pickle path (default: from KG_PKL_FILE env)
    --regex-patterns  PATH   JSON file mapping test label → list of regex strings.
                             Defaults to the built-in REGEX_PATTERNS constant below.
"""

import argparse
import gc
import json
import os
import pickle
import random
import re
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

# ── Regex patterns (one pattern string per target test) ───────────────────────
# Each string is a pipe-separated alternation matched case-insensitively.
# Override at runtime with --regex-patterns <json_file>.

# ["US", "Abdo"]
regex_us_abdo = '[Aa]ppendicitis|[Oo]varian|[Ii]ntussusception|[Vv]olvulus|[Pp]yloric\\sstenosis|[Pp]yelonephritis|[Uu]reteric\\sstone|[Rr]enal\\sstone|[Kk]idney\\sstone|[Gg]all(\\s)?stone(s)?|[Rr]enal\\scalculi|[Bb]ile\\sduct\\sstone|[Hh]ydronephrosis|[Hh]ydroureter|[Aa]bdominal\\smass'
 
# (ii) ECG***
# ['ECG', ['12 Lead','15 Lead']]
regex_ecg_all = '[Cc]hest\\spain|[Cc]ostochondral|[Cc]ostochondroitis|[Ss]yncope|[Vv]asovagal|[Cc]hest\\sdiscomfort|[Pp]alpitation|[Mm]yocarditis|[Pp]ericarditis|[Pp]re\\-syncope|[Ff]ainting|[Hh]eart\\sblock|[Ee]ating\\sdisorder|[Bb]radycardia|[Tt]achycardia|[Ss][Vv][Tt]|[Ss]upraventricular|[Tt]etralogy|[Pp]ulmonary\\sstenosis'
 
 
# (iii) Forearm and wrist X-ray
# ['X-Ray', ['Forearm','Wrist']]
regex_xray_arm = '([Bb]uckle\\s)?[Ff]racture\\s(of|in)\\s(left\\s)?(right\\s)?(radius|ulna|wrist)'
 
# (iv) Ultrasound-Testes***
# ['US', 'Testes']
regex_us_testes = '[Tt]esticular|[Tt]estis|[Ss]crotal|[Vv]aricocele|[Ss]crotum|[Hh]ydrocele|[Ee]pididymo\\-Orchitis|[Oo]rchitis|[Ee]pididymitis'

REGEX_PATTERNS: dict[str, list[str]] = {
    "ECG":                    [regex_ecg_all],
    "Testicular Ultrasound":  [regex_us_testes],
    "Arm X-Ray":              [regex_xray_arm],
    "Appendix Ultrasound":    [regex_us_abdo],
}


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
# Regex helpers
# ─────────────────────────────────────────────────────────────────────────────

def compile_regex_patterns(patterns: dict[str, list[str]]) -> dict[str, list[re.Pattern]]:
    """Pre-compile all regex patterns (case-insensitive). Call once at startup."""
    return {
        label: [re.compile(p, re.IGNORECASE) for p in pat_list]
        for label, pat_list in patterns.items()
    }


def get_graph_matches_regex(
    dx: str,
    compiled: dict[str, list[re.Pattern]],
) -> list[str]:
    """
    Run a diagnosis string through the compiled regex pattern lists.

    Returns a list of test labels whose pattern list had at least one match.
    Order matches the iteration order of `compiled` (insertion order).
    """
    matched: list[str] = []
    for label, patterns in compiled.items():
        if any(p.search(dx) for p in patterns):
            matched.append(label)
    return matched


# ─────────────────────────────────────────────────────────────────────────────
# Phase 3 — Cold-start measurement
# ─────────────────────────────────────────────────────────────────────────────

def measure_cold_start(
    graph_pkl_path: str,
    warm_dx: str,
    top_k: int,
    min_score: float,
    regex_patterns: dict[str, list[str]],
):
    """
    Measure one-time init costs separately from per-query latency.

    Returns dict with:
        kg              networkx graph
        kg_load_s       float   seconds to load the KG pickle
        semantic_init_s float   seconds for first ChromaDB query (model load included)
        regex_compile_s float   seconds to compile all regex patterns
        compiled_regex  dict    pre-compiled regex patterns (reused in phases 1 & 2)
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

    # Regex cold start: pattern compilation
    n_patterns = sum(len(v) for v in regex_patterns.values())
    print(
        f"  [regex]     Compiling {n_patterns} patterns across "
        f"{len(regex_patterns)} tests …",
        end=" ",
        flush=True,
    )
    t0 = time.perf_counter()
    compiled_regex = compile_regex_patterns(regex_patterns)
    regex_compile_s = time.perf_counter() - t0
    print(f"{_ms(regex_compile_s)}")

    return {
        "kg":              kg,
        "kg_load_s":       kg_load_s,
        "semantic_init_s": semantic_init_s,
        "regex_compile_s": regex_compile_s,
        "compiled_regex":  compiled_regex,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Phase 1 — Match-only timing (pool-based)
# ─────────────────────────────────────────────────────────────────────────────

def time_match_pool(
    diagnoses: list[str],
    kg,
    n_iter: int,
    top_k: int,
    min_score: float,
    rng: random.Random,
    compiled_regex: dict[str, list[re.Pattern]],
) -> dict:
    """
    Run n_iter iterations, randomly picking a diagnosis from the pool each time.

    Returns:
        iter_records    list[dict]  one entry per iteration:
                            {"dx", "sub_s", "sem_s", "rgx_s",
                             "sub_matches", "sem_matches", "rgx_matches"}
        last_matches    dict[str, {"sub": list, "sem": list, "rgx": list}]
                            final match result seen for each unique dx
    """
    # Fix the random sequence so all passes see the same draws
    draw_seq = [rng.choice(diagnoses) for _ in range(n_iter)]

    # Substring pass — collect (dx, time, result) in draw order
    sub_run: list[tuple[str, float, list[dict]]] = []
    for dx in draw_seq:
        gc.collect()
        t0 = time.perf_counter()
        result = get_graph_matches(dx, kg)
        sub_run.append((dx, time.perf_counter() - t0, result))

    # Semantic pass — same draw order
    sem_run: list[tuple[str, float, list[dict]]] = []
    for dx in draw_seq:
        gc.collect()
        t0 = time.perf_counter()
        result = get_graph_matches_semantic(dx, top_k=top_k, min_score=min_score)
        sem_run.append((dx, time.perf_counter() - t0, result))

    # Regex pass — same draw order
    rgx_run: list[tuple[str, float, list[str]]] = []
    for dx in draw_seq:
        gc.collect()
        t0 = time.perf_counter()
        result = get_graph_matches_regex(dx, compiled_regex)
        rgx_run.append((dx, time.perf_counter() - t0, result))

    # Collate per-iteration records
    iter_records: list[dict] = []
    for (dx, sub_s, sub_m), (_, sem_s, sem_m), (_, rgx_s, rgx_m) in zip(
        sub_run, sem_run, rgx_run
    ):
        iter_records.append({
            "dx":          dx,
            "sub_s":       sub_s,
            "sem_s":       sem_s,
            "rgx_s":       rgx_s,
            "sub_matches": sub_m,
            "sem_matches": sem_m,
            "rgx_matches": rgx_m,
        })

    # Keep the last seen match result for each unique dx (for output comparison)
    last_matches: dict[str, dict] = {}
    for rec in iter_records:
        last_matches[rec["dx"]] = {
            "sub": rec["sub_matches"],
            "sem": rec["sem_matches"],
            "rgx": rec["rgx_matches"],
        }

    return {"iter_records": iter_records, "last_matches": last_matches}


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


def _run_regex_row(
    dx: str,
    compiled_regex: dict[str, list[re.Pattern]],
) -> dict:
    """Replicate regex pipeline logic for one dx string (multi-term aware)."""
    terms = [t.strip() for t in dx.split(";") if t.strip()]
    matched_tests: list[str] = []
    seen: set[str] = set()
    for term in terms:
        for label in get_graph_matches_regex(term, compiled_regex):
            if label not in seen:
                seen.add(label)
                matched_tests.append(label)
    return {
        "potential_tests": ", ".join(matched_tests) or "No linked tests found",
    }


def time_full_row_pool(
    diagnoses: list[str],
    kg,
    n_iter: int,
    top_k: int,
    min_score: float,
    rng: random.Random,
    compiled_regex: dict[str, list[re.Pattern]],
) -> dict:
    """
    Run n_iter iterations of the full single-row pipeline, randomly picking
    a diagnosis from the pool each time (same draw sequence for all methods).

    Returns:
        iter_records    list[dict]  one entry per iteration:
                            {"dx", "sub_s", "sem_s", "rgx_s",
                             "sub_result", "sem_result", "rgx_result"}
        last_results    dict[str, {"sub": dict, "sem": dict, "rgx": dict}]
                            last pipeline output seen per unique dx
    """
    draw_seq = [rng.choice(diagnoses) for _ in range(n_iter)]

    sub_run: list[tuple[str, float, dict]] = []
    for dx in draw_seq:
        gc.collect()
        t0 = time.perf_counter()
        result = _run_substring_row(dx, kg)
        sub_run.append((dx, time.perf_counter() - t0, result))

    sem_run: list[tuple[str, float, dict]] = []
    for dx in draw_seq:
        gc.collect()
        t0 = time.perf_counter()
        result = _run_semantic_row(dx, kg, top_k=top_k, min_score=min_score)
        sem_run.append((dx, time.perf_counter() - t0, result))

    rgx_run: list[tuple[str, float, dict]] = []
    for dx in draw_seq:
        gc.collect()
        t0 = time.perf_counter()
        result = _run_regex_row(dx, compiled_regex)
        rgx_run.append((dx, time.perf_counter() - t0, result))

    iter_records: list[dict] = []
    for (dx, sub_s, sub_r), (_, sem_s, sem_r), (_, rgx_s, rgx_r) in zip(
        sub_run, sem_run, rgx_run
    ):
        iter_records.append({
            "dx":         dx,
            "sub_s":      sub_s,
            "sem_s":      sem_s,
            "rgx_s":      rgx_s,
            "sub_result": sub_r,
            "sem_result": sem_r,
            "rgx_result": rgx_r,
        })

    last_results: dict[str, dict] = {}
    for rec in iter_records:
        last_results[rec["dx"]] = {
            "sub": rec["sub_result"],
            "sem": rec["sem_result"],
            "rgx": rec["rgx_result"],
        }

    return {"iter_records": iter_records, "last_results": last_results}


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


def _per_dx_stats(iter_records: list[dict], diagnoses: list[str], time_key: str) -> dict[str, dict]:
    """Group iteration timings by dx and compute stats for each unique diagnosis."""
    grouped: dict[str, list[float]] = {dx: [] for dx in diagnoses}
    for rec in iter_records:
        grouped[rec["dx"]].append(rec[time_key])
    return {dx: _stats_row(times) if times else {} for dx, times in grouped.items()}


def _speedup_str(baseline_ms: float, other_ms: float) -> str:
    if baseline_ms <= 0:
        return "   —    "
    ratio = other_ms / baseline_ms
    return f"{ratio:>6.1f}×"


def print_phase1_table(diagnoses: list[str], p1: dict) -> None:
    _hr("PHASE 1 — Match-only timing summary  (per unique diagnosis)")
    records = p1["iter_records"]
    sub_stats = _per_dx_stats(records, diagnoses, "sub_s")
    sem_stats = _per_dx_stats(records, diagnoses, "sem_s")
    rgx_stats = _per_dx_stats(records, diagnoses, "rgx_s")

    col_w = 35
    hdr = (
        f"{'Diagnosis':<{col_w}} {'Method':<10}"
        f" {'Mean':>9} {'Std':>9} {'Min':>9} {'Max':>9} {'n':>4}"
        f"  {'vs substr':>8}"
    )
    sep = "─" * len(hdr)
    print(hdr)
    print(sep)
    for dx in diagnoses:
        sub = sub_stats.get(dx, {})
        sem = sem_stats.get(dx, {})
        rgx = rgx_stats.get(dx, {})
        if not sub or not sem:
            print(f"  {dx[:col_w-1]!r}  (not sampled in this run)")
            continue
        dx_short = dx[:col_w - 1]
        print(
            f"{dx_short:<{col_w}} {'substring':<10}"
            f" {sub['mean_ms']:>8.3f}ms {sub['std_ms']:>8.3f}ms"
            f" {sub['min_ms']:>8.3f}ms {sub['max_ms']:>8.3f}ms {sub['n']:>4}"
        )
        print(
            f"{'':<{col_w}} {'semantic':<10}"
            f" {sem['mean_ms']:>8.3f}ms {sem['std_ms']:>8.3f}ms"
            f" {sem['min_ms']:>8.3f}ms {sem['max_ms']:>8.3f}ms {sem['n']:>4}"
            f"  {_speedup_str(sub['mean_ms'], sem['mean_ms'])}"
        )
        if rgx:
            print(
                f"{'':<{col_w}} {'regex':<10}"
                f" {rgx['mean_ms']:>8.3f}ms {rgx['std_ms']:>8.3f}ms"
                f" {rgx['min_ms']:>8.3f}ms {rgx['max_ms']:>8.3f}ms {rgx['n']:>4}"
                f"  {_speedup_str(sub['mean_ms'], rgx['mean_ms'])}"
            )
        print(sep)


def print_phase2_table(diagnoses: list[str], p2: dict) -> None:
    _hr("PHASE 2 — Full single-row pipeline timing summary  (per unique diagnosis)")
    records = p2["iter_records"]
    sub_stats = _per_dx_stats(records, diagnoses, "sub_s")
    sem_stats = _per_dx_stats(records, diagnoses, "sem_s")
    rgx_stats = _per_dx_stats(records, diagnoses, "rgx_s")

    col_w = 35
    hdr = (
        f"{'Diagnosis':<{col_w}} {'Method':<10}"
        f" {'Mean':>9} {'Std':>9} {'Min':>9} {'Max':>9} {'n':>4}"
        f"  {'vs substr':>8}"
    )
    sep = "─" * len(hdr)
    print(hdr)
    print(sep)
    for dx in diagnoses:
        sub = sub_stats.get(dx, {})
        sem = sem_stats.get(dx, {})
        rgx = rgx_stats.get(dx, {})
        if not sub or not sem:
            continue
        dx_short = dx[:col_w - 1]
        print(
            f"{dx_short:<{col_w}} {'substring':<10}"
            f" {sub['mean_ms']:>8.3f}ms {sub['std_ms']:>8.3f}ms"
            f" {sub['min_ms']:>8.3f}ms {sub['max_ms']:>8.3f}ms {sub['n']:>4}"
        )
        print(
            f"{'':<{col_w}} {'semantic':<10}"
            f" {sem['mean_ms']:>8.3f}ms {sem['std_ms']:>8.3f}ms"
            f" {sem['min_ms']:>8.3f}ms {sem['max_ms']:>8.3f}ms {sem['n']:>4}"
            f"  {_speedup_str(sub['mean_ms'], sem['mean_ms'])}"
        )
        if rgx:
            print(
                f"{'':<{col_w}} {'regex':<10}"
                f" {rgx['mean_ms']:>8.3f}ms {rgx['std_ms']:>8.3f}ms"
                f" {rgx['min_ms']:>8.3f}ms {rgx['max_ms']:>8.3f}ms {rgx['n']:>4}"
                f"  {_speedup_str(sub['mean_ms'], rgx['mean_ms'])}"
            )
        print(sep)


def print_output_comparison(diagnoses: list[str], p1: dict, p2: dict | None) -> None:
    _hr("OUTPUT COMPARISON — matched nodes & recommended tests  (last seen per dx)")
    last_matches = p1["last_matches"]
    last_results = p2["last_results"] if p2 else {}

    for dx in diagnoses:
        matches = last_matches.get(dx)
        if matches is None:
            print(f"\n  dx: {dx!r}  (not sampled)")
            continue

        print(f"\n  dx: {dx!r}")
        sub_m = _fmt_matches(matches["sub"])
        sem_m = _fmt_matches(matches["sem"])
        rgx_m = ", ".join(matches["rgx"]) if matches["rgx"] else "(no match)"
        print(f"    [substring] nodes  : {sub_m}")
        print(f"    [semantic]  nodes  : {sem_m}")
        print(f"    [regex]     tests  : {rgx_m}")

        if last_results.get(dx):
            sub_t = last_results[dx]["sub"].get("potential_tests", "—")
            sem_t = last_results[dx]["sem"].get("potential_tests", "—")
            rgx_t = last_results[dx]["rgx"].get("potential_tests", "—")
            sem_ts = last_results[dx]["sem"].get("test_scores", "")
            print(f"    [substring] tests  : {sub_t}")
            print(f"    [semantic]  tests  : {sem_t}")
            if sem_ts:
                print(f"    [semantic]  scores : {sem_ts}")
            print(f"    [regex]     tests  : {rgx_t}")

        if set(sub_m.split(" | ")) != set(sem_m.split(" | ")):
            print(f"    ⚑ substring/semantic returned DIFFERENT matched nodes for this dx")


def print_aggregate_summary(p1: dict, p2: dict | None, cold: dict) -> None:
    _hr("AGGREGATE SUMMARY")
    all_sub_p1 = [rec["sub_s"] for rec in p1["iter_records"]]
    all_sem_p1 = [rec["sem_s"] for rec in p1["iter_records"]]
    all_rgx_p1 = [rec["rgx_s"] for rec in p1["iter_records"]]
    mean_sub   = _mean(all_sub_p1) * 1000
    mean_sem   = _mean(all_sem_p1) * 1000
    mean_rgx   = _mean(all_rgx_p1) * 1000

    print(f"  Phase 1 — match-only  ({len(all_sub_p1)} iterations across pool):")
    print(f"    Substring mean         : {mean_sub:.3f} ms")
    print(f"    Semantic  mean         : {mean_sem:.3f} ms  "
          f"({_speedup_str(mean_sub, mean_sem).strip()} vs substring)")
    print(f"    Regex     mean         : {mean_rgx:.3f} ms  "
          f"({_speedup_str(mean_sub, mean_rgx).strip()} vs substring)\n")

    print(f"  Cold-start overhead:")
    print(f"    KG pickle load         : {_ms(cold['kg_load_s'])}")
    print(f"    Semantic model init    : {_ms(cold['semantic_init_s'])}")
    print(f"    Regex pattern compile  : {_ms(cold['regex_compile_s'])}")

    if p2:
        all_sub_p2 = [rec["sub_s"] for rec in p2["iter_records"]]
        all_sem_p2 = [rec["sem_s"] for rec in p2["iter_records"]]
        all_rgx_p2 = [rec["rgx_s"] for rec in p2["iter_records"]]
        if all_sub_p2 and all_sem_p2:
            mean_sub_p2 = _mean(all_sub_p2) * 1000
            mean_sem_p2 = _mean(all_sem_p2) * 1000
            mean_rgx_p2 = _mean(all_rgx_p2) * 1000
            print(f"\n  Phase 2 — full row  ({len(all_sub_p2)} iterations across pool):")
            print(f"    Substring mean         : {mean_sub_p2:.3f} ms")
            print(f"    Semantic  mean         : {mean_sem_p2:.3f} ms  "
                  f"({_speedup_str(mean_sub_p2, mean_sem_p2).strip()} vs substring)")
            print(f"    Regex     mean         : {mean_rgx_p2:.3f} ms  "
                  f"({_speedup_str(mean_sub_p2, mean_rgx_p2).strip()} vs substring)")


def export_csv(p1: dict, p2: dict | None, cold: dict, output_path: str) -> None:
    rows = []
    for i, rec in enumerate(p1["iter_records"], 1):
        sub_s = rec["sub_s"]
        rows.append({
            "dx":          rec["dx"],
            "phase":       "phase1",
            "iteration":   i,
            "substring_s": sub_s,
            "semantic_s":  rec["sem_s"],
            "regex_s":     rec["rgx_s"],
            "sem_vs_sub":  rec["sem_s"] / sub_s if sub_s > 0 else None,
            "rgx_vs_sub":  rec["rgx_s"] / sub_s if sub_s > 0 else None,
        })
    if p2:
        for i, rec in enumerate(p2["iter_records"], 1):
            sub_s = rec["sub_s"]
            rows.append({
                "dx":          rec["dx"],
                "phase":       "phase2",
                "iteration":   i,
                "substring_s": sub_s,
                "semantic_s":  rec["sem_s"],
                "regex_s":     rec["rgx_s"],
                "sem_vs_sub":  rec["sem_s"] / sub_s if sub_s > 0 else None,
                "rgx_vs_sub":  rec["rgx_s"] / sub_s if sub_s > 0 else None,
            })

    kg_load = cold["kg_load_s"]
    cold_row = pd.DataFrame([{
        "dx":          "__cold_start__",
        "phase":       "cold",
        "iteration":   1,
        "substring_s": kg_load,
        "semantic_s":  cold["semantic_init_s"],
        "regex_s":     cold["regex_compile_s"],
        "sem_vs_sub":  cold["semantic_init_s"] / kg_load if kg_load > 0 else None,
        "rgx_vs_sub":  cold["regex_compile_s"] / kg_load if kg_load > 0 else None,
    }])
    df = pd.concat([cold_row, pd.DataFrame(rows)], ignore_index=True)
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
                        help=f"Unique diagnoses to include in the pool (default: {_DEFAULT_SAMPLE}).")
    parser.add_argument("--seed",       type=int, default=_DEFAULT_SEED, metavar="S",
                        help=f"Random seed for pool selection and draw sequence (default: {_DEFAULT_SEED}).")
    parser.add_argument("--iterations", type=int, default=_DEFAULT_ITERATIONS, metavar="N",
                        help=f"Total pool draws in Phase 1 (default: {_DEFAULT_ITERATIONS}). "
                             "Each draw randomly picks one diagnosis from the pool. "
                             "Phase 2 uses N // 2 draws.")
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
    parser.add_argument("--regex-patterns", default=None, metavar="PATH",
                        help="JSON file mapping test label → list of regex strings. "
                             "Defaults to the built-in REGEX_PATTERNS constant.")
    args = parser.parse_args()

    # ── Load regex patterns ───────────────────────────────────────────────────
    if args.regex_patterns:
        with open(args.regex_patterns) as fh:
            regex_patterns: dict[str, list[str]] = json.load(fh)
    else:
        regex_patterns = REGEX_PATTERNS

    # ── Load input CSV, deduplicate dx, then sample ──────────────────────────
    _hr("Setup")
    print(f"  Input CSV    : {args.input_csv}")
    input_df = pd.read_csv(args.input_csv)
    if "dx" not in input_df.columns:
        sys.exit("  ERROR: input CSV must contain a 'dx' column.")

    # Deduplicate: work on unique dx values only
    unique_dx = input_df["dx"].dropna().unique().tolist()
    rng = random.Random(args.seed)
    n_sample = min(args.sample, len(unique_dx))
    diagnoses = rng.sample(unique_dx, k=n_sample)

    print(f"  Total rows       : {len(input_df)}")
    print(f"  Unique dx values : {len(unique_dx)}")
    print(f"  Sampled (unique) : {n_sample}  (seed={args.seed})")
    print(f"  Iterations       : {args.iterations} pool draws (Phase 1) / "
          f"{args.iterations // 2} (Phase 2)")
    print(f"  Graph PKL        : {args.graph_pkl}")
    print(f"  top_k            : {args.top_k}   min_score: {args.min_score}")
    print(f"  Regex patterns   : {args.regex_patterns or '(built-in)'} "
          f"— {len(regex_patterns)} tests, "
          f"{sum(len(v) for v in regex_patterns.values())} patterns total")
    print(f"  Diagnosis pool:")
    for i, dx in enumerate(diagnoses, 1):
        print(f"    {i:>2}. {dx!r}")

    # ── Phase 3: Cold-start measurement ──────────────────────────────────────
    cold = measure_cold_start(
        args.graph_pkl,
        warm_dx=diagnoses[0],
        top_k=args.top_k,
        min_score=args.min_score,
        regex_patterns=regex_patterns,
    )
    kg             = cold["kg"]
    compiled_regex = cold["compiled_regex"]

    # ── Phase 1: match-only, pool-based ──────────────────────────────────────
    _hr("PHASE 1 — Match-only timing  (warm model, random pool draws)")
    print(f"  {args.iterations} iterations, each randomly draws one dx from the pool …\n")
    p1 = time_match_pool(
        diagnoses, kg, args.iterations, args.top_k, args.min_score, rng, compiled_regex
    )

    # ── Phase 2: full row, pool-based ────────────────────────────────────────
    p2: dict | None = None
    if not args.no_full_row:
        n_full = max(1, args.iterations // 2)
        _hr("PHASE 2 — Full single-row pipeline  (warm model, random pool draws)")
        print(f"  {n_full} iterations …\n")
        p2 = time_full_row_pool(
            diagnoses, kg, n_full, args.top_k, args.min_score, rng, compiled_regex
        )

    # ── Print reports ─────────────────────────────────────────────────────────
    print_phase1_table(diagnoses, p1)
    if p2:
        print_phase2_table(diagnoses, p2)
    print_output_comparison(diagnoses, p1, p2)
    print_aggregate_summary(p1, p2, cold)

    # ── Export raw timings CSV ────────────────────────────────────────────────
    export_csv(p1, p2, cold, args.output_csv)

    print("\nDone.\n")


if __name__ == "__main__":
    main()
