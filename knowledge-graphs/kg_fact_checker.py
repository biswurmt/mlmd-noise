"""
kg_fact_checker.py
==================
Standalone fact-checker for the medical triage knowledge graph.

For every clinically meaningful edge in the graph this script queries
external evidence APIs and, where the primary terms fail, retries using
OLS4 synonyms stored on the nodes.  A formatted pass/fail report is
printed at the end.

Edge types checked
------------------
  INDICATES_CONDITION   → Europe PMC literature search   (threshold: >= 5 papers)
  REQUIRES_TEST         → Europe PMC literature search   (threshold: >= 5 papers)
  RECOMMENDS_TREATMENT  → ClinicalTrials.gov study count (threshold: >= 1 trial)

Edge types skipped (no external evidence check defined)
-------------------------------------------------------
  DIRECTLY_INDICATES_TEST, HAS_ADVERSE_EVENT

Usage
-----
    python kg_fact_checker.py
    python kg_fact_checker.py --kg-path /path/to/triage_knowledge_graph.pkl
    python kg_fact_checker.py --verbose      # also lists all passing edges
"""

import argparse
import pickle
import os
import requests

# =====================================================================
# CONFIG
# =====================================================================

KG_DEFAULT_PATH = "triage_knowledge_graph.pkl"

# Relationships that will be fact-checked, and their evidence config.
EDGE_CHECKS = {
    "INDICATES_CONDITION":  {"api": "literature", "threshold": 5},
    "REQUIRES_TEST":        {"api": "literature", "threshold": 5},
    "RECOMMENDS_TREATMENT": {"api": "trials",     "threshold": 1},
}

# =====================================================================
# EVIDENCE API FUNCTIONS  (no keys required; simple memory caches)
# =====================================================================

_lit_cache:   dict = {}
_trial_cache: dict = {}


def check_literature(term1: str, term2: str) -> int:
    """Return the Europe PMC hitCount for articles co-mentioning term1 and term2."""
    key = (term1.lower(), term2.lower())
    if key in _lit_cache:
        return _lit_cache[key]

    # Use parentheses for boolean grouping instead of strict exact-phrase quotes
    # This prevents symbols like ">" and "<" from breaking the search parser
    params = {
        "query":    f'({term1}) AND ({term2})',
        "format":   "json",
        "pageSize": 1,          # pageSize=0 is invalid in Europe PMC
    }
    try:
        resp = requests.get(
            "https://www.ebi.ac.uk/europepmc/webservices/rest/search",
            params=params,
            timeout=10,
        )
        resp.raise_for_status()
        count = resp.json().get("hitCount", 0)
    except Exception as exc:
        print(f"    [API WARN] Europe PMC error for ('{term1}', '{term2}'): {exc}")
        count = 0

    _lit_cache[key] = count
    return count


def check_trials(condition: str, treatment: str) -> int:
    """Return the ClinicalTrials.gov v2 totalCount for (condition, treatment)."""
    key = (condition.lower(), treatment.lower())
    if key in _trial_cache:
        return _trial_cache[key]

    params = {
        "query.cond": condition,
        "query.intr": treatment,
        "countTotal": "true",   # REQUIRED by ClinicalTrials API to return totalCount
        "pageSize":   1,
    }
    try:
        resp = requests.get(
            "https://clinicaltrials.gov/api/v2/studies",
            params=params,
            timeout=10,
        )
        resp.raise_for_status()
        count = resp.json().get("totalCount", 0)
    except Exception as exc:
        print(f"    [API WARN] ClinicalTrials.gov error for ('{condition}', '{treatment}'): {exc}")
        count = 0

    _trial_cache[key] = count
    return count

# =====================================================================
# HELPERS
# =====================================================================

def _strip_prefix(node_id: str) -> str:
    """Remove the KG node-type prefix (e.g. 'Symptom: ') leaving the bare term."""
    return node_id.split(": ", 1)[-1] if ": " in node_id else node_id


def _get_synonyms(G, node_id: str) -> list:
    """Return the synonyms list stored on a node, defaulting to []."""
    val = G.nodes[node_id].get("synonyms", [])
    return val if isinstance(val, list) else []


# =====================================================================
# CORE VALIDATION LOGIC
# =====================================================================

def _validate_edge(
    relationship: str,
    src_term: str,
    tgt_term: str,
    src_synonyms: list,
    tgt_synonyms: list,
) -> dict:
    """
    Attempt to validate a single edge using the appropriate evidence API,
    with a synonym fallback loop.

    Returns a result dict:
        status        — "passed" | "passed_synonym" | "failed" | "skipped"
        count         — best evidence count found across all attempts
        matched_src   — term that finally passed (may be a synonym)
        matched_tgt   — term that finally passed (may be a synonym)
        threshold     — the threshold that was applied
    """
    if relationship not in EDGE_CHECKS:
        return {
            "status": "skipped", "count": 0,
            "matched_src": src_term, "matched_tgt": tgt_term,
            "threshold": None,
        }

    cfg       = EDGE_CHECKS[relationship]
    threshold = cfg["threshold"]
    api_fn    = check_literature if cfg["api"] == "literature" else check_trials

    # Build candidates: primary first, then source synonyms, then target synonyms.
    # We intentionally keep cross-synonym pairs out to limit API call volume.
    candidates = [(src_term, tgt_term)]
    for syn in src_synonyms:
        candidates.append((syn, tgt_term))
    for syn in tgt_synonyms:
        candidates.append((src_term, syn))

    best_count = 0

    for src_cand, tgt_cand in candidates:
        count = api_fn(src_cand, tgt_cand)
        best_count = max(best_count, count)

        if count >= threshold:
            is_primary = (src_cand == src_term and tgt_cand == tgt_term)
            return {
                "status":      "passed" if is_primary else "passed_synonym",
                "count":       count,
                "matched_src": src_cand,
                "matched_tgt": tgt_cand,
                "threshold":   threshold,
            }

    # No candidate passed
    return {
        "status":      "failed",
        "count":       best_count,
        "matched_src": src_term,
        "matched_tgt": tgt_term,
        "threshold":   threshold,
    }


# =====================================================================
# MAIN
# =====================================================================

def run_fact_checker(kg_path: str, verbose: bool = False) -> None:
    # --- Load graph ---
    if not os.path.exists(kg_path):
        raise FileNotFoundError(
            f"Graph not found at '{kg_path}'. Run 'python build_kg.py' first."
        )
    with open(kg_path, "rb") as fh:
        G = pickle.load(fh)

    print(f"\nLoaded graph: {kg_path}")
    print(f"  Nodes: {G.number_of_nodes()}  |  Edges: {G.number_of_edges()}\n")
    print("Running fact-checks...\n")

    # Accumulate results
    passed         = []   # (src_id, tgt_id, rel, count)
    passed_synonym = []   # (src_id, tgt_id, rel, count, matched_src, matched_tgt)
    failed         = []   # (src_id, tgt_id, rel, count, threshold)
    skipped_counts = {}   # relationship → int

    total_checked = 0

    for src_id, tgt_id, edge_data in G.edges(data=True):
        rel = edge_data.get("relationship", "")

        if rel not in EDGE_CHECKS:
            skipped_counts[rel] = skipped_counts.get(rel, 0) + 1
            continue

        total_checked += 1
        src_term     = _strip_prefix(src_id)
        tgt_term     = _strip_prefix(tgt_id)
        src_synonyms = _get_synonyms(G, src_id)
        tgt_synonyms = _get_synonyms(G, tgt_id)

        result = _validate_edge(rel, src_term, tgt_term, src_synonyms, tgt_synonyms)

        if result["status"] == "passed":
            passed.append((src_id, tgt_id, rel, result["count"]))

        elif result["status"] == "passed_synonym":
            passed_synonym.append((
                src_id, tgt_id, rel, result["count"],
                result["matched_src"], result["matched_tgt"],
            ))

        elif result["status"] == "failed":
            failed.append((src_id, tgt_id, rel, result["count"], result["threshold"]))

    # =====================================================================
    # REPORT
    # =====================================================================
    sep    = "=" * 66
    subsep = "-" * 66

    total_passed = len(passed) + len(passed_synonym)
    pass_rate    = (total_passed / total_checked * 100) if total_checked else 0.0

    print(sep)
    print("  KNOWLEDGE GRAPH FACT-CHECKER REPORT")
    print(sep)
    print(f"  Graph       : {kg_path}")
    print(f"  Checked     : {total_checked} edges  "
          f"({', '.join(EDGE_CHECKS.keys())})")

    if skipped_counts:
        skip_detail = ", ".join(f"{r}: {n}" for r, n in sorted(skipped_counts.items()))
        print(f"  Skipped     : {sum(skipped_counts.values())} edges  ({skip_detail})")

    print(f"  Passed      : {total_passed} / {total_checked}  ({pass_rate:.1f}%)")
    print(f"    ├─ primary terms : {len(passed)}")
    print(f"    └─ via synonym   : {len(passed_synonym)}")
    print(f"  Failed      : {len(failed)}")
    print(sep)

    # --- Verbose: list primary-passed edges ---
    if verbose and passed:
        print("\n  PASSED  (primary terms)")
        print(subsep)
        for src_id, tgt_id, rel, count in passed:
            label = "papers" if EDGE_CHECKS[rel]["api"] == "literature" else "trials"
            print(f"  [PASSED]  {src_id}  →  {tgt_id}")
            print(f"            {rel}  |  {label}: {count:,}")
        print()

    # --- Passed via synonym ---
    if passed_synonym:
        print("\n  PASSED  (via synonym fallback)")
        print(subsep)
        for src_id, tgt_id, rel, count, m_src, m_tgt in passed_synonym:
            label = "papers" if EDGE_CHECKS[rel]["api"] == "literature" else "trials"
            print(f"  [PASSED via Synonym]  {src_id}  →  {tgt_id}")
            print(f"    Matched using : '{m_src}'  AND  '{m_tgt}'")
            print(f"    {rel}  |  {label}: {count:,}")
        print()

    # --- Failed / flagged ---
    if failed:
        print("\n  FLAGGED / FAILED")
        print(subsep)
        for src_id, tgt_id, rel, count, threshold in failed:
            label = "papers" if EDGE_CHECKS[rel]["api"] == "literature" else "trials"
            print(f"  [FLAGGED]  {src_id}  →  {tgt_id}")
            print(f"    {rel}  |  {label} found: {count:,}  "
                  f"(threshold: >= {threshold})")
        print()
    else:
        print("\n  No edges failed validation.\n")

    print(sep)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fact-check the medical triage knowledge graph against "
                    "Europe PMC and ClinicalTrials.gov."
    )
    parser.add_argument(
        "--kg-path",
        default=KG_DEFAULT_PATH,
        help=f"Path to the serialized graph .pkl file (default: {KG_DEFAULT_PATH})",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Also print all passing edges, not just failures and synonym matches.",
    )
    args = parser.parse_args()
    run_fact_checker(kg_path=args.kg_path, verbose=args.verbose)
