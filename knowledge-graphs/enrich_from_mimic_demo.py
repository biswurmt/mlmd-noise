"""
enrich_from_mimic_demo.py
=========================
Enriches the triage KG with real-world diagnosis-procedure co-occurrence
counts from the MIMIC-IV Clinical Database Demo.

MIMIC-IV Demo (physionet.org/content/mimic-iv-demo/2.2/) is a freely
available, de-identified subset of MIMIC-IV covering 100 patients and
~130 hospital admissions.  No credentialing is required.

Enrichment strategy
-------------------
For each hospital admission in the demo we check whether:
  (a) the admission's ICD-10 diagnosis codes match any KG Condition node, AND
  (b) the admission's procedure codes match any KG Diagnostic_Test node
      (matched via keyword search against d_icd_procedures long_title).

When both conditions are true the (Condition, Test) pair is counted.
The resulting counts are attached as additive attributes on
REQUIRES_TEST and DIRECTLY_INDICATES_TEST edges:

  mimic_demo_count       — raw co-occurrence count (0–~130)
  mimic_demo_normalized  — count / total_admissions

These are purely additive — existing guideline-derived edge weights
(literature_articles, source) are never modified.

MIMIC files needed (place in --mimic-dir, default: knowledge-graphs/mimic_demo/)
---------------------------------------------------------------------------------
  diagnoses_icd.csv    — ICD diagnosis codes per admission
  procedures_icd.csv   — ICD procedure codes per admission
  d_icd_procedures.csv — procedure code descriptions (for keyword matching)

The script also accepts files in a hosp/ subdirectory, matching the
original MIMIC-IV Demo directory layout.

Reads:
    triage_knowledge_graph_enriched.pkl  (ClinGraph-enriched KG, or base)
    mimic_demo/diagnoses_icd.csv
    mimic_demo/procedures_icd.csv
    mimic_demo/d_icd_procedures.csv

Writes:
    triage_knowledge_graph_mimic_enriched.pkl

Usage:
    python enrich_from_mimic_demo.py
    python enrich_from_mimic_demo.py --mimic-dir path/to/mimic_demo/
    python enrich_from_mimic_demo.py --base-pkl triage_knowledge_graph.pkl
"""

import argparse
import copy
import os
import pickle
import sys
from collections import defaultdict

import networkx as nx
import pandas as pd


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

# Keywords to search against d_icd_procedures long_title (case-insensitive).
# A procedure matches a test if ANY keyword is found in its description.
# Covers both ICD-9-CM and ICD-10-PCS procedure descriptions.
_TEST_DESCRIPTION_KEYWORDS: dict[str, list[str]] = {
    # Node labels come directly from KG (not title-cased) — "ECG" not "Ecg"
    "ECG": [
        "electrocardiogram",
        "electrocardiograph",
        "cardiac electrical activity",
        " ecg ",
        " ekg ",
    ],
    "Testicular Ultrasound": [
        "testicular",
        "scrotal",
        "scrotum",
        "ultrason of testes",
        "ultrason of testis",
        "ultrason of scrotum",
    ],
    "Arm X-Ray": [
        "radiograph of upper extremit",
        "radiograph of lower arm",
        "radiograph of upper arm",
        "radiograph of wrist",
        "radiograph of forearm",
        "radiograph of humerus",
        "radiograph of radius",
        "radiograph of ulna",
        "x-ray of upper extremit",
        "x-ray of arm",
        "plain film of arm",
        "plain film of wrist",
        "plain film of forearm",
        "plain film of upper extremit",
    ],
    "Appendix Ultrasound": [
        "ultrasound of abdomen",
        "diagnostic ultrasound of abdomen",
        "ultrasonography of abdomen",
        "ultrason of appendix",
    ],
    "Abdominal Ultrasound": [
        "ultrasound of abdomen",
        "diagnostic ultrasound of abdomen",
        "ultrasonography of abdomen",
        "diagnostic ultrasound of digestive",
    ],
    "CT Head": [
        "computerized axial tomography of head",
        "ct of head",
        "cat scan of head",
        "computerized tomography of head",
        "computed tomography of head",
        "tomography of head and neck",
    ],
}

# Fallback hard-coded ICD-10-PCS codes used when keyword search finds nothing.
# These are the canonical 7-character ICD-10-PCS representations.
_TEST_FALLBACK_CODES: dict[str, list[str]] = {
    "ECG":                    ["4A02X4Z"],
    "Testicular Ultrasound":  ["BV43ZZZ", "BV40ZZZ"],
    "Arm X-Ray":              ["BP2XZZZ", "BP3XZZZ", "BP4XZZZ", "BP0XZZZ", "BPWXZZZ"],
    "Appendix Ultrasound":    ["BW43ZZZ", "BW40ZZZ"],
    "Abdominal Ultrasound":   ["BW40ZZZ", "BW43ZZZ"],
    "CT Head":                ["B030ZZZ", "B030YZZ"],
}


# ─────────────────────────────────────────────────────────────────────────────
# File Location Helper
# ─────────────────────────────────────────────────────────────────────────────

def _find_mimic_file(mimic_dir: str, filename: str) -> str:
    """
    Locate a MIMIC CSV file, checking both mimic_dir/ and mimic_dir/hosp/
    (the original MIMIC-IV Demo directory layout places files in hosp/).
    Accepts both plain .csv and gzip-compressed .csv.gz variants.
    Raises FileNotFoundError with download instructions if not found.
    """
    candidates = [
        os.path.join(mimic_dir, filename),
        os.path.join(mimic_dir, filename + ".gz"),
        os.path.join(mimic_dir, "hosp", filename),
        os.path.join(mimic_dir, "hosp", filename + ".gz"),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    raise FileNotFoundError(
        f"MIMIC file '{filename}' (or '{filename}.gz') not found.\n"
        f"  Tried:\n"
        + "\n".join(f"    {p}" for p in candidates)
        + f"\n\n  Download MIMIC-IV Demo (no account required):\n"
        f"    https://physionet.org/content/mimic-iv-demo/2.2/\n"
        f"  Then copy the hosp/ CSV files into: {mimic_dir}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Data Loaders
# ─────────────────────────────────────────────────────────────────────────────

def load_mimic_tables(
    mimic_dir: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load and validate:
      diagnoses_icd.csv    — diagnosis codes per admission
      procedures_icd.csv   — procedure codes per admission
      d_icd_procedures.csv — procedure code descriptions

    Filters both diagnoses and procedures to ICD-10 records only
    (icd_version == "10") so that ICD-9 prefix collisions don't
    produce false co-occurrence matches against our ICD-10-coded KG.

    Returns: (diagnoses_df, procedures_df, proc_desc_df)
    """
    diag_path      = _find_mimic_file(mimic_dir, "diagnoses_icd.csv")
    proc_path      = _find_mimic_file(mimic_dir, "procedures_icd.csv")
    proc_desc_path = _find_mimic_file(mimic_dir, "d_icd_procedures.csv")

    print(f"  Reading {diag_path} ...")
    diag = pd.read_csv(diag_path, dtype=str)
    diag.columns = [c.strip() for c in diag.columns]
    print(f"  Columns: {list(diag.columns)}")
    _require_columns(diag, ["hadm_id", "icd_code"], "diagnoses_icd.csv")
    if "icd_version" in diag.columns:
        diag = diag[diag["icd_version"].astype(str).str.strip() == "10"].copy()
    print(f"  Loaded {len(diag):,} ICD-10 diagnosis rows.")

    print(f"  Reading {proc_path} ...")
    proc = pd.read_csv(proc_path, dtype=str)
    proc.columns = [c.strip() for c in proc.columns]
    print(f"  Columns: {list(proc.columns)}")
    _require_columns(proc, ["hadm_id", "icd_code"], "procedures_icd.csv")
    # Keep all procedure versions — d_icd_procedures descriptions cover both
    # ICD-9-CM and ICD-10-PCS, so keyword matching works across versions.
    print(f"  Loaded {len(proc):,} procedure rows.")

    print(f"  Reading {proc_desc_path} ...")
    proc_desc = pd.read_csv(proc_desc_path, dtype=str)
    proc_desc.columns = [c.strip() for c in proc_desc.columns]
    print(f"  Columns: {list(proc_desc.columns)}")
    _require_columns(proc_desc, ["icd_code"], "d_icd_procedures.csv")
    print(f"  Loaded {len(proc_desc):,} procedure descriptions.")

    return diag, proc, proc_desc


def _require_columns(df: pd.DataFrame, required: list[str], filename: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(
            f"{filename}: missing expected column(s): {missing}\n"
            f"Found: {list(df.columns)}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Test Node → Procedure Code Matching
# ─────────────────────────────────────────────────────────────────────────────

def build_test_procedure_codes(
    proc_desc: pd.DataFrame,
    test_labels: list[str],
) -> dict[str, set[str]]:
    """
    For each KG Test node label (e.g. "Ecg", "Arm X-Ray"), find all matching
    ICD procedure codes by searching d_icd_procedures long_title descriptions.
    Falls back to _TEST_FALLBACK_CODES when no keyword match is found.

    Returns: {test_label: set of icd_code strings}
    """
    # Find the description column
    title_col = None
    for candidate in ["long_title", "long_title_x", "title", "description", "short_title"]:
        if candidate in proc_desc.columns:
            title_col = candidate
            break
    if title_col is None:
        raise KeyError(
            f"d_icd_procedures.csv: cannot find a description column.\n"
            f"Tried: ['long_title', 'title', 'description', 'short_title']\n"
            f"Found: {list(proc_desc.columns)}"
        )

    desc_lower = proc_desc[title_col].astype(str).str.lower()
    result: dict[str, set[str]] = {}

    for test_label in test_labels:
        keywords = _TEST_DESCRIPTION_KEYWORDS.get(test_label, [])
        matched_codes: set[str] = set()

        for kw in keywords:
            mask = desc_lower.str.contains(kw.lower(), na=False, regex=False)
            matched_codes.update(
                proc_desc.loc[mask, "icd_code"].astype(str).str.strip().str.upper().tolist()
            )

        if matched_codes:
            print(f"    '{test_label}': {len(matched_codes)} procedure code(s) via keyword match.")
        else:
            fallback = [c.upper() for c in _TEST_FALLBACK_CODES.get(test_label, [])]
            matched_codes.update(fallback)
            if fallback:
                print(
                    f"    '{test_label}': no keyword matches — "
                    f"using {len(fallback)} fallback ICD-10-PCS code(s)."
                )
            else:
                print(f"    '{test_label}': WARNING — no keyword matches and no fallback codes defined.")

        result[test_label] = matched_codes

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Co-occurrence Matrix
# ─────────────────────────────────────────────────────────────────────────────

def build_cooccurrence_matrix(
    G: nx.DiGraph,
    diagnoses_df: pd.DataFrame,
    procedures_df: pd.DataFrame,
    proc_desc: pd.DataFrame,
) -> tuple[dict[tuple[str, str], int], int]:
    """
    Count how many MIMIC admissions have both:
      - a diagnosis matching a KG Condition node (ICD-10 prefix match), AND
      - a procedure matching a KG Diagnostic_Test node (via keyword-matched codes)

    ICD-10 prefix matching: Condition node icd10_code "I21" matches any
    admission diagnosis code starting with "I21" (e.g. "I21.0", "I21.9").

    Returns:
        cooccurrence  — {(condition_node_id, test_node_id): count}
        total_admissions — number of unique admissions in the demo
    """
    # ── Collect Condition nodes with ICD-10 prefixes ──────────────────────
    condition_nodes: dict[str, str] = {}  # node_id → 3-char ICD-10 prefix
    for node, attrs in G.nodes(data=True):
        if attrs.get("type") != "Condition":
            continue
        icd = attrs.get("icd10_code")
        if not icd or str(icd).lower() in ("nan", "none", ""):
            continue
        prefix = str(icd).strip().split(".")[0][:3].upper()
        if prefix:
            condition_nodes[node] = prefix

    if not condition_nodes:
        print("  WARNING: No Condition nodes have icd10_code populated.")
        print("  Re-run build_kg.py with a UMLS API key to populate ICD-10 codes first.")
        return {}, 0

    print(f"  Condition nodes with ICD-10 prefixes: {len(condition_nodes)}")
    for node, prefix in condition_nodes.items():
        print(f"    {node:<50} ICD-10 prefix: {prefix}")

    # ── Collect Test nodes and match to procedure codes ───────────────────
    test_node_map: dict[str, str] = {
        node: node.split(": ", 1)[-1]
        for node, attrs in G.nodes(data=True)
        if attrs.get("type") == "Diagnostic_Test"
    }
    if not test_node_map:
        print("  WARNING: No Diagnostic_Test nodes found in KG.")
        return {}, 0

    print(f"  Test nodes found: {list(test_node_map.values())}")
    print("  Matching test nodes to procedure codes ...")
    unique_labels = list(dict.fromkeys(test_node_map.values()))  # preserve order, dedupe
    test_proc_codes = build_test_procedure_codes(proc_desc, unique_labels)

    # ── Build per-admission lookups ───────────────────────────────────────
    hadm_diag: dict[str, set[str]] = defaultdict(set)
    for _, row in diagnoses_df.iterrows():
        code = str(row["icd_code"]).strip().upper()
        hadm_diag[str(row["hadm_id"])].add(code)

    hadm_proc: dict[str, set[str]] = defaultdict(set)
    for _, row in procedures_df.iterrows():
        code = str(row["icd_code"]).strip().upper()
        hadm_proc[str(row["hadm_id"])].add(code)

    all_hadm_ids = set(hadm_diag.keys()) | set(hadm_proc.keys())
    total_admissions = len(all_hadm_ids)
    print(f"  Total unique admissions: {total_admissions}")

    # ── Count co-occurrences ──────────────────────────────────────────────
    cooccurrence: dict[tuple[str, str], int] = defaultdict(int)

    for hadm_id in all_hadm_ids:
        diag_codes = hadm_diag.get(hadm_id, set())
        proc_codes = hadm_proc.get(hadm_id, set())
        if not diag_codes or not proc_codes:
            continue

        for cond_node, icd_prefix in condition_nodes.items():
            cond_present = any(code.startswith(icd_prefix) for code in diag_codes)
            if not cond_present:
                continue

            for test_node, test_label in test_node_map.items():
                match_codes = test_proc_codes.get(test_label, set())
                if proc_codes & match_codes:
                    cooccurrence[(cond_node, test_node)] += 1

    return dict(cooccurrence), total_admissions


# ─────────────────────────────────────────────────────────────────────────────
# Edge Enrichment
# ─────────────────────────────────────────────────────────────────────────────

def enrich_edges(
    G: nx.DiGraph,
    cooccurrence: dict[tuple[str, str], int],
    total_admissions: int,
) -> int:
    """
    Annotate REQUIRES_TEST and DIRECTLY_INDICATES_TEST edges with MIMIC
    co-occurrence counts.

    REQUIRES_TEST (Condition → Test):
        Direct lookup: cooccurrence[(condition_node, test_node)]

    DIRECTLY_INDICATES_TEST (Finding → Test):
        Walk the finding's INDICATES_CONDITION edges to collect all associated
        conditions, then take the max co-occurrence count across them.
        (Max rather than sum to avoid double-counting patients with multiple
        related diagnoses in the same admission.)

    Adds per edge:
        mimic_demo_count       — int, raw co-occurrence count
        mimic_demo_normalized  — float, count / total_admissions

    Returns the number of edges annotated.
    """
    annotated = 0

    for u, v, data in G.edges(data=True):
        rel = data.get("relationship", "")

        if rel == "REQUIRES_TEST":
            count = cooccurrence.get((u, v), 0)
            if count > 0:
                G[u][v]["mimic_demo_count"] = count
                G[u][v]["mimic_demo_normalized"] = round(count / total_admissions, 6)
                annotated += 1

        elif rel == "DIRECTLY_INDICATES_TEST":
            # u = clinical finding (Symptom, Vital, etc.), v = Diagnostic_Test
            max_count = 0
            for _, cond_node, ind_data in G.out_edges(u, data=True):
                if ind_data.get("relationship") == "INDICATES_CONDITION":
                    c = cooccurrence.get((cond_node, v), 0)
                    if c > max_count:
                        max_count = c
            if max_count > 0:
                G[u][v]["mimic_demo_count"] = max_count
                G[u][v]["mimic_demo_normalized"] = round(max_count / total_admissions, 6)
                annotated += 1

    print(f"  Annotated {annotated} edge(s) with mimic_demo_count.")
    return annotated


# ─────────────────────────────────────────────────────────────────────────────
# Co-occurrence Report
# ─────────────────────────────────────────────────────────────────────────────

def print_cooccurrence_report(
    cooccurrence: dict[tuple[str, str], int],
    total_admissions: int,
) -> None:
    """Print a summary table of co-occurrence pairs sorted by count descending."""
    if not cooccurrence:
        print("\n  No co-occurrences found between KG Condition and Test nodes.")
        print("  This is expected with only 100 patients — counts will often be 0.")
        print("  Check that icd10_code attributes are populated on Condition nodes.")
        return

    rows = sorted(cooccurrence.items(), key=lambda x: -x[1])
    w_cond = max(len(c.split(": ", 1)[-1]) for (c, _), _ in rows) + 2
    w_test = max(len(t.split(": ", 1)[-1]) for (_, t), _ in rows) + 2
    w_cond = max(w_cond, 20)
    w_test = max(w_test, 20)

    header = f"  {'Condition':<{w_cond}} {'Test':<{w_test}} {'Count':>6}  {'%':>6}"
    sep    = f"  {'-'*w_cond} {'-'*w_test} {'-'*6}  {'-'*6}"
    print(f"\n{header}\n{sep}")
    for (cond, test), count in rows:
        cond_label = cond.split(": ", 1)[-1]
        test_label = test.split(": ", 1)[-1]
        pct = 100.0 * count / total_admissions if total_admissions else 0.0
        print(f"  {cond_label:<{w_cond}} {test_label:<{w_test}} {count:>6}  {pct:>5.1f}%")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    script_dir = os.path.dirname(os.path.abspath(__file__))

    parser = argparse.ArgumentParser(
        description=(
            "Enrich the triage KG with MIMIC-IV Clinical Database Demo "
            "diagnosis-procedure co-occurrence counts."
        )
    )
    parser.add_argument(
        "--base-pkl",
        default=os.path.join(script_dir, "triage_knowledge_graph_enriched.pkl"),
        help=(
            "Input KG pickle (default: triage_knowledge_graph_enriched.pkl). "
            "Falls back to triage_knowledge_graph.pkl if not found."
        ),
    )
    parser.add_argument(
        "--mimic-dir",
        default=os.path.join(script_dir, "data", "mimic_demo_data"),
        help=(
            "Directory containing MIMIC-IV Demo CSV files "
            "(default: data/mimic_demo_data/). Accepts files directly in "
            "the directory or inside a hosp/ subdirectory."
        ),
    )
    parser.add_argument(
        "--out-pkl",
        default=os.path.join(
            script_dir, "triage_knowledge_graph_mimic_enriched.pkl"
        ),
        help="Output pickle path (default: triage_knowledge_graph_mimic_enriched.pkl)",
    )
    args = parser.parse_args()

    # ── Fallback base pkl ─────────────────────────────────────────────────
    if not os.path.exists(args.base_pkl):
        fallback = os.path.join(script_dir, "triage_knowledge_graph.pkl")
        print(f"  NOTE: {os.path.basename(args.base_pkl)} not found — trying {os.path.basename(fallback)}")
        args.base_pkl = fallback

    if not os.path.exists(args.base_pkl):
        print(f"ERROR: Base KG pickle not found: {args.base_pkl}", file=sys.stderr)
        print("  Run build_kg.py first to generate the base KG.", file=sys.stderr)
        sys.exit(1)

    if not os.path.isdir(args.mimic_dir):
        print(
            f"ERROR: MIMIC demo directory not found: {args.mimic_dir}\n"
            f"  Download the MIMIC-IV Clinical Database Demo (no account required):\n"
            f"    https://physionet.org/content/mimic-iv-demo/2.2/\n"
            f"  Then copy the hosp/ CSV files into: {args.mimic_dir}\n"
            f"  Or pass a custom path: --mimic-dir path/to/your/files/",
            file=sys.stderr,
        )
        sys.exit(1)

    # ── Step 1: Load base KG ──────────────────────────────────────────────
    print(f"\n[1/5] Loading base KG from {os.path.basename(args.base_pkl)} ...")
    with open(args.base_pkl, "rb") as f:
        G_base: nx.DiGraph = pickle.load(f)
    print(
        f"  Base KG: {G_base.number_of_nodes()} nodes, "
        f"{G_base.number_of_edges()} edges."
    )

    # ── Step 2: Load MIMIC tables ─────────────────────────────────────────
    print("\n[2/5] Loading MIMIC-IV Demo tables ...")
    diagnoses_df, procedures_df, proc_desc = load_mimic_tables(args.mimic_dir)

    # ── Step 3: Build co-occurrence matrix ────────────────────────────────
    print("\n[3/5] Building diagnosis-procedure co-occurrence matrix ...")
    cooccurrence, total_admissions = build_cooccurrence_matrix(
        G_base, diagnoses_df, procedures_df, proc_desc
    )
    print_cooccurrence_report(cooccurrence, total_admissions)

    # ── Step 4: Annotate edges ────────────────────────────────────────────
    print("\n[4/5] Annotating edges with MIMIC co-occurrence counts ...")
    G_enriched = copy.deepcopy(G_base)
    annotated = enrich_edges(G_enriched, cooccurrence, total_admissions)

    # ── Step 5: Save ──────────────────────────────────────────────────────
    print(f"\n[5/5] Saving enriched KG ...")
    with open(args.out_pkl, "wb") as f:
        pickle.dump(G_enriched, f)

    n_base_n = G_base.number_of_nodes()
    n_base_e = G_base.number_of_edges()
    n_enr_n  = G_enriched.number_of_nodes()
    n_enr_e  = G_enriched.number_of_edges()

    print(f"\n{'─'*58}")
    print(f"  Base KG    :  {n_base_n:>5} nodes  {n_base_e:>5} edges")
    print(f"  MIMIC Demo :  {n_enr_n - n_base_n:>+5} nodes  {n_enr_e - n_base_e:>+5} edges")
    print(f"               {annotated} edge(s) annotated with mimic_demo_count")
    print(f"  Enriched   :  {n_enr_n:>5} nodes  {n_enr_e:>5} edges")
    print(f"{'─'*58}")
    print(f"  Saved → {args.out_pkl}")
    print(
        f"\n  Note: counts are from {total_admissions} admissions "
        f"(MIMIC-IV Demo, 100 patients).\n"
        f"  For statistically robust evidence weights, consider upgrading\n"
        f"  to the full MIMIC-IV-Ext-CEKG (physionet.org, requires DUA)."
    )

    # ── Regression check ──────────────────────────────────────────────────
    missing_edges = [
        (u, v) for u, v in G_base.edges() if not G_enriched.has_edge(u, v)
    ]
    if missing_edges:
        print(
            f"\nWARNING: {len(missing_edges)} base edge(s) missing from enriched graph!",
            file=sys.stderr,
        )
        for u, v in missing_edges[:5]:
            print(f"  Missing: {u} → {v}", file=sys.stderr)
    else:
        print("  Regression check: all base edges preserved. ✓")


if __name__ == "__main__":
    main()
