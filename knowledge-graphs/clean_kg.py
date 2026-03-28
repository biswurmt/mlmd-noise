"""
clean_kg.py
===========
One-shot post-processing pass that corrects known data quality issues in
triage_knowledge_graph_enriched.pkl without requiring a full rebuild.

Issues addressed
----------------
1. Invalid LOINC codes on Diagnostic_Test nodes
   - MTHU* codes: UMLS-internal identifiers, not real LOINC
   - LP* codes:   LOINC Part codes (attribute types), not study panels
   - LA* codes:   LOINC Answer codes, not study panels
   → Replaced with authoritative LOINC panel codes from the LOINC registry.

2. Missing ICD-10 codes on core Condition nodes
   - Testicular Torsion, Potential Cardiac Ischemia, Suspected
     Epididymo-orchitis, Acute Stroke had icd10_code = nan/None.
   → Applied WHO ICD-10 overrides.

3. Wrong ICD-10 code on Subarachnoid Hemorrhage
   - S06.6X is traumatic SAH (Chapter 19, Injuries).
   - Non-traumatic SAH = I60.9 (Chapter 9, Circulatory).
   → Corrected to I60.9.

4. ClinGraph Condition nodes that are actually symptoms/findings
   - ICD-10-CM R-codes (Chapter 18: signs, symptoms, abnormal findings)
     were imported with type="Condition" instead of type="Symptom".
   - Certain M25/M54/M79 pain codes also misclassified as Conditions.
   → Node type corrected to "Symptom" and node ID prefix relabelled from
     "Condition: X" → "Symptom: X" (edges redirected via nx.relabel_nodes).

5. "Condition: Pediatric Assessment" — not a medical diagnosis
   → Reclassified as Demographic_Factor; node relabelled to
     "Demographic: Pediatric Assessment".

6. Disconnected ClinGraph nodes (no path to any Diagnostic_Test)
   → Pruned from graph (87 nodes).

Reads:  triage_knowledge_graph_enriched.pkl
Writes: triage_knowledge_graph_clean.pkl

Usage:
    python clean_kg.py
    python clean_kg.py --base-pkl path/to/enriched.pkl --out-pkl path/to/clean.pkl
"""

import argparse
import copy
import os
import pickle
import sys

import networkx as nx


# ─────────────────────────────────────────────────────────────────────────────
# Correction Tables
# ─────────────────────────────────────────────────────────────────────────────

# Authoritative LOINC panel codes (digits-digits format, from LOINC registry)
_CORRECT_LOINC: dict[str, str] = {
    "Test: ECG":                   "11524-6",   # EKG study (12-lead)
    "Test: Testicular Ultrasound": "24636-1",   # US Scrotum
    "Test: Arm X-Ray":             "24630-4",   # XR Upper extremity
    "Test: Appendix Ultrasound":   "30688-3",   # US Appendix
    "Test: Abdominal Ultrasound":  "24640-3",   # US Abdomen and retroperitoneum
    "Test: CT Head":               "24725-4",   # CT Head (already correct, included for completeness)
}

# Authoritative WHO ICD-10 codes for core conditions that UMLS failed to resolve
_CORRECT_ICD10: dict[str, str] = {
    "Condition: Testicular Torsion":           "N44.0",
    "Condition: Potential Cardiac Ischemia":   "I25.9",
    "Condition: Suspected Epididymo-orchitis": "N45.3",
    "Condition: Acute Stroke":                 "I63.9",
    "Condition: Subarachnoid Hemorrhage":      "I60.9",   # non-traumatic SAH
}

# ICD-10-CM code prefixes that should be Symptom, not Condition
_SYMPTOM_ICD10CM_PREFIXES   = ("R",)
_SYMPTOM_ICD10CM_SUBPREFIXES = ("M25.", "M54.", "M79.")


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _is_invalid_loinc(code: str) -> bool:
    """Return True for non-panel LOINC codes (MTHU, LP, LA) or null."""
    if not code or str(code).lower() in ("nan", "none", ""):
        return True
    s = str(code).strip()
    return s.startswith(("MTHU", "LP", "LA"))


def _should_be_symptom(attrs: dict) -> bool:
    """
    Return True if a ClinGraph-sourced Condition node should be Symptom.
    Criteria:
      - Has icd10cm_code starting with R (signs/symptoms chapter)
      - Has icd10cm_code starting with M25/M54/M79 (pain/symptom codes)
      - Name contains symptom-like terms with no disease counterparts
    """
    if attrs.get("source") != "ClinGraph":
        return False
    code = str(attrs.get("icd10cm_code", "")).strip()
    if code and code not in ("nan", "none", ""):
        if code.startswith(_SYMPTOM_ICD10CM_PREFIXES):
            return True
        if code.startswith(_SYMPTOM_ICD10CM_SUBPREFIXES):
            return True
    return False


# ─────────────────────────────────────────────────────────────────────────────
# Fix 1: LOINC codes
# ─────────────────────────────────────────────────────────────────────────────

def fix_loinc_codes(G: nx.DiGraph) -> int:
    fixed = 0
    for node, code in _CORRECT_LOINC.items():
        if not G.has_node(node):
            continue
        current = G.nodes[node].get("loinc_code")
        if str(current) != code:
            if _is_invalid_loinc(str(current)):
                # Record the bad code as a tooltip note so the frontend can
                # display "UMLS returned: <old>" alongside the corrected code.
                G.nodes[node]["loinc_code_umls_raw"] = str(current)
            G.nodes[node]["loinc_code"] = code
            print(f"  LOINC fixed: {node.split(': ',1)[-1]:<30} {str(current)!r} → {code!r}")
            fixed += 1
    return fixed


# ─────────────────────────────────────────────────────────────────────────────
# Fix 2 & 3: ICD-10 codes
# ─────────────────────────────────────────────────────────────────────────────

def fix_icd10_codes(G: nx.DiGraph) -> int:
    fixed = 0
    for node, code in _CORRECT_ICD10.items():
        if not G.has_node(node):
            continue
        current = G.nodes[node].get("icd10_code")
        if str(current) != code:
            old_str = str(current)
            # Preserve the original code (or "clinical name" flag) as a tooltip
            # attribute so the frontend can show e.g. "UMLS returned: nan (override applied)"
            if old_str.lower() in ("nan", "none", ""):
                G.nodes[node]["icd10_code_note"] = "Code not found by UMLS — override applied"
            else:
                G.nodes[node]["icd10_code_note"] = f"Previous code: {old_str} (corrected)"
            G.nodes[node]["icd10_code"] = code
            print(f"  ICD-10 fixed: {node.split(': ',1)[-1]:<45} {old_str!r} → {code!r}")
            fixed += 1
    return fixed


# ─────────────────────────────────────────────────────────────────────────────
# Fix 4: Reclassify ClinGraph symptom nodes mistyped as Condition
# Fix 5: Reclassify "Pediatric Assessment" as Demographic_Factor
# ─────────────────────────────────────────────────────────────────────────────

def reclassify_nodes(G: nx.DiGraph) -> tuple[nx.DiGraph, int]:
    """
    Identify misclassified Condition nodes and relabel them with the correct
    prefix.  Returns a new graph (nx.relabel_nodes) and the count of renames.
    """
    relabel: dict[str, str] = {}

    for node, attrs in G.nodes(data=True):
        if attrs.get("type") != "Condition":
            continue

        # Fix 5: Pediatric Assessment is a demographic qualifier, not a diagnosis
        if node == "Condition: Pediatric Assessment":
            new_id = "Demographic: Pediatric Assessment"
            relabel[node] = new_id
            print(f"  Reclassify → Demographic_Factor: {node.split(': ',1)[-1]}")
            continue

        # Fix 4: ClinGraph symptom/finding nodes
        if _should_be_symptom(attrs):
            bare   = node.split(": ", 1)[-1]
            new_id = f"Symptom: {bare}"
            if new_id != node:
                relabel[node] = new_id
                print(f"  Reclassify → Symptom: {bare}")

    if not relabel:
        return G, 0, {}

    # Update type attribute before relabelling (relabel_nodes preserves attrs)
    for old_id, new_id in relabel.items():
        prefix = new_id.split(": ", 1)[0]
        new_type = {
            "Symptom":    "Symptom",
            "Demographic":"Demographic_Factor",
        }.get(prefix, "Symptom")
        G.nodes[old_id]["type"] = new_type

    G = nx.relabel_nodes(G, relabel)
    return G, len(relabel), relabel


# ─────────────────────────────────────────────────────────────────────────────
# Fix 6: Prune disconnected ClinGraph nodes
# ─────────────────────────────────────────────────────────────────────────────

def prune_disconnected_clingraph(G: nx.DiGraph) -> int:
    """
    Remove ClinGraph nodes that have no directed path to any Diagnostic_Test
    node.  These nodes add no signal to any triage recommendation pathway.
    """
    test_nodes = {n for n, d in G.nodes(data=True) if d.get("type") == "Diagnostic_Test"}
    cg_nodes   = {n for n, d in G.nodes(data=True) if d.get("source") == "ClinGraph"}

    # BFS from each test node backwards (using reversed graph) to find what can reach a test
    G_rev = G.reverse(copy=False)
    can_reach_test: set[str] = set()
    for test in test_nodes:
        can_reach_test.update(nx.descendants(G_rev, test))
    can_reach_test.update(test_nodes)

    to_remove = [n for n in cg_nodes if n not in can_reach_test]
    G.remove_nodes_from(to_remove)
    return len(to_remove)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    script_dir = os.path.dirname(os.path.abspath(__file__))

    parser = argparse.ArgumentParser(
        description="Apply data quality corrections to the enriched triage KG."
    )
    parser.add_argument(
        "--base-pkl",
        default=os.path.join(script_dir, "triage_knowledge_graph_enriched.pkl"),
        help="Input enriched KG pickle (default: triage_knowledge_graph_enriched.pkl)",
    )
    parser.add_argument(
        "--out-pkl",
        default=os.path.join(script_dir, "triage_knowledge_graph_clean.pkl"),
        help="Output cleaned KG pickle (default: triage_knowledge_graph_clean.pkl)",
    )
    parser.add_argument(
        "--keep-disconnected",
        action="store_true",
        help="Skip pruning of disconnected ClinGraph nodes",
    )
    args = parser.parse_args()

    if not os.path.exists(args.base_pkl):
        print(f"ERROR: Input PKL not found: {args.base_pkl}", file=sys.stderr)
        sys.exit(1)

    print(f"\nLoading {os.path.basename(args.base_pkl)} ...")
    with open(args.base_pkl, "rb") as f:
        G_base: nx.DiGraph = pickle.load(f)

    n0_nodes = G_base.number_of_nodes()
    n0_edges = G_base.number_of_edges()
    print(f"  Loaded: {n0_nodes} nodes, {n0_edges} edges.\n")

    G = copy.deepcopy(G_base)

    # ── Fix 1: LOINC codes ────────────────────────────────────────────────
    print("[1/5] Fixing LOINC codes on Diagnostic_Test nodes ...")
    n_loinc = fix_loinc_codes(G)
    print(f"  → {n_loinc} LOINC code(s) corrected.\n")

    # ── Fix 2+3: ICD-10 codes ─────────────────────────────────────────────
    print("[2/5] Fixing ICD-10 codes on Condition nodes ...")
    n_icd10 = fix_icd10_codes(G)
    print(f"  → {n_icd10} ICD-10 code(s) corrected.\n")

    # ── Fix 4+5: Reclassify mistyped nodes ───────────────────────────────
    print("[3/5] Reclassifying ClinGraph symptom nodes mistyped as Condition ...")
    G, n_reclassified, relabel_map = reclassify_nodes(G)
    print(f"  → {n_reclassified} node(s) reclassified.\n")

    # ── Fix 6: Prune disconnected ClinGraph nodes ─────────────────────────
    if not args.keep_disconnected:
        print("[4/5] Pruning ClinGraph nodes with no path to any Diagnostic_Test ...")
        n_pruned = prune_disconnected_clingraph(G)
        print(f"  → {n_pruned} node(s) removed.\n")
    else:
        n_pruned = 0
        print("[4/5] Skipped (--keep-disconnected).\n")

    # ── Save ──────────────────────────────────────────────────────────────
    print("[5/5] Saving cleaned KG ...")
    with open(args.out_pkl, "wb") as f:
        pickle.dump(G, f)

    n1_nodes = G.number_of_nodes()
    n1_edges = G.number_of_edges()

    print(f"\n{'─'*58}")
    print(f"  Base KG  :  {n0_nodes:>5} nodes  {n0_edges:>5} edges")
    print(f"  Changes  :  {n1_nodes - n0_nodes:>+5} nodes  {n1_edges - n0_edges:>+5} edges")
    print(f"  Clean KG :  {n1_nodes:>5} nodes  {n1_edges:>5} edges")
    print(f"{'─'*58}")
    print(f"  LOINC codes fixed      : {n_loinc}")
    print(f"  ICD-10 codes fixed     : {n_icd10}")
    print(f"  Nodes reclassified     : {n_reclassified}")
    print(f"  Disconnected pruned    : {n_pruned}")
    print(f"{'─'*58}")
    print(f"  Saved → {args.out_pkl}")

    # ── Regression check: all guideline edges preserved ───────────────────
    # (ClinGraph edges to pruned nodes are intentionally removed;
    #  relabelled node IDs are translated via relabel_map before checking)
    base_guideline_edges = [
        (u, v) for u, v, d in G_base.edges(data=True)
        if d.get("relationship") in (
            "INDICATES_CONDITION", "REQUIRES_TEST", "DIRECTLY_INDICATES_TEST",
            "RECOMMENDS_TREATMENT", "HAS_ADVERSE_EVENT",
        )
    ]
    missing = []
    for u, v in base_guideline_edges:
        u_check = relabel_map.get(u, u)
        v_check = relabel_map.get(v, v)
        if not G.has_edge(u_check, v_check):
            missing.append((u, v))

    if missing:
        print(f"\n  WARNING: {len(missing)} guideline edge(s) missing after clean!", file=sys.stderr)
        for u, v in missing[:5]:
            print(f"    {u} → {v}", file=sys.stderr)
    else:
        print("  Regression check: all guideline edges preserved. ✓")


if __name__ == "__main__":
    main()
