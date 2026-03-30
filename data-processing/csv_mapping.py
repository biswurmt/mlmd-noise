import argparse
import os
import pickle
import pandas as pd
import networkx as nx

from dotenv import load_dotenv

# 1. Get the directory of your current script (data-processing)
current_dir = os.path.dirname(__file__)

# 2. Go UP one level (..), then DOWN into knowledge-graphs
env_path = os.path.abspath(os.path.join(current_dir, "..", "knowledge-graphs", ".env"))

# 3. Load the file
load_dotenv(env_path, override=True)
load_dotenv(env_path)

# Node-type prefix mapping (mirrors build_kg.py _NODE_PREFIX)
_PREFIX_TO_TYPE = {
    "Condition":   "Condition",
    "Symptom":     "Symptom",
    "Vital":       "Vital_Sign_Threshold",
    "Demographic": "Demographic_Factor",
    "Risk Factor": "Risk_Factor",
    "Attribute":   "Clinical_Attribute",
    "MOI":         "Mechanism_of_Injury",
}

# Edges that lead directly to a Test node
_TEST_EDGES = {"REQUIRES_TEST", "DIRECTLY_INDICATES_TEST"}


# ---------------------------------------------------------
# Graph-first entity matching (no LLM cost)
# ---------------------------------------------------------
def get_graph_matches(raw_dx: str, kg) -> list[dict]:
    """Return KG nodes whose label substring-matches raw_dx (or vice-versa).

    Case-insensitive. Returns a list of {"label", "node_type", "confidence"}
    dicts (confidence=100 for exact substring hits). Empty list if no match.
    """
    needle = raw_dx.strip().lower()
    matches: list[dict] = []
    seen: set[str] = set()
    for node_id in kg.nodes:
        node_str = str(node_id)
        for prefix, node_type in _PREFIX_TO_TYPE.items():
            if node_str.startswith(f"{prefix}: "):
                label = node_str[len(prefix) + 2:]
                label_lower = label.lower()
                if needle in label_lower or label_lower in needle:
                    key = f"{node_type}:{label}"
                    if key not in seen:
                        seen.add(key)
                        matches.append({"label": label, "node_type": node_type, "confidence": 100})
                break
    return matches

# Maps KG test-node label → CSV binary ground-truth column name
TEST_COLUMN_MAP: dict[str, str] = {
    "ECG":                   "ecg_dx",
    "Arm X-Ray":             "xray_arm_dx",
    "Appendix Ultrasound":   "us_app_dx",
    "Testicular Ultrasound": "us_testes_dx",
}

# ---------------------------------------------------------
# 2. Validation Helpers
# ---------------------------------------------------------

def _parse_predicted_tests(val) -> set[str]:
    """Return the set of test names from a comma-separated potential_tests cell."""
    if not val or (isinstance(val, float)):
        return set()
    s = str(val).strip()
    if s == "No linked tests found" or not s:
        return set()
    return {t.strip() for t in s.split(",")}


def _run_validation(df: pd.DataFrame) -> pd.DataFrame:
    """Compare potential_tests predictions to binary ground-truth columns.

    Adds pred_* binary columns to df in-place, prints a metrics table, and
    returns a DataFrame of per-test metrics.
    """
    # Build binary prediction columns
    pred_col_map: dict[str, str] = {
        "ecg_dx":         "ecg_kg",
        "xray_arm_dx":    "xray_arm_kg",
        "us_app_dx":      "us_app_kg",
        "us_testes_dx":   "us_testes_kg",
    }
    # Reverse: gt_col → test name(s)
    gt_to_tests: dict[str, list[str]] = {}
    for test_name, gt_col in TEST_COLUMN_MAP.items():
        gt_to_tests.setdefault(gt_col, []).append(test_name)

    predicted_sets = df["potential_tests"].apply(_parse_predicted_tests)

    for gt_col, test_names in gt_to_tests.items():
        pred_col = pred_col_map[gt_col]
        df[pred_col] = predicted_sets.apply(
            lambda s, names=test_names: int(bool(s.intersection(names)))
        )

    # Compute per-test metrics
    rows = []
    header = (
        f"\n{'Test':<25} {'GT Column':<18} {'TP':>4} {'FP':>4} {'TN':>5} {'FN':>4}"
        f"  {'Prec':>6} {'Rec':>6} {'F1':>6} {'Acc':>6}"
    )
    separator = "-" * len(header)
    print("\n=== Validation Report ===")
    print(header)
    print(separator)

    total = len(df)
    for gt_col, test_names in gt_to_tests.items():
        pred_col = pred_col_map[gt_col]
        gt = df[gt_col].astype(int)
        pred = df[pred_col].astype(int)

        tp = int(((pred == 1) & (gt == 1)).sum())
        fp = int(((pred == 1) & (gt == 0)).sum())
        tn = int(((pred == 0) & (gt == 0)).sum())
        fn = int(((pred == 0) & (gt == 1)).sum())

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1        = (2 * precision * recall / (precision + recall)
                     if (precision + recall) > 0 else 0.0)
        accuracy  = (tp + tn) / total if total > 0 else 0.0

        label = " / ".join(test_names)
        print(
            f"{label:<25} {gt_col:<18} {tp:>4} {fp:>4} {tn:>5} {fn:>4}"
            f"  {precision:>6.3f} {recall:>6.3f} {f1:>6.3f} {accuracy:>6.3f}"
        )
        rows.append({
            "test_name":  label,
            "gt_column":  gt_col,
            "tp": tp, "fp": fp, "tn": tn, "fn": fn,
            "precision":  round(precision, 4),
            "recall":     round(recall, 4),
            "f1":         round(f1, 4),
            "accuracy":   round(accuracy, 4),
        })

    print(separator)
    return pd.DataFrame(rows)


# ---------------------------------------------------------
# 3. Graph Traversal Helper
# ---------------------------------------------------------
def get_tests_for_node(kg, label: str, node_type: str) -> list[str]:
    """Return all Test node labels reachable from a graph node via test edges.

    Follows two paths:
      1-hop:  node  --[DIRECTLY_INDICATES_TEST | REQUIRES_TEST]--> Test
      2-hop:  node  --[INDICATES_CONDITION]--> Condition --[REQUIRES_TEST]--> Test
    """
    reverse_prefix = {v: k for k, v in _PREFIX_TO_TYPE.items()}
    prefix = reverse_prefix.get(node_type, node_type)
    node_id = f"{prefix}: {label}"

    if node_id not in kg.nodes:
        return []

    seen: set[str] = set()
    tests: list[str] = []

    for _, target, edge_data in kg.out_edges(node_id, data=True):
        rel = edge_data.get("relationship")
        if rel in _TEST_EDGES:
            # 1-hop direct test edge
            test_label = str(target).replace("Test: ", "")
            if test_label not in seen:
                seen.add(test_label)
                tests.append(test_label)
        elif rel == "INDICATES_CONDITION":
            # 2-hop: follow condition → test
            condition_node = target
            for _, test_node, cond_edge in kg.out_edges(condition_node, data=True):
                if cond_edge.get("relationship") == "REQUIRES_TEST":
                    test_label = str(test_node).replace("Test: ", "")
                    if test_label not in seen:
                        seen.add(test_label)
                        tests.append(test_label)

    return tests


# ---------------------------------------------------------
# 5. Main Processing Pipeline
# ---------------------------------------------------------
def map_diagnoses_with_llm(
    csv_input_path: str,
    graph_pkl_path: str,
    csv_output_path: str,
    row_limit: int | None = None,
    resume: bool = False,
):
    # --- Load Graph & Build Node Catalogue ---
    print("Loading Knowledge Graph...")
    with open(graph_pkl_path, "rb") as f:
        kg = pickle.load(f)

    print(f"Graph loaded: {kg.number_of_nodes()} nodes, {kg.number_of_edges()} edges.")

    # --- Load CSV ---
    df = pd.read_csv(csv_input_path)
    print(list(df.columns))
    if 'dx' not in df.columns:
        raise ValueError("CSV must contain a 'dx' column.")

    if row_limit is not None:
        print(f"  [--limit] Restricting to first {row_limit} rows for testing.")
        df = df.head(row_limit)

    # --- Resume: seed mapping_dictionary from existing output CSV ---
    already_mapped: set[str] = set()
    mapping_dictionary: dict[str, list[dict]] = {}

    if resume and os.path.exists(csv_output_path):
        existing_df = pd.read_csv(csv_output_path)
        if 'matched_graph_nodes' in existing_df.columns and 'potential_tests' in existing_df.columns:
            for _, row in existing_df.dropna(subset=['matched_graph_nodes']).iterrows():
                dx_val = row['dx']
                if pd.isna(dx_val) or str(row.get('matched_graph_nodes', '')).strip() == '':
                    continue
                # Reconstruct a lightweight enriched list sufficient for formatting
                nodes_str = str(row['matched_graph_nodes'])
                tests_str = str(row['potential_tests']) if not pd.isna(row.get('potential_tests')) else ''
                tests_list = [t.strip() for t in tests_str.split(',') if t.strip() and t.strip() != 'No linked tests found']
                enriched_resume = []
                for part in nodes_str.split(' | '):
                    part = part.strip()
                    if '(' in part and part.endswith(')'):
                        lbl = part[:part.rfind('(')].strip()
                        ntype = part[part.rfind('(') + 1:-1].strip()
                        enriched_resume.append({'label': lbl, 'node_type': ntype, 'tests': tests_list, 'confidence': '?'})
                mapping_dictionary[dx_val] = enriched_resume
                already_mapped.add(dx_val)
            print(f"  [resume] Loaded {len(already_mapped)} already-mapped diagnoses from '{csv_output_path}'.")
        else:
            print(f"  [resume] Output CSV found but missing expected columns — starting fresh.")

    unique_raw_diagnoses = df['dx'].dropna().unique()
    remaining = [d for d in unique_raw_diagnoses if d not in already_mapped]
    print(f"Found {len(unique_raw_diagnoses)} unique raw diagnoses ({len(already_mapped)} already mapped, {len(remaining)} to process).")

    # --- Perform Matching ---
    print("Starting graph-only substring matching...")
    graph_matched = 0
    graph_unmatched = 0

    for raw_diag in remaining:
        # Split on ';' to handle compound dx values (e.g. "chest pain; diaphoresis")
        terms = [t.strip() for t in str(raw_diag).split(";") if t.strip()]

        seen_labels: set[str] = set()
        enriched: list[dict] = []

        for term in terms:
            term_matches = get_graph_matches(term, kg)
            if term_matches:
                graph_matched += 1
            else:
                graph_unmatched += 1

            for m in term_matches:
                key = f"{m['node_type']}:{m['label']}"
                if key not in seen_labels:
                    seen_labels.add(key)
                    tests = get_tests_for_node(kg, m["label"], m["node_type"])
                    enriched.append({**m, "tests": tests})

            labels = " | ".join(
                f"{m['label']} ({m['node_type']})"
                for m in term_matches
            ) or "no match"
            print(f"  [graph] '{term}' -> {labels}")

        mapping_dictionary[raw_diag] = enriched

    total_terms = graph_matched + graph_unmatched
    print(f"\n--- Matching Summary ---")
    print(f"  Matched   : {graph_matched} / {total_terms} terms ({graph_matched/total_terms*100:.1f}%)" if total_terms else "  Matched   : 0")
    print(f"  Unmatched : {graph_unmatched} / {total_terms} terms ({graph_unmatched/total_terms*100:.1f}%)" if total_terms else "  Unmatched : 0")

    # --- Map Results Back to DataFrame ---
    def format_matched_nodes(matches: list[dict]) -> str:
        if not matches:
            return ""
        return " | ".join(f"{m['label']} ({m['node_type']})" for m in matches)

    def format_potential_tests(matches: list[dict]) -> str:
        seen: set[str] = set()
        tests: list[str] = []
        for m in matches:
            for t in m.get("tests", []):
                if t not in seen:
                    seen.add(t)
                    tests.append(t)
        return ", ".join(tests) if tests else "No linked tests found"

    df['matched_graph_nodes'] = df['dx'].map(
        lambda dx: format_matched_nodes(mapping_dictionary.get(dx, []))
    )
    df['potential_tests'] = df['dx'].map(
        lambda dx: format_potential_tests(mapping_dictionary.get(dx, []))
    )

    # --- Save Output ---
    df.to_csv(csv_output_path, index=False)
    print(f"\nSuccess! Saved enriched data to '{csv_output_path}'")

    # --- Validation (always runs when ground-truth columns are present) ---
    missing = [col for col in TEST_COLUMN_MAP.values() if col not in df.columns]
    if missing:
        print(f"\n[!] Validation skipped: ground-truth column(s) not found in CSV: {missing}")
    else:
        metrics_df = _run_validation(df)
        validation_path = csv_output_path.replace(".csv", "_validation.csv")
        metrics_df.to_csv(validation_path, index=False)
        print(f"\nValidation report saved to '{validation_path}'")
        # Re-save enriched CSV with *_kg columns included
        df.to_csv(csv_output_path, index=False)

    return df

# ---------------------------------------------------------
# Execute
# ---------------------------------------------------------
if __name__ == "__main__":
    _KG_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "knowledge-graphs"))
    _KG_PKL_FILE = os.environ.get("KG_PKL_FILE", "triage_knowledge_graph_enriched.pkl")
    _DEFAULT_PKL = os.path.join(_KG_DIR, _KG_PKL_FILE)

    parser = argparse.ArgumentParser(
        description="Semantically map patient diagnoses (dx column) to knowledge-graph conditions."
    )
    parser.add_argument(
        "--input-csv", required=True,
        help="Path to the input CSV file containing a 'dx' column.",
    )
    parser.add_argument(
        "--output-csv", default="patient_diagnoses_with_tests.csv",
        help="Path for the enriched output CSV (default: patient_diagnoses_with_tests.csv).",
    )
    parser.add_argument(
        "--graph-pkl",
        default=_DEFAULT_PKL,
        help=f"Path to the knowledge graph PKL file (default: {_KG_PKL_FILE} from KG_PKL_FILE env).",
    )
    parser.add_argument(
        "--limit", type=int, default=None, metavar="N",
        help="Process only the first N rows of the CSV. Useful for testing before a full run.",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help=(
            "Resume from an existing output CSV. Diagnoses already present in the output "
            "are skipped; only new ones are processed."
        ),
    )
    args = parser.parse_args()

    result_df = map_diagnoses_with_llm(
        args.input_csv,
        args.graph_pkl,
        args.output_csv,
        row_limit=args.limit,
        resume=args.resume,
    )

    print("\n--- Output Preview ---")
    print(result_df[['dx', 'matched_graph_nodes', 'potential_tests']].head())