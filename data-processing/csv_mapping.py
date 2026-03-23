import argparse
import os
import json
import pickle
import pandas as pd
import networkx as nx
from azure.identity import AzureCliCredential, get_bearer_token_provider
from openai import AzureOpenAI

from dotenv import load_dotenv

# 1. Get the directory of your current script (data-processing)
current_dir = os.path.dirname(__file__)

# 2. Go UP one level (..), then DOWN into knowledge-graphs
env_path = os.path.abspath(os.path.join(current_dir, "..", "knowledge-graphs", ".env"))

# 3. Load the file
load_dotenv(env_path, override=True)
load_dotenv(env_path)
# ---------------------------------------------------------
# 1. Setup Azure OpenAI Client
# ---------------------------------------------------------
_endpoint = os.environ.get("ENDPOINT_URL")
_deployment = os.environ.get("DEPLOYMENT_NAME")

if not _endpoint or not _deployment:
    raise ValueError("ENDPOINT_URL or DEPLOYMENT_NAME missing from environment.")

_credential = AzureCliCredential()
_token_provider = get_bearer_token_provider(
    _credential, "https://cognitiveservices.azure.com/.default"
)

_CLIENT = AzureOpenAI(
    azure_endpoint=_endpoint,
    azure_ad_token_provider=_token_provider,
    api_version="2025-01-01-preview",
)

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
        "ecg_dx":         "pred_ecg",
        "xray_arm_dx":    "pred_xray_arm",
        "us_app_dx":      "pred_us_app",
        "us_testes_dx":   "pred_us_testes",
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
# 3. Semantic Matching Function
# ---------------------------------------------------------
def get_semantic_matches(raw_diagnosis: str, available_nodes: list[dict]) -> list[dict]:
    """Uses Azure OpenAI to return up to 3 high-confidence matches from the graph.

    Each match is {"label": str, "node_type": str} where label is the node label
    exactly as it appears in available_nodes.  Returns an empty list when no
    reasonable match exists.
    """
    system_prompt = (
        "You are a clinical NLP assistant mapping raw patient diagnoses to nodes in a "
        "medical knowledge graph. A node can be a Condition (named diagnosis), Symptom "
        "(clinical finding), Vital_Sign_Threshold, Risk_Factor, Demographic_Factor, "
        "Clinical_Attribute, or Mechanism_of_Injury.\n\n"
        "For each candidate node, estimate your confidence (0–100) that it is clinically "
        "relevant to the raw diagnosis. Return ALL nodes where your confidence is 85 or "
        "above. There is no cap on the number of matches — return one, many, or none "
        "depending on how many genuinely meet the threshold. "
        "If no node reaches 85% confidence, return an empty array.\n\n"
        "Respond in strictly valid JSON: "
        "{\"matches\": [{\"label\": \"...\", \"node_type\": \"...\", \"confidence\": <int 0-100>}, ...]}"
    )

    user_prompt = (
        f"Raw Patient Diagnosis: '{raw_diagnosis}'\n\n"
        f"Available Graph Nodes:\n{json.dumps(available_nodes, indent=2)}\n\n"
        "Return each label exactly as it appears in the node list above. "
        "Only include matches with confidence >= 85."
    )

    try:
        response = _CLIENT.chat.completions.create(
            model=_deployment,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_format={"type": "json_object"},
        )
        result = json.loads(response.choices[0].message.content)
        matches = result.get("matches", [])
        if not isinstance(matches, list):
            return []
        return [
            m for m in matches
            if isinstance(m, dict)
            and "label" in m
            and "node_type" in m
            and int(m.get("confidence", 0)) >= 85
        ]
    except Exception as e:
        print(f"  [!] LLM mapping failed for '{raw_diagnosis}': {e}")
        return []


# ---------------------------------------------------------
# 4. Graph Traversal Helper
# ---------------------------------------------------------
def get_tests_for_node(kg, label: str, node_type: str) -> list[str]:
    """Return all Test node labels reachable from a graph node via test edges.

    Conditions are connected via REQUIRES_TEST; all other node types are
    connected via DIRECTLY_INDICATES_TEST.
    """
    # Reconstruct the full node ID from label + type
    reverse_prefix = {v: k for k, v in _PREFIX_TO_TYPE.items()}
    prefix = reverse_prefix.get(node_type, node_type)
    node_id = f"{prefix}: {label}"

    if node_id not in kg.nodes:
        return []

    tests = []
    for _, target, edge_data in kg.out_edges(node_id, data=True):
        if edge_data.get("relationship") in _TEST_EDGES:
            tests.append(str(target).replace("Test: ", ""))
    return tests


# ---------------------------------------------------------
# 5. Main Processing Pipeline
# ---------------------------------------------------------
def map_diagnoses_with_llm(
    csv_input_path: str,
    graph_pkl_path: str,
    csv_output_path: str,
    row_limit: int | None = None,
    validate: bool = False,
):
    # --- Load Graph & Build Node Catalogue ---
    print("Loading Knowledge Graph...")
    with open(graph_pkl_path, "rb") as f:
        kg = pickle.load(f)

    # Build a catalogue of all matchable nodes: conditions + clinical finding types
    available_nodes: list[dict] = []
    for node_id, attrs in kg.nodes(data=True):
        node_str = str(node_id)
        for prefix, node_type in _PREFIX_TO_TYPE.items():
            if node_str.startswith(f"{prefix}: "):
                label = node_str[len(prefix) + 2:]
                available_nodes.append({"label": label, "node_type": node_type})
                break

    cond_count = sum(1 for n in available_nodes if n["node_type"] == "Condition")
    symp_count = sum(1 for n in available_nodes if n["node_type"] != "Condition")
    print(f"Graph catalogue: {cond_count} conditions, {symp_count} clinical finding nodes.")

    # --- Load CSV ---
    df = pd.read_csv(csv_input_path)
    print(list(df.columns))
    if 'dx' not in df.columns:
        raise ValueError("CSV must contain a 'dx' column.")

    if row_limit is not None:
        print(f"  [--limit] Restricting to first {row_limit} rows for testing.")
        df = df.head(row_limit)

    unique_raw_diagnoses = df['dx'].dropna().unique()
    print(f"Found {len(unique_raw_diagnoses)} unique raw diagnoses to map.")

    # --- Perform LLM Mapping ---
    print("Starting Azure OpenAI semantic matching...")
    # mapping_dictionary: dx → list of {"label": ..., "node_type": ..., "tests": [...]}
    mapping_dictionary: dict[str, list[dict]] = {}

    for raw_diag in unique_raw_diagnoses:
        matches = get_semantic_matches(raw_diag, available_nodes)
        enriched = []
        for m in matches:
            tests = get_tests_for_node(kg, m["label"], m["node_type"])
            enriched.append({**m, "tests": tests})
        mapping_dictionary[raw_diag] = enriched
        labels = " | ".join(
            f"{m['label']} ({m['node_type']}, {m.get('confidence', '?')}%)"
            for m in enriched
        ) or "no match"
        print(f"  '{raw_diag}' -> {labels}")

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

    # --- Validation ---
    if validate:
        missing = [col for col in TEST_COLUMN_MAP.values() if col not in df.columns]
        if missing:
            print(f"\n[!] --validate: ground-truth column(s) not found in CSV: {missing}")
            print("    Skipping validation. Check that the CSV contains the expected columns.")
        else:
            metrics_df = _run_validation(df)
            validation_path = csv_output_path.replace(".csv", "_validation.csv")
            metrics_df.to_csv(validation_path, index=False)
            print(f"\nValidation report saved to '{validation_path}'")
            # Re-save enriched CSV with pred_* columns included
            df.to_csv(csv_output_path, index=False)

    return df

# ---------------------------------------------------------
# Execute
# ---------------------------------------------------------
if __name__ == "__main__":
    _KG_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "knowledge-graphs"))

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
        default=os.path.join(_KG_DIR, "triage_knowledge_graph.pkl"),
        help="Path to triage_knowledge_graph.pkl.",
    )
    parser.add_argument(
        "--limit", type=int, default=None, metavar="N",
        help="Process only the first N rows of the CSV. Useful for testing before a full run.",
    )
    parser.add_argument(
        "--validate", action="store_true",
        help=(
            "After mapping, compare predicted tests against binary ground-truth columns "
            "(ecg_dx, xray_arm_dx, us_app_dx, us_testes_dx) and print per-test metrics. "
            "Writes a <output>_validation.csv alongside the enriched CSV."
        ),
    )
    args = parser.parse_args()

    result_df = map_diagnoses_with_llm(
        args.input_csv,
        args.graph_pkl,
        args.output_csv,
        row_limit=args.limit,
        validate=args.validate,
    )

    print("\n--- Output Preview ---")
    print(result_df[['dx', 'matched_graph_nodes', 'potential_tests']].head())