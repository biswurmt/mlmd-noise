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

# ---------------------------------------------------------
# 2. Semantic Matching Function
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
# 3. Graph Traversal Helper
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
# 4. Main Processing Pipeline
# ---------------------------------------------------------
def map_diagnoses_with_llm(
    csv_input_path: str,
    graph_pkl_path: str,
    csv_output_path: str,
    row_limit: int | None = None,
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
    args = parser.parse_args()

    result_df = map_diagnoses_with_llm(
        args.input_csv,
        args.graph_pkl,
        args.output_csv,
        row_limit=args.limit,
    )

    print("\n--- Output Preview ---")
    print(result_df[['dx', 'matched_graph_nodes', 'potential_tests']].head())