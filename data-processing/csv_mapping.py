import os
import json
import pickle
import pandas as pd
import networkx as nx
from azure.identity import AzureCliCredential, get_bearer_token_provider
from openai import AzureOpenAI

from dotenv import load_dotenv # <--- ADD THIS

# --- Load the .env file! --- # <--- ADD THIS BLOCK
env_path = os.path.join(os.path.dirname(__file__), ".env")
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

# ---------------------------------------------------------
# 2. Semantic Matching Function
# ---------------------------------------------------------
def get_semantic_match(raw_diagnosis: str, valid_conditions: list) -> str:
    """Uses Azure OpenAI to fuzzily match a raw diagnosis to a known graph condition."""
    
    system_prompt = (
        "You are a clinical NLP assistant. Your job is to map a raw patient diagnosis "
        "to the closest matching official condition from a provided list. "
        "If there is no medically reasonable match, return null. "
        "You must respond in strictly valid JSON format with a single key: 'matched_condition'."
    )
    
    user_prompt = (
        f"Raw Patient Diagnosis: '{raw_diagnosis}'\n"
        f"Available Official Conditions: {json.dumps(valid_conditions)}\n\n"
        "Return the closest matching Official Condition exactly as it appears in the list."
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
        
        # Parse the JSON response
        result_json = json.loads(response.choices[0].message.content)
        return result_json.get("matched_condition")
        
    except Exception as e:
        print(f"  [!] LLM mapping failed for '{raw_diagnosis}': {e}")
        return None

# ---------------------------------------------------------
# 3. Main Processing Pipeline
# ---------------------------------------------------------
def map_diagnoses_with_llm(csv_input_path: str, graph_pkl_path: str, csv_output_path: str):
    
    # --- Load Graph & Extract Valid Conditions ---
    print("Loading Knowledge Graph...")
    with open(graph_pkl_path, "rb") as f:
        kg = pickle.load(f)
        
    # Extract all nodes that start with "Condition: " and strip the prefix
    available_conditions = [
        node.replace("Condition: ", "") 
        for node in kg.nodes 
        if str(node).startswith("Condition: ")
    ]
    print(f"Found {len(available_conditions)} unique conditions in the graph.")

    # --- Load CSV & Find Unique Raw Diagnoses ---
    df = pd.read_csv(csv_input_path)
    print(list(df.columns))
    if 'dx' not in df.columns:
        raise ValueError("CSV must contain a 'diagnosis' column.")
        
    unique_raw_diagnoses = df['dx'].dropna().unique()
    print(f"Found {len(unique_raw_diagnoses)} unique raw diagnoses in the CSV to map.")

    # --- Perform LLM Mapping ---
    print("Starting Azure OpenAI semantic matching...")
    mapping_dictionary = {}
    
    for raw_diag in unique_raw_diagnoses:
        match = get_semantic_match(raw_diag, available_conditions)
        mapping_dictionary[raw_diag] = match
        print(f"  Mapped: '{raw_diag}' -> '{match}'")

    # --- Map Results Back to DataFrame ---
    # Create a new column for the matched condition
    df['matched_graph_condition'] = df['dx'].map(mapping_dictionary)

    # --- Traverse Graph for Tests ---
    print("Extracting required tests from graph...")
    def get_tests_for_condition(matched_condition):
        if not matched_condition:
            return "No match found"
            
        node_name = f"Condition: {matched_condition}"
        tests = []
        
        if node_name in kg.nodes:
            for _, target_node, edge_data in kg.out_edges(node_name, data=True):
                if edge_data.get('relationship') == 'REQUIRES_TEST':
                    tests.append(target_node.replace("Test: ", ""))
                    
        return ", ".join(tests) if tests else "No linked tests found"

    df['potential_tests'] = df['matched_graph_condition'].apply(get_tests_for_condition)

    # --- Save Output ---
    df.to_csv(csv_output_path, index=False)
    print(f"\nSuccess! Saved enriched data to '{csv_output_path}'")
    return df

# ---------------------------------------------------------
# Execute
# ---------------------------------------------------------
if __name__ == "__main__":
    # --- Configuration ---
    # Update these paths to match your local file structure
    INPUT_CSV = "/home/achua/student_data.csv" 
    OUTPUT_CSV = "patient_diagnoses_with_tests.csv"
    GRAPH_PKL = "triage_knowledge_graph.pkl" # Sourced from your builder script
    
    # Run the mapping
    result_df = map_diagnoses_with_llm(INPUT_CSV, GRAPH_PKL, OUTPUT_CSV)
    
    # Display a preview
    print("\n--- Output Preview ---")
    print(result_df[['dx', 'potential_tests']].head())