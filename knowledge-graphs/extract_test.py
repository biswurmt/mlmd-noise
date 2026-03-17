import os
import json
import pandas as pd
from azure.identity import AzureCliCredential, get_bearer_token_provider
from openai import AzureOpenAI
from dotenv import load_dotenv

# --- Load the .env file! --- #
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
# 2. Define the Prompt & LLM Call
# ---------------------------------------------------------
def get_diagnostic_tests_for_batch(diagnoses_list):
    """
    Prompts the LLM to identify and consolidate diagnostic tests for a list of diagnoses.
    """
    
    system_prompt = """
    You are an expert medical informatics AI. I will provide a list of medical diagnoses. 
    Your task is to identify the standard diagnostic tests (imaging, labs, procedures) used to evaluate or confirm each diagnosis.
    
    CRITICAL INSTRUCTION - CONSOLIDATION:
    You must consolidate and standardize the names of the diagnostic tests as much as possible to be used as nodes in a knowledge graph. 
    - E.g., Use "Appendix Ultrasound" for all variations like "Ultrasound of appendix", "US Appendix", etc.
    - E.g., Use "Complete Blood Count (CBC)" universally.
    - Keep test names broad enough to group related conditions, but specific enough to be clinically useful.
    
    Respond STRICTLY in valid JSON format with the following structure:
    {
      "results": [
        {
          "diagnosis": "exact diagnosis name provided",
          "diagnostic_tests": ["Standardized Test Name 1", "Standardized Test Name 2"]
        }
      ]
    }
    """

    user_prompt = f"Here is the list of diagnoses. Please process them:\n{json.dumps(diagnoses_list)}"

    try:
        response = _CLIENT.chat.completions.create(
            model=_deployment,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            response_format={"type": "json_object"}, # Forces JSON output
        )
        
        # Parse the JSON response string back into a Python dictionary
        return json.loads(response.choices[0].message.content)
        
    except Exception as e:
        print(f"Error calling Azure OpenAI: {e}")
        return None

# ---------------------------------------------------------
# 3. Process the CSV
# ---------------------------------------------------------
def process_diagnoses_csv(csv_path, output_path, batch_size=20):
    # Read the CSV
    df = pd.read_csv(csv_path)
    
    # Ensure the column exists (adjust 'diagnosis' to your exact column name if different)
    if 'dx' not in df.columns:
        raise ValueError("CSV must contain a 'diagnosis' column.")
        
    unique_diagnoses = df['dx'].dropna().unique().tolist()
    print(f"Found {len(unique_diagnoses)} unique diagnoses. Processing in batches of {batch_size}...")

    all_results = []

    # Process in batches to ensure the LLM has context for standardization, 
    # but doesn't hit context limits or hallucinate.
    for i in range(0, len(unique_diagnoses), batch_size):
        batch = unique_diagnoses[i:i + batch_size]
        print(f"Processing batch {i//batch_size + 1}...")
        
        batch_result = get_diagnostic_tests_for_batch(batch)
        if batch_result and "results" in batch_result:
            all_results.extend(batch_result["results"])

    # Save to a new JSON file
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=4)
        
    print(f"Finished! Results saved to {output_path}")

# ---------------------------------------------------------
# 4. Execution
# ---------------------------------------------------------
if __name__ == "__main__":
    # Replace with your actual file paths
    INPUT_CSV = "/home/achua/student_data.csv" 
    OUTPUT_JSON = "diagnoses_to_tests_graph_data.json"
    
    process_diagnoses_csv(INPUT_CSV, OUTPUT_JSON, 1000)