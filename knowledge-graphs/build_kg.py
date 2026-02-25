import requests
import pandas as pd
import networkx as nx
from dotenv import load_dotenv
import os
import pickle

# --- 1. Curated Ground Truth ---
guideline_rules = [
    {"raw_symptom": "chest pain", "condition": "Acute Coronary Syndrome", "test": "ECG", "source": "AHA/ACC"},
    {"raw_symptom": "syncope", "condition": "Arrhythmia", "test": "ECG", "source": "AHA/ACC"},
    {"raw_symptom": "scrotal pain", "condition": "Testicular Torsion", "test": "Testicular Ultrasound", "source": "ACR"},
    {"raw_symptom": "forearm trauma", "condition": "Forearm Fracture", "test": "Arm X-Ray", "source": "ACR"},
    {"raw_symptom": "right lower quadrant pain", "condition": "Appendicitis", "test": "Appendix Ultrasound", "source": "ACR"},
    {"raw_symptom": "diffuse abdominal pain", "condition": "Abdominal Aortic Aneurysm", "test": "Abdominal Ultrasound", "source": "ACR"},
    # Adding rules based on Vitals and Demographics
    {"raw_symptom": "heart rate > 120", "condition": "Tachycardia", "test": "ECG", "source": "AHA/ACC"},
    # We map 'pediatric' as a demographic factor indicating a specific protocol
    {"raw_symptom": "age < 18", "condition": "Pediatric Assessment", "test": "Appendix Ultrasound", "source": "ACR"},
    # --- NICE Guidelines (UK) ---
    # NICE explicitly dictates ultrasound for equivocal acute scrotal pain (checking for epididymo-orchitis vs torsion)
    {"raw_symptom": "acute scrotal pain", "condition": "Suspected Epididymo-orchitis", "test": "Testicular Ultrasound", "source": "NICE"},
    # NICE clinical pathway for appendicitis often relies on localized pain and fever
    {"raw_symptom": "right lower quadrant pain", "condition": "Appendicitis", "test": "Appendix Ultrasound", "source": "NICE"},
    {"raw_symptom": "fever > 37.3", "condition": "Appendicitis", "test": "Appendix Ultrasound", "source": "NICE"},

    # --- CTAS Modifiers (Canada) ---
    # CTAS mandates an ECG within 10 minutes for chest pain with cardiac features, heavily weighted by age
    {"raw_symptom": "atypical chest pain", "condition": "Potential Cardiac Ischemia", "test": "ECG", "source": "CTAS"},
    {"raw_symptom": "age > 35", "condition": "Potential Cardiac Ischemia", "test": "ECG", "source": "CTAS"},
    # CTAS First Order Modifier for Vitals
    {"raw_symptom": "heart rate > 120", "condition": "Tachycardia", "test": "ECG", "source": "CTAS"}
]
df_rules = pd.DataFrame(guideline_rules)

# --- 2. API Keys & Tokens ---
load_dotenv() # This loads the variables from the .env file

UMLS_API_KEY = os.getenv("UMLS_API_KEY")
INFOWAY_CLIENT_ID = os.getenv("INFOWAY_CLIENT_ID")
INFOWAY_CLIENT_SECRET = os.getenv("INFOWAY_CLIENT_SECRET")

# --- 3. The API Functions (Kept exactly as you wrote them) ---
def get_snomed_concept(term, api_key):
    """
    Queries the UMLS REST API to find the standardized SNOMED CT concept for a raw symptom.
    """
    base_uri = "https://uts-ws.nlm.nih.gov/rest/search/current"
    params = {
        "string": term,
        "sabs": "SNOMEDCT_US", # Restrict search specifically to SNOMED CT
        "returnIdType": "code",
        "apiKey": api_key
    }
    
    try:
        response = requests.get(base_uri, params=params)
        response.raise_for_status()
        data = response.json()
        
        # Extract the best matching concept name and ID
        results = data.get("result", {}).get("results", [])
        if results:
            best_match = results[0]
            return best_match["name"], best_match["ui"] # Returns (Standard Name, SNOMED Code)
        return term, None
    except Exception as e:
        print(f"API Error for term '{term}': {e}")
        return term, None

def get_infoway_access_token(client_id, client_secret):
    """
    Exchanges a Client ID and Client Secret for an OAuth2 Access Token
    from Canada Health Infoway's authorization server.
    """
    auth_url = "https://terminologystandardsservice.ca/authorisation/auth/realms/terminology/protocol/openid-connect/token"
    
    payload = {
        "grant_type": "client_credentials",
        "client_id": client_id,
        "client_secret": client_secret
    }
    
    # The payload must be sent as form data (application/x-www-form-urlencoded)
    headers = {
        "Content-Type": "application/x-www-form-urlencoded"
    }
    
    try:
        response = requests.post(auth_url, data=payload, headers=headers)
        response.raise_for_status() # Will raise an exception for 4xx/5xx errors
        token_data = response.json()
        
        return token_data.get("access_token")
        
    except Exception as e:
        print(f"Failed to obtain access token: {e}")
        return None

def get_infoway_snomed_concept(term, access_token):
    """
    Queries Canada Health Infoway's FHIR API to find a SNOMED CT CA concept.
    """
    # The official Canadian FHIR Terminology Service endpoint
    base_uri = "https://terminologystandardsservice.ca/fhir/ValueSet/$expand"
    
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Accept": "application/fhir+json"
    }
    
    # Standard FHIR search parameters filtering for the SNOMED CT CA CodeSystem
    params = {
        "url": "http://snomed.info/sct?fhir_vs", # Target SNOMED
        "filter": term,
        "count": 1
    }
    
    try:
        response = requests.get(base_uri, headers=headers, params=params)
        response.raise_for_status()
        data = response.json()
        
        # Extract the concept from the FHIR expansion payload
        contains = data.get("expansion", {}).get("contains", [])
        if contains:
            best_match = contains[0]
            # Returns (Standard Name, SNOMED Code)
            return best_match.get("display"), best_match.get("code")
        return term, None
        
    except Exception as e:
        print(f"Infoway API Error for term '{term}': {e}")
        return term, None

def get_open_medical_concept(term):
    """
    Queries the EMBL-EBI OLS API to find a standardized medical concept WITHOUT an API key.
    It targets HP (Phenotypes/Symptoms), MONDO (Diseases), and EFO (Experimental Factors).
    """
    base_uri = "https://www.ebi.ac.uk/ols4/api/search"
    params = {
        "q": term,
        # Restrict search to the best clinical open ontologies
        "ontology": "hp,mondo,efo", 
        "queryFields": "label,synonym",
        "exact": "false",
        "rows": 1
    }
    
    try:
        response = requests.get(base_uri, params=params)
        response.raise_for_status()
        data = response.json()
        
        # Extract the best matching concept
        results = data.get("response", {}).get("docs", [])
        if results:
            best_match = results[0]
            # Returns (Standard Name, Ontology ID like 'HP_0002027')
            return best_match.get("label"), best_match.get("short_form") 
        return term, None
    except Exception as e:
        print(f"API Error for term '{term}': {e}")
        return term, None

# --- 4. The Unified Extraction Pipeline ---
print("Extracting cross-mapped ontology codes...")

# 1. Get US SNOMED (UMLS) # Currently no API Access Token
# df_rules[['umls_name', 'umls_code']] = df_rules.apply(
#     lambda row: pd.Series(get_snomed_concept(row['raw_symptom'], UMLS_API_KEY)), axis=1
# )

# 2. Get Open Ontologies (EMBL-EBI)
df_rules[['ebi_name', 'ebi_code']] = df_rules.apply(
    lambda row: pd.Series(get_open_medical_concept(row['raw_symptom'])), axis=1
)

# 3. Get Canadian SNOMED (Infoway)

# 3.2. Fetch the fresh token
print("Fetching access token...")
INFOWAY_TOKEN = get_infoway_access_token(INFOWAY_CLIENT_ID, INFOWAY_CLIENT_SECRET)

if INFOWAY_TOKEN:
    df_rules[['infoway_name', 'infoway_code']] = df_rules.apply(
        lambda row: pd.Series(get_infoway_snomed_concept(row['raw_symptom'], INFOWAY_TOKEN)), axis=1
    )

# --- 5. Building the Multi-Ontology Knowledge Graph (Refined Schema) ---
def build_unified_medical_kg(df):
    G = nx.DiGraph()
    
    for _, row in df.iterrows():
        # 1. Dynamic Node Typing
        # Determine if the 'raw_symptom' is actually a vital sign or demographic
        if ">" in row['raw_symptom'] or "<" in row['raw_symptom']:
            if "age" in row['raw_symptom']:
                node_type = "Demographic_Factor"
                node_prefix = "Demographic"
            else:
                node_type = "Vital_Sign_Threshold"
                node_prefix = "Vital"
        else:
            node_type = "Symptom"
            node_prefix = "Symptom"

        primary_node = f"{node_prefix}: {row['raw_symptom'].title()}"
        condition_node = f"Condition: {row['condition']}"
        test_node = f"Test: {row['test']}"
        
        # 2. Add the primary node with ALL extracted codes as metadata
        G.add_node(primary_node, 
                   type=node_type, 
                #    snomed_us_code=row['umls_code'],
                   snomed_ca_code=row['infoway_code'],
                   ebi_open_code=row['ebi_code'])
                   
        G.add_node(condition_node, type="Condition")
        G.add_node(test_node, type="Diagnostic_Test", guideline_source=row['source'])
        
        # 3. Add relationships (Including the missing shortcut edge!)
        # Add relationships with the source attached to the edge!
        G.add_edge(primary_node, condition_node, relationship="INDICATES_CONDITION", source=row['source'])
        G.add_edge(condition_node, test_node, relationship="REQUIRES_TEST", source=row['source'])
        G.add_edge(primary_node, test_node, relationship="DIRECTLY_INDICATES_TEST", source=row['source'])
        
    return G

kg = build_unified_medical_kg(df_rules)

# Save the NetworkX graph to a file
with open('triage_knowledge_graph.pkl', 'wb') as f:
    pickle.dump(kg, f)

print("\nKnowledge Graph successfully saved to 'triage_knowledge_graph.pkl'")

# Let's inspect one of the nodes to see the merged data!
print("\n--- Node Inspection ---")
example_node = "Symptom: Chest Pain"
if example_node in kg.nodes:
    print(f"Properties for '{example_node}':")
    print(kg.nodes[example_node])