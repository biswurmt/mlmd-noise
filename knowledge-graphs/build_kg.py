import json
import requests
import pandas as pd
import networkx as nx
from dotenv import load_dotenv
import os
import pickle

# Load environment variables from the .env file co-located with this script.
# Using an explicit path ensures the right file is found even when this module
# is imported by an external process (e.g. the Diagnotix backend server).
load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

UMLS_API_KEY          = os.getenv("UMLS_API_KEY")
INFOWAY_CLIENT_ID     = os.getenv("INFOWAY_CLIENT_ID")
INFOWAY_CLIENT_SECRET = os.getenv("INFOWAY_CLIENT_SECRET")


# --- Guideline URL Grounding ---
# Maps (source, test) → official guideline URL so every edge and Test node
# carries a citable reference back to the primary source document.
GUIDELINE_URLS = {
    # ECG pathways
    ("AHA/ACC", "ECG"): (
        "https://www.ahajournals.org/doi/10.1161/CIR.0000000000001029"
        # 2021 AHA/ACC/ASE/CHEST/SAEM/SCCT/SCMR Guideline for the Evaluation
        # and Diagnosis of Chest Pain — Circulation
    ),
    ("CTAS",    "ECG"): (
        "https://www.cambridge.org/core/journals/canadian-journal-of-emergency-medicine"
        "/article/revisions-to-the-canadian-emergency-department-triage-and-acuity-scale"
        "-ctas-guidelines-2016/E2CB3E2063C54E11259313FA4FEAE495"
        # Revisions to the CTAS Guidelines 2016 — CJEM
    ),

    # Testicular Ultrasound pathways
    ("ACR",  "Testicular Ultrasound"): (
        "https://acsearch.acr.org/docs/69363/Narrative"
        # ACR AC® Acute Onset of Scrotal Pain — Without Trauma, Without
        # Antecedent Mass (2024 update)
    ),
    ("NICE", "Testicular Ultrasound"): (
        "https://cks.nice.org.uk/topics/scrotal-pain-swelling/"
        # NICE CKS: Scrotal Pain and Swelling (last revised Aug 2024)
    ),

    # Arm X-Ray pathways
    ("ACR",  "Arm X-Ray"): (
        "https://acsearch.acr.org/docs/69418/narrative/"
        # ACR AC® Acute Hand and Wrist Trauma
    ),

    # Appendix / Abdominal Ultrasound pathways
    ("ACR",  "Appendix Ultrasound"): (
        "https://acsearch.acr.org/docs/69357/narrative/"
        # ACR AC® Right Lower Quadrant Pain — Suspected Appendicitis (2022)
    ),
    ("NICE", "Appendix Ultrasound"): (
        "https://cks.nice.org.uk/topics/appendicitis"
        # NICE CKS: Appendicitis (last revised May 2025)
    ),
    ("ACR",  "Abdominal Ultrasound"): (
        "https://acsearch.acr.org/docs/69357/narrative/"
        # ACR AC® Right Lower Quadrant Pain — closest applicable criteria
    ),
}

# In-memory caches for evidence-weight APIs — keyed by normalised term tuples
# so repeated calls with identical arguments are served from memory across
# multiple generate_knowledge_graph() invocations in the same process.
_lit_cache:   dict = {}
_trial_cache: dict = {}

# Node-type prefix mapping (used as the node ID prefix in the graph)
_NODE_PREFIX = {
    "Symptom":              "Symptom",
    "Vital_Sign_Threshold": "Vital",
    "Demographic_Factor":   "Demographic",
    "Risk_Factor":          "Risk Factor",
    "Clinical_Attribute":   "Attribute",
    "Mechanism_of_Injury":  "MOI",
}


# ─────────────────────────────────────────────────────────────────────────────
# API Helper Functions
# ─────────────────────────────────────────────────────────────────────────────

def get_snomed_concept(term, api_key):
    """Query the UMLS REST API for a SNOMED CT US concept code."""
    base_uri = "https://uts-ws.nlm.nih.gov/rest/search/current"
    params = {
        "string": term,
        "sabs": "SNOMEDCT_US",
        "returnIdType": "code",
        "apiKey": api_key,
    }
    try:
        response = requests.get(base_uri, params=params)
        response.raise_for_status()
        results = response.json().get("result", {}).get("results", [])
        if results:
            best = results[0]
            return best["name"], best["ui"]
        return term, None
    except Exception as e:
        print(f"  UMLS API error for '{term}': {e}")
        return term, None


def get_infoway_access_token(client_id, client_secret):
    """Exchange Infoway client credentials for an OAuth2 access token."""
    auth_url = (
        "https://terminologystandardsservice.ca/authorisation/auth"
        "/realms/terminology/protocol/openid-connect/token"
    )
    payload = {
        "grant_type": "client_credentials",
        "client_id": client_id,
        "client_secret": client_secret,
    }
    headers = {"Content-Type": "application/x-www-form-urlencoded"}
    try:
        response = requests.post(auth_url, data=payload, headers=headers)
        response.raise_for_status()
        return response.json().get("access_token")
    except Exception as e:
        print(f"  Infoway token error: {e}")
        return None


def get_infoway_snomed_concept(term, access_token):
    """Query Canada Health Infoway FHIR API for a SNOMED CT CA concept."""
    base_uri = "https://terminologystandardsservice.ca/fhir/ValueSet/$expand"
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Accept": "application/fhir+json",
    }
    params = {
        "url": "http://snomed.info/sct?fhir_vs",
        "filter": term,
        "count": 1,
    }
    try:
        response = requests.get(base_uri, headers=headers, params=params)
        response.raise_for_status()
        contains = response.json().get("expansion", {}).get("contains", [])
        if contains:
            best = contains[0]
            return best.get("display"), best.get("code")
        return term, None
    except Exception as e:
        print(f"  Infoway API error for '{term}': {e}")
        return term, None


def get_open_medical_concept(term):
    """Query EMBL-EBI OLS4 API for an open ontology concept (HP/MONDO/EFO). No key required.

    Returns (label, short_form, synonyms) where synonyms is a (possibly empty) list
    of alternative clinical terms for the concept.
    """
    base_uri = "https://www.ebi.ac.uk/ols4/api/search"
    _VALID_OLS_PREFIXES = ("HP_", "MONDO_", "EFO_")
    params = {
        "q": term,
        "ontology": "hp,mondo,efo",
        "queryFields": "label,synonym",
        "exact": "false",
        "rows": 5,
    }
    try:
        response = requests.get(base_uri, params=params)
        response.raise_for_status()
        docs = response.json().get("response", {}).get("docs", [])
        for doc in docs:
            if doc.get("short_form", "").startswith(_VALID_OLS_PREFIXES):
                synonyms = doc.get("synonym") or []
                return doc.get("label"), doc.get("short_form"), synonyms
        return term, None, []
    except Exception as e:
        print(f"  EBI OLS API error for '{term}': {e}")
        return term, None, []


# def get_umls_concept(term, sabs, api_key):
#     """Query the UMLS REST API for any vocabulary supported by the sabs parameter.

#     Examples:
#         sabs="LNC"      → LOINC codes for diagnostic tests
#         sabs="ICD10CM"  → ICD-10-CM codes for conditions
#     Returns (canonical_name, code) or (term, None) on failure.
#     """
#     base_uri = "https://uts-ws.nlm.nih.gov/rest/search/current"
#     params = {
#         "string": term,
#         "sabs": sabs,
#         "returnIdType": "code",
#         "apiKey": api_key,
#     }
#     try:
#         response = requests.get(base_uri, params=params)
#         response.raise_for_status()
#         results = response.json().get("result", {}).get("results", [])
#         if results:
#             best = results[0]
#             return best["name"], best["ui"]
#         return term, None
#     except Exception as e:
#         print(f"  UMLS ({sabs}) error for '{term}': {e}")
#         return term, None

def get_umls_concept(term, sabs, api_key):
    """
    A 2-step UMLS lookup that completely isolates the text search from 
    the vocabulary filter to avoid NLM 500 Server Errors.
    """
    search_url = "https://uts-ws.nlm.nih.gov/rest/search/current"
    
    # STEP 1: Get the CUI using a specific search type
    search_params = {
        "string": term,
        "searchType": "normalizedString",
        "apiKey": api_key
    }
    
    try:
        search_resp = requests.get(search_url, params=search_params)
        search_resp.raise_for_status()
        search_results = search_resp.json().get("result", {}).get("results", [])
        
        if not search_results:
            return term, None
            
        cui = search_results[0]["ui"]
        best_name = search_results[0]["name"]
        
        # STEP 2: Now that we safely have the CUI, ask for the specific vocabulary code
        atoms_url = f"https://uts-ws.nlm.nih.gov/rest/content/current/CUI/{cui}/atoms"
        atoms_params = {
            "sabs": sabs, 
            "apiKey": api_key
        }
        
        atoms_resp = requests.get(atoms_url, params=atoms_params)
        
        # Handle the potential 404 error gracefully
        try:
            atoms_resp.raise_for_status()
        except requests.exceptions.HTTPError as e:
            if atoms_resp.status_code == 404:
                # 404 means the CUI exists, but no atoms match the 'sabs' filter
                return best_name, None
            else:
                # If it's a 401 (Unauthorized) or 500 (Server Error), we still want it to crash/log
                print(f"API Error: {e}")
                return best_name, None
                
        # If no error, proceed as normal
        atoms_results = atoms_resp.json().get("result", [])
        
        if atoms_results:
            # "code" is the source vocabulary code (e.g. "I21.9" for ICD-10-CM,
            # "11524-6" for LOINC). "ui" is the UMLS-internal AUI and must NOT
            # be used here — it looks like "A17786738" and is not a real code.
            source_code = atoms_results[0].get("code") or atoms_results[0].get("ui")
            return best_name, source_code
            
        return best_name, None

    except requests.exceptions.RequestException as e:
        print(f"  UMLS API error for '{term}': {e}")
        
        # Check if the error contains a response from the server
        if hasattr(e, 'response') and e.response is not None:
            print(f"  --- Detailed Server Response ---")
            # Print the raw text sent back by the server
            print(f"  {e.response.text}")
            print(f"  --------------------------------")
            
        return term, None


def get_rxnorm_rxcui(term):
    """Query the free NLM RxNorm REST API for an RxCUI given a drug name.

    Returns the RxCUI string, or None if not found.
    """
    url = f"https://rxnav.nlm.nih.gov/REST/rxcui.json?name={requests.utils.quote(term)}"
    try:
        response = requests.get(url)
        response.raise_for_status()
        id_group = response.json().get("idGroup", {})
        rxcuis = id_group.get("rxnormId", [])
        return rxcuis[0] if rxcuis else None
    except Exception as e:
        print(f"  RxNorm API error for '{term}': {e}")
        return None


def get_openfda_adverse_event(term):
    """Query the OpenFDA drug adverse event API for the most-reported reaction
    to a given medication.

    Returns the reaction MedDRA term string, or None if not found.
    """
    url = "https://api.fda.gov/drug/event.json"
    params = {
        "search": f'patient.drug.medicinalproduct:"{term}"',
        "count": "patient.reaction.reactionmeddrapt.exact",
        "limit": 1,
    }
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        results = response.json().get("results", [])
        if results:
            return results[0].get("term")
        return None
    except Exception as e:
        print(f"  OpenFDA API error for '{term}': {e}")
        return None


def get_literature_cooccurrence(term1, term2):
    """Query Europe PMC for the number of articles mentioning both term1 and term2.

    Uses the Europe PMC REST search API (no key required).
    Returns the hitCount integer, or 0 on failure / no results.
    """
    key = (term1.lower(), term2.lower())
    if key in _lit_cache:
        return _lit_cache[key]

    params = {
        "query":    f'({term1}) AND ({term2})',
        "format":   "json",
        "pageSize": 1,
    }
    try:
        response = requests.get(
            "https://www.ebi.ac.uk/europepmc/webservices/rest/search",
            params=params,
        )
        response.raise_for_status()
        hit_count = response.json().get("hitCount", 0)
    except Exception as e:
        print(f"  Europe PMC error for ('{term1}', '{term2}'): {e}")
        hit_count = 0

    _lit_cache[key] = hit_count
    return hit_count


def get_clinical_trials_count(condition, treatment):
    """Query ClinicalTrials.gov (v2 API) for the number of studies matching
    a condition and an intervention/treatment.

    No API key required.
    Returns the totalCount integer, or 0 on failure / no results.
    """
    key = (condition.lower(), treatment.lower())
    if key in _trial_cache:
        return _trial_cache[key]

    params = {
        "query.cond": condition,
        "query.intr": treatment,
        "countTotal": "true",
        "pageSize":   1,
    }
    try:
        response = requests.get(
            "https://clinicaltrials.gov/api/v2/studies",
            params=params,
        )
        response.raise_for_status()
        total_count = response.json().get("totalCount", 0)
    except Exception as e:
        print(f"  ClinicalTrials.gov error for ('{condition}', '{treatment}'): {e}")
        total_count = 0

    _trial_cache[key] = total_count
    return total_count


# ─────────────────────────────────────────────────────────────────────────────
# Graph Builder Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _resolve_node_type(row, df):
    """Return (node_type, node_prefix) using the explicit node_type column when present,
    falling back to the original heuristic for legacy rows."""
    if "node_type" in df.columns and pd.notna(row.get("node_type")):
        nt = row["node_type"]
        return nt, _NODE_PREFIX.get(nt, "Node")

    # Legacy heuristic: infer type from raw_symptom string
    raw = row["raw_symptom"]
    if ">" in raw or "<" in raw:
        if "age" in raw:
            return "Demographic_Factor", "Demographic"
        return "Vital_Sign_Threshold", "Vital"
    return "Symptom", "Symptom"


def build_unified_medical_kg(df):
    """Build a NetworkX DiGraph from a fully-annotated rules DataFrame."""
    G = nx.DiGraph()

    for _, row in df.iterrows():
        node_type, node_prefix = _resolve_node_type(row, df)

        primary_node   = f"{node_prefix}: {row['raw_symptom'].title()}"
        condition_node = f"Condition: {row['condition']}"
        test_node      = f"Test: {row['test']}"

        # Build node attribute dict; include ontology codes when available
        node_attrs = {"type": node_type}
        for col, attr in [("ebi_code", "ebi_open_code"), ("infoway_code", "snomed_ca_code")]:
            if col in df.columns:
                node_attrs[attr] = row[col]

        # Synonyms: stored as a list; normalise NaN (rows without a treatment
        # or failed OLS lookups) to an empty list so the attribute is always iterable.
        raw_synonyms  = row.get("synonyms") if "synonyms" in df.columns else None
        node_synonyms = raw_synonyms if isinstance(raw_synonyms, list) else []

        guideline_url          = row["url"]                          if "url"                     in df.columns else None
        loinc_code             = row.get("loinc_code")               if "loinc_code"              in df.columns else None
        icd10_code             = row.get("icd10_code")               if "icd10_code"              in df.columns else None
        literature_weight      = row.get("literature_weight")        if "literature_weight"       in df.columns else None
        test_literature_weight = row.get("test_literature_weight")   if "test_literature_weight"  in df.columns else None

        G.add_node(primary_node, **node_attrs, synonyms=node_synonyms)
        G.add_node(condition_node, type="Condition", icd10_code=icd10_code, synonyms=[])
        G.add_node(test_node,      type="Diagnostic_Test", guideline_source=row["source"],
                   guideline_url=guideline_url, loinc_code=loinc_code)

        # Three relationship edges per rule:
        #   (1) primary → condition  (2) condition → test  (3) primary → test (shortcut)
        G.add_edge(primary_node,   condition_node, relationship="INDICATES_CONDITION",
                   source=row["source"], source_url=guideline_url,
                   literature_weight=literature_weight)
        G.add_edge(condition_node, test_node,      relationship="REQUIRES_TEST",
                   source=row["source"], source_url=guideline_url,
                   test_literature_weight=test_literature_weight)
        G.add_edge(primary_node,   test_node,      relationship="DIRECTLY_INDICATES_TEST",
                   source=row["source"], source_url=guideline_url)

        # Treatment / Adverse Event sub-graph — only for rules that carry a treatment
        if "treatment" in df.columns and pd.notna(row.get("treatment")):
            treatment_name = row["treatment"]
            treatment_node = f"Treatment: {treatment_name}"
            rxcui         = row.get("rxcui")         if "rxcui"         in df.columns else None
            adverse_event = row.get("adverse_event") if "adverse_event" in df.columns else None
            trial_count   = row.get("trial_count")   if "trial_count"   in df.columns else None

            G.add_node(treatment_node, type="Treatment", rxcui=rxcui)
            G.add_edge(condition_node, treatment_node,
                       relationship="RECOMMENDS_TREATMENT",
                       source=row["source"], source_url=guideline_url,
                       trial_count=trial_count)

            if pd.notna(adverse_event):
                adverse_event_node = f"Adverse Event: {str(adverse_event).title()}"
                G.add_node(adverse_event_node, type="Adverse_Event")
                G.add_edge(treatment_node, adverse_event_node,
                           relationship="HAS_ADVERSE_EVENT")

    return G


# ─────────────────────────────────────────────────────────────────────────────
# Main Pipeline Function (importable by external callers)
# ─────────────────────────────────────────────────────────────────────────────

def generate_knowledge_graph():
    """Run the full multi-ontology pipeline and save the serialised graph.

    Loads guideline_rules.json fresh on every call so dynamically added rules
    (e.g. those appended by the Diagnotix backend) are always picked up.

    Returns a dict with keys:
        nodes    — total node count in the rebuilt graph
        edges    — total edge count in the rebuilt graph
        pkl_path — absolute path to the saved .pkl file
    """
    _data_file = os.path.join(os.path.dirname(__file__), "guideline_rules.json")
    with open(_data_file, "r") as f:
        raw = json.load(f)

    guideline_rules = [r for r in raw if "raw_symptom" in r]
    df_rules = pd.DataFrame(guideline_rules)

    # Attach guideline URLs
    df_rules["url"] = df_rules.apply(
        lambda row: GUIDELINE_URLS.get((row["source"], row["test"]), None), axis=1
    )

    print("Extracting cross-mapped ontology codes...")

    # 1. US SNOMED via UMLS (requires a provisioned key — commented out until available)
    # df_rules[['umls_name', 'umls_code']] = df_rules.apply(
    #     lambda row: pd.Series(get_snomed_concept(row['raw_symptom'], UMLS_API_KEY)), axis=1
    # )

    # 2. Open ontology codes + synonyms via EMBL-EBI OLS (no key required)
    df_rules[['ebi_name', 'ebi_code', 'synonyms']] = df_rules.apply(
        lambda row: pd.Series(get_open_medical_concept(row['raw_symptom'])), axis=1
    )

    # 3. Canadian SNOMED CT via Canada Health Infoway
    print("Fetching Infoway access token...")
    infoway_token = get_infoway_access_token(INFOWAY_CLIENT_ID, INFOWAY_CLIENT_SECRET)

    if infoway_token:
        print("Token obtained. Querying Infoway FHIR API...")
        df_rules[['infoway_name', 'infoway_code']] = df_rules.apply(
            lambda row: pd.Series(get_infoway_snomed_concept(row['raw_symptom'], infoway_token)), axis=1
        )
    else:
        print("  WARNING: Infoway token unavailable — Canadian SNOMED CT codes will be omitted.")

    # 4. LOINC codes via UMLS — one lookup per unique test name to avoid redundant calls
    if UMLS_API_KEY:
        print("Looking up LOINC codes via UMLS...")
        loinc_map = {}
        for test_name in df_rules["test"].unique():
            _, code = get_umls_concept(test_name, "LNC", UMLS_API_KEY)
            loinc_map[test_name] = code
        df_rules["loinc_code"] = df_rules["test"].map(loinc_map)
    else:
        print("  WARNING: UMLS API key unavailable — LOINC codes will be omitted.")
        df_rules["loinc_code"] = None

    # 5. ICD-10-CM codes via UMLS — one lookup per unique condition name
    if UMLS_API_KEY:
        print("Looking up ICD-10-CM codes via UMLS...")
        icd10_map = {}
        for condition_name in df_rules["condition"].unique():
            _, code = get_umls_concept(condition_name, "ICD10CM", UMLS_API_KEY)
            icd10_map[condition_name] = code
        df_rules["icd10_code"] = df_rules["condition"].map(icd10_map)
    else:
        print("  WARNING: UMLS API key unavailable — ICD-10-CM codes will be omitted.")
        df_rules["icd10_code"] = None

    # 6. RxNorm RxCUI + OpenFDA adverse events — only for rows that carry a treatment
    if "treatment" in df_rules.columns:
        unique_treatments = df_rules["treatment"].dropna().unique()
        if len(unique_treatments) > 0:
            print("Looking up RxNorm and OpenFDA data for treatments...")
            rxcui_map         = {}
            adverse_event_map = {}
            for drug in unique_treatments:
                rxcui_map[drug]         = get_rxnorm_rxcui(drug)
                adverse_event_map[drug] = get_openfda_adverse_event(drug)
            df_rules["rxcui"]         = df_rules["treatment"].map(rxcui_map)
            df_rules["adverse_event"] = df_rules["treatment"].map(adverse_event_map)

    # 7. Europe PMC literature co-occurrence weights (symptom ↔ condition)
    print("Fetching Europe PMC literature co-occurrence weights (symptom ↔ condition)...")
    unique_sc_pairs = df_rules[["raw_symptom", "condition"]].drop_duplicates()
    lit_weight_map = {}
    for _, pair in unique_sc_pairs.iterrows():
        lit_weight_map[(pair["raw_symptom"], pair["condition"])] = \
            get_literature_cooccurrence(pair["raw_symptom"], pair["condition"])
    df_rules["literature_weight"] = df_rules.apply(
        lambda row: lit_weight_map.get((row["raw_symptom"], row["condition"]), 0), axis=1
    )

    # 8. Europe PMC literature co-occurrence weights (condition ↔ test)
    print("Fetching Europe PMC literature co-occurrence weights (condition ↔ test)...")
    unique_ct_lit_pairs = df_rules[["condition", "test"]].drop_duplicates()
    test_lit_weight_map = {}
    for _, pair in unique_ct_lit_pairs.iterrows():
        test_lit_weight_map[(pair["condition"], pair["test"])] = \
            get_literature_cooccurrence(pair["condition"], pair["test"])
    df_rules["test_literature_weight"] = df_rules.apply(
        lambda row: test_lit_weight_map.get((row["condition"], row["test"]), 0), axis=1
    )

    # 9. ClinicalTrials.gov trial counts (condition ↔ treatment)
    if "treatment" in df_rules.columns:
        unique_ct_pairs = (
            df_rules[df_rules["treatment"].notna()][["condition", "treatment"]]
            .drop_duplicates()
        )
        if not unique_ct_pairs.empty:
            print("Fetching ClinicalTrials.gov trial counts...")
            trial_count_map = {}
            for _, pair in unique_ct_pairs.iterrows():
                trial_count_map[(pair["condition"], pair["treatment"])] = \
                    get_clinical_trials_count(pair["condition"], pair["treatment"])
            df_rules["trial_count"] = df_rules.apply(
                lambda row: (
                    trial_count_map.get((row["condition"], row["treatment"]))
                    if pd.notna(row.get("treatment")) else None
                ),
                axis=1,
            )

    # Build the graph
    kg = build_unified_medical_kg(df_rules)

    # Save to the same directory as this script so paths are stable regardless
    # of the caller's working directory.
    pkl_path = os.path.join(os.path.dirname(__file__), "triage_knowledge_graph.pkl")
    with open(pkl_path, "wb") as f:
        pickle.dump(kg, f)

    print(f"\nKnowledge graph saved to '{pkl_path}'")
    print(f"  Nodes : {kg.number_of_nodes()}")
    print(f"  Edges : {kg.number_of_edges()}")

    return {
        "nodes":    kg.number_of_nodes(),
        "edges":    kg.number_of_edges(),
        "pkl_path": pkl_path,
    }


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    stats = generate_knowledge_graph()

    # Reload the saved graph for spot-checks
    with open(stats["pkl_path"], "rb") as f:
        kg = pickle.load(f)

    # --- Node Inspection ---
    print("\n--- Node Inspection ---")
    spot_check_nodes = [
        "Symptom: Chest Pain",
        "Risk Factor: Hypertension",
        "MOI: Fall On Outstretched Hand",
        "Attribute: Rebound Tenderness",
        "Attribute: Sudden Pain Onset",
        "Demographic: Female Of Childbearing Age",
    ]
    for node in spot_check_nodes:
        if node in kg.nodes:
            print(f"  [{node}] -> {dict(kg.nodes[node])}")
        else:
            print(f"  [{node}] NOT FOUND")

    # --- Guideline URL Verification ---
    print("\n--- Guideline URL Verification ---")
    for test_node in ["Test: ECG", "Test: Testicular Ultrasound",
                      "Test: Arm X-Ray", "Test: Appendix Ultrasound"]:
        if test_node in kg.nodes:
            attrs = kg.nodes[test_node]
            print(f"  [{test_node}]")
            print(f"    source       : {attrs.get('guideline_source')}")
            print(f"    guideline_url: {attrs.get('guideline_url')}")
        else:
            print(f"  [{test_node}] NOT FOUND")

    # --- Edge Verification (INDICATES_CONDITION) ---
    print("\n--- Edge Verification (INDICATES_CONDITION) ---")
    for src, tgt in [
        ("Symptom: Chest Pain",            "Condition: Acute Myocardial Infarction"),
        ("Symptom: Scrotal Pain",          "Condition: Testicular Torsion"),
        ("MOI: Fall On Outstretched Hand", "Condition: Wrist Fracture"),
        ("Attribute: Rebound Tenderness",  "Condition: Acute Appendicitis"),
    ]:
        if kg.has_edge(src, tgt):
            ed = kg.edges[src, tgt]
            print(f"  ({src}) -> ({tgt})")
            print(f"    source           : {ed.get('source')}")
            print(f"    source_url       : {ed.get('source_url')}")
            print(f"    literature_weight: {ed.get('literature_weight')}")
        else:
            print(f"  EDGE NOT FOUND: ({src}) -> ({tgt})")

    # --- Edge Verification (REQUIRES_TEST) ---
    print("\n--- Edge Verification (REQUIRES_TEST) ---")
    for src, tgt in [
        ("Condition: Acute Myocardial Infarction", "Test: ECG"),
        ("Condition: Testicular Torsion",          "Test: Testicular Ultrasound"),
        ("Condition: Wrist Fracture",              "Test: Arm X-Ray"),
        ("Condition: Acute Appendicitis",          "Test: Appendix Ultrasound"),
    ]:
        if kg.has_edge(src, tgt):
            ed = kg.edges[src, tgt]
            print(f"  ({src}) -> ({tgt})")
            print(f"    source                : {ed.get('source')}")
            print(f"    test_literature_weight: {ed.get('test_literature_weight')}")
        else:
            print(f"  EDGE NOT FOUND: ({src}) -> ({tgt})")

    # --- Edge Verification (RECOMMENDS_TREATMENT) ---
    print("\n--- Edge Verification (RECOMMENDS_TREATMENT) ---")
    for src, tgt in [
        ("Condition: Acute Myocardial Infarction", "Treatment: Aspirin"),
        ("Condition: Arm Fracture",               "Treatment: Ibuprofen"),
        ("Condition: Wrist Fracture",             "Treatment: Ibuprofen"),
    ]:
        if kg.has_edge(src, tgt):
            ed = kg.edges[src, tgt]
            print(f"  ({src}) -> ({tgt})")
            print(f"    source     : {ed.get('source')}")
            print(f"    trial_count: {ed.get('trial_count')}")
        else:
            print(f"  EDGE NOT FOUND: ({src}) -> ({tgt})")

    # --- LOINC & ICD-10 Code Verification ---
    print("\n--- LOINC & ICD-10 Code Verification ---")
    for node in ["Test: ECG", "Test: Arm X-Ray",
                 "Test: Appendix Ultrasound", "Test: Testicular Ultrasound"]:
        if node in kg.nodes:
            print(f"  [{node}]  loinc_code={kg.nodes[node].get('loinc_code')}")
        else:
            print(f"  [{node}] NOT FOUND")

    for node in ["Condition: Acute Myocardial Infarction", "Condition: Arm Fracture",
                 "Condition: Wrist Fracture", "Condition: Acute Appendicitis"]:
        if node in kg.nodes:
            print(f"  [{node}]  icd10_code={kg.nodes[node].get('icd10_code')}")
        else:
            print(f"  [{node}] NOT FOUND")

    # --- Treatment & Adverse Event Node Verification ---
    print("\n--- Treatment & Adverse Event Node Verification ---")
    for node in ["Treatment: Aspirin", "Treatment: Ibuprofen"]:
        if node in kg.nodes:
            attrs = kg.nodes[node]
            print(f"  [{node}]  rxcui={attrs.get('rxcui')}")
            for _, tgt, ed in kg.out_edges(node, data=True):
                if ed.get("relationship") == "HAS_ADVERSE_EVENT":
                    print(f"    -> HAS_ADVERSE_EVENT -> [{tgt}]")
        else:
            print(f"  [{node}] NOT FOUND")
