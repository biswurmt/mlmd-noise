
# Medical Triage Knowledge Graph

A pipeline to build, enrich, and query a multi-ontology knowledge graph for Emergency Department triage. Clinical guidelines from ACR, AHA/ACC, NICE, and CTAS are encoded as structured rules that map symptoms, vitals, demographics, risk factors, mechanisms of injury, and clinical attributes to specific diagnostic tests and treatments.

---

## Repository Layout

```
knowledge-graphs/
├── guideline_rules.json          # Curated triage rules (source of truth)
├── build_kg.py                   # Builds the NetworkX graph → .pkl
├── visualize_kg.py               # Renders the graph → interactive HTML
├── triage_extraction_pipeline.py # GraphRAG pipeline for patient triage
├── triage_knowledge_graph.pkl    # Compiled graph (generated artifact)
├── triage_knowledge_graph.html   # Interactive visualisation (generated artifact)
├── requirement.txt               # Python dependencies
└── .env                          # API credentials (not committed)
```

---

## Prerequisites

```bash
pip install -r requirement.txt
```

---

## Environment Setup

Create a `.env` file inside `knowledge-graphs/` with your API credentials:

```text
UMLS_API_KEY=your_umls_api_key_here
INFOWAY_CLIENT_ID=your_infoway_client_id_here
INFOWAY_CLIENT_SECRET=your_infoway_client_secret_here
```

| Variable | Source | Used for |
|---|---|---|
| `UMLS_API_KEY` | [UMLS account](https://uts.nlm.nih.gov/uts/login) | SNOMED CT US, LOINC, ICD-10-CM lookups |
| `INFOWAY_CLIENT_ID` / `INFOWAY_CLIENT_SECRET` | [Canada Health Infoway](https://tss.infoway-inforoute.ca/) | SNOMED CT CA lookups |

The EBI OLS4, NLM RxNorm, and OpenFDA APIs require no credentials.

---

## Usage

Run scripts from inside the `knowledge-graphs/` directory.

```bash
# Step 1 — build the graph and save it to disk
python build_kg.py

# Step 2 — render the interactive HTML visualisation
python visualize_kg.py

# Step 3 — run the GraphRAG triage pipeline (all 4 mock patients, full output)
python triage_extraction_pipeline.py

# Step 3 variants
python triage_extraction_pipeline.py --output entities   # entity extraction only
python triage_extraction_pipeline.py --output context    # graph retrieval results only
python triage_extraction_pipeline.py --output prompt     # assembled LLM prompts only
python triage_extraction_pipeline.py --kg-path /path/to/triage_knowledge_graph.pkl
```

---

## How the Knowledge Graph Was Built

The graph is constructed by `build_kg.py` in a nine-step pipeline. Each step enriches a shared pandas DataFrame (`df_rules`) before the final graph object is assembled and serialised.

### Step 0 — Curated Ground-Truth Rules

Everything starts with `guideline_rules.json`, a hand-authored list of 45 clinical rules drawn from four evidence-based guidelines. Each rule encodes one clinical observation (a symptom, vital sign threshold, demographic factor, risk factor, clinical attribute, or mechanism of injury), the condition it suggests, and the diagnostic test that condition warrants. Three rules additionally carry a `treatment` field for first-line pharmacological management.

```
raw_symptom  ──→  condition  ──→  test  (+ optional treatment)
    │
  node_type (Symptom | Vital_Sign_Threshold | Risk_Factor | ...)
  source    (AHA/ACC | ACR | NICE | CTAS)
```

The four diagnostic pathways covered:

| Pathway | Guidelines | Target conditions |
|---|---|---|
| ECG | AHA/ACC, CTAS | Acute MI, Arrhythmia, Tachycardia, Cardiogenic Shock |
| Testicular Ultrasound | ACR, NICE | Testicular Torsion, Epididymitis |
| Arm X-Ray | ACR | Arm / Wrist / Shoulder Fracture |
| Appendix Ultrasound | ACR, NICE | Acute Appendicitis, Ectopic Pregnancy, Ovarian Torsion |

---

### Step 1 — Guideline URL Grounding

A static `GUIDELINE_URLS` dict maps every `(source, test)` pair to the DOI or official page of the originating guideline document. The URL is applied to `df_rules` as a `url` column and later stamped onto every edge and Test node in the graph, making every clinical assertion directly traceable to its primary source.

---

### Steps 2–6 — Ontology Enrichment via External APIs

Six APIs are queried to attach standardised medical codes to each row. Lookups are deduplicated where possible (e.g. LOINC is fetched once per unique test name, not once per row) to minimise network calls.

| Step | API | Credential | Enriches | Column(s) added |
|---|---|---|---|---|
| 2 | EMBL-EBI OLS4 (HP/MONDO/EFO) | None | `raw_symptom` | `ebi_name`, `ebi_code`, `synonyms` |
| 3 | Canada Health Infoway FHIR (SNOMED CT CA) | OAuth2 client credentials | `raw_symptom` | `infoway_name`, `infoway_code` |
| 4 | UMLS REST API (LOINC) | `UMLS_API_KEY` | `test` (unique values only) | `loinc_code` |
| 5 | UMLS REST API (ICD-10-CM) | `UMLS_API_KEY` | `condition` (unique values only) | `icd10_code` |
| 6 | NLM RxNorm + OpenFDA | None | `treatment` (where present) | `rxcui`, `adverse_event` |

The EBI OLS4 step also extracts the `synonym` field from the top-ranked ontology match and stores it as a Python list. These synonyms are later used by `kg_fact_checker.py` to improve evidence-lookup recall when the primary term returns too few results.

---

### Steps 7–9 — Evidence Weighting via Literature and Trial APIs

Three further API calls attach real-world evidence counts directly to the DataFrame before the graph is assembled. These counts become numeric edge attributes that drive edge-width scaling in the HTML visualisation.

| Step | API | Query pair | Column added | Used on edge |
|---|---|---|---|---|
| 7 | Europe PMC REST search | `raw_symptom` ↔ `condition` | `literature_weight` | `INDICATES_CONDITION` |
| 8 | Europe PMC REST search | `condition` ↔ `test` | `test_literature_weight` | `REQUIRES_TEST` |
| 9 | ClinicalTrials.gov v2 | `condition` ↔ `treatment` | `trial_count` | `RECOMMENDS_TREATMENT` |

Both Europe PMC steps deduplicate to unique pairs before hitting the API and use a shared in-memory cache (`_lit_cache`) so repeated term combinations are never re-fetched. The ClinicalTrials step only runs for the three rules that carry a `treatment` value.

---

### Final Step — Graph Assembly

`build_unified_medical_kg(df)` iterates the fully-enriched DataFrame once and constructs the NetworkX `DiGraph`. For every row it creates up to five nodes and up to five edges:

```
[Primary]──INDICATES_CONDITION──▶[Condition]──REQUIRES_TEST──▶[Test]
    │                                  │
    └──DIRECTLY_INDICATES_TEST─────────┘
                                       │
                               RECOMMENDS_TREATMENT
                                       │
                                  [Treatment]──HAS_ADVERSE_EVENT──▶[Adverse Event]
```

Every node carries the ontology codes and synonyms fetched in Steps 2–6. Every edge carries the guideline `source`, `source_url`, and the appropriate evidence-weight attribute from Steps 7–9. The completed graph is serialised to `triage_knowledge_graph.pkl` with `pickle`.

---

### Data Flow Diagram

```
guideline_rules.json
        │
        ▼
   df_rules (DataFrame)
        │
        ├─ Step 1 ──→ url            (GUIDELINE_URLS static map)
        ├─ Step 2 ──→ ebi_code, synonyms         (EBI OLS4)
        ├─ Step 3 ──→ infoway_code               (Infoway FHIR)
        ├─ Step 4 ──→ loinc_code                 (UMLS / LNC)
        ├─ Step 5 ──→ icd10_code                 (UMLS / ICD10CM)
        ├─ Step 6 ──→ rxcui, adverse_event        (RxNorm + OpenFDA)
        ├─ Step 7 ──→ literature_weight           (Europe PMC)
        ├─ Step 8 ──→ test_literature_weight      (Europe PMC)
        └─ Step 9 ──→ trial_count                (ClinicalTrials.gov)
                │
                ▼
   build_unified_medical_kg(df)
                │
                ▼
   triage_knowledge_graph.pkl   ──→   triage_knowledge_graph.html
                │                              (visualize_kg.py)
                ▼
   kg_fact_checker.py  /  triage_extraction_pipeline.py
```

---

## File Reference

### `guideline_rules.json`

The single source of truth for all curated triage rules. Each entry is a JSON object with the following fields:

| Field | Type | Required | Description |
|---|---|---|---|
| `raw_symptom` | string | yes | Free-text clinical entity (symptom, vital threshold, etc.) |
| `node_type` | string | yes | KG node category (see table below) |
| `condition` | string | yes | Target diagnosis this entity indicates |
| `test` | string | yes | Recommended diagnostic test |
| `source` | string | yes | Clinical guideline (`AHA/ACC`, `ACR`, `NICE`, `CTAS`) |
| `treatment` | string | no | First-line treatment associated with the condition |

**`node_type` vocabulary**

| Value | Meaning |
|---|---|
| `Symptom` | Clinical presenting symptom |
| `Vital_Sign_Threshold` | Numeric vital sign cutoff (e.g., `heart rate > 100`) |
| `Demographic_Factor` | Age or sex-based triage modifier |
| `Risk_Factor` | Pre-existing condition or relevant history |
| `Clinical_Attribute` | Onset quality, exam finding, or pain character |
| `Mechanism_of_Injury` | Mechanism driving a trauma presentation |

Comment-only entries (objects with only a `_comment` key) are used as section separators and are ignored by the loader.

---

### `build_kg.py`

Loads `guideline_rules.json`, queries five medical ontology APIs to enrich the rules with standardised codes, builds a NetworkX `DiGraph`, and serialises it to `triage_knowledge_graph.pkl`.

#### API functions

**`get_snomed_concept(term, api_key) → (name, code)`**
Queries the UMLS REST API (`uts-ws.nlm.nih.gov`) for a SNOMED CT US concept matching `term`. Returns the canonical concept name and its UI code, or `(term, None)` on failure.

**`get_umls_concept(term, sabs, api_key) → (name, code)`**
Generalised UMLS lookup accepting any vocabulary via the `sabs` parameter (e.g., `"LNC"` for LOINC, `"ICD10CM"` for ICD-10-CM). Returns `(name, code)` or `(term, None)` on failure. Used to map diagnostic test names to LOINC codes and conditions to ICD-10-CM codes.

**`get_infoway_access_token(client_id, client_secret) → token | None`**
Exchanges Infoway OAuth2 client credentials for a bearer access token using the `client_credentials` grant. Returns the token string or `None` on failure.

**`get_infoway_snomed_concept(term, access_token) → (display, code)`**
Queries the Canada Health Infoway FHIR `ValueSet/$expand` endpoint for a SNOMED CT CA concept matching `term`. Returns `(display, code)` or `(term, None)` on failure.

**`get_open_medical_concept(term) → (label, short_form, synonyms)`**
Queries the EMBL-EBI OLS4 API (no key required) across the HP, MONDO, and EFO ontologies for an open concept matching `term`. Returns `(label, short_form, synonyms_list)` where `synonyms_list` is a (possibly empty) list of alternative clinical terms extracted from the OLS4 `synonym` field. Returns `(term, None, [])` on failure.

**`get_rxnorm_rxcui(term) → rxcui | None`**
Queries the free NLM RxNorm REST API (`rxnav.nlm.nih.gov`) for the RxCUI identifier of a drug name. Returns the RxCUI string or `None` if not found.

**`get_openfda_adverse_event(term) → reaction_term | None`**
Queries the OpenFDA drug adverse event API (`api.fda.gov/drug/event.json`) and retrieves the most-reported MedDRA reaction term for a given medication using the count aggregation endpoint. Returns the term string or `None` on failure.

#### Graph-builder helpers

**`_resolve_node_type(row, df) → (node_type, node_prefix)`**
Resolves the `node_type` and its display prefix for a given DataFrame row. Reads the explicit `node_type` column when present; falls back to a heuristic that infers the type from the `raw_symptom` string (checking for `>` / `<` operators and the word `age`).

**`build_unified_medical_kg(df) → nx.DiGraph`**
Iterates over every row in the enriched DataFrame and adds the following nodes and edges to a directed graph:

- **Primary node** — the clinical entity (`Symptom`, `Vital`, `Attribute`, etc.) with `ebi_open_code` and `snomed_ca_code` attributes when available.
- **Condition node** — the target diagnosis, with `icd10_code` attribute.
- **Test node** — the recommended diagnostic test, with `guideline_source`, `guideline_url`, and `loinc_code` attributes.
- **Treatment node** *(when `treatment` is present)* — drug name node with `rxcui` attribute, linked from the Condition node.
- **Adverse Event node** *(when OpenFDA data is available)* — MedDRA reaction term node, linked from the Treatment node.

Edges created per rule:

| Edge | Relationship type |
|---|---|
| Primary → Condition | `INDICATES_CONDITION` |
| Condition → Test | `REQUIRES_TEST` |
| Primary → Test | `DIRECTLY_INDICATES_TEST` |
| Condition → Treatment | `RECOMMENDS_TREATMENT` |
| Treatment → Adverse Event | `HAS_ADVERSE_EVENT` |

All edges carry `source` (guideline name) and `source_url` (official guideline URL) attributes.

---

### `visualize_kg.py`

Loads `triage_knowledge_graph.pkl` and renders it as an interactive HTML page using PyVis.

**`visualize_interactive_kg(G)`**
Accepts a NetworkX `DiGraph` and produces `triage_knowledge_graph.html`. Behaviour:

- Assigns a distinct colour to each `node_type` (symptom = red/pink, condition = blue, diagnostic test = green, vital = orange, demographic = purple, risk factor = yellow, mechanism of injury = rose, clinical attribute = mint).
- Node hover tooltips show all stored attributes (ontology codes, guideline URLs, LOINC codes, etc.), excluding `NaN` values.
- Edge labels display the relationship type and guideline source directly on the arrow; hover tooltips repeat this information.
- Adds an in-browser physics control panel so the layout can be adjusted interactively.

---

### `triage_extraction_pipeline.py`

A four-stage GraphRAG pipeline that takes a synthetic patient triage record, extracts clinical entities, traverses the knowledge graph to identify evidence-grounded diagnostic tests, and assembles a structured LLM system prompt.

#### Stage 1 — Synthea Mocker

**`generate_mock_patients() → list[dict]`**
Returns four synthetic patient triage records, one per diagnostic pathway (ECG, Testicular Ultrasound, Arm X-Ray, Appendix Ultrasound). Each record contains `patient_id`, `pathway`, `demographics`, `vitals`, `chief_complaint`, `pain_scale`, `pain_onset`, `mechanism_of_injury`, `clinical_attributes`, and `medical_history`.

#### Stage 2 — Entity Extractor

**`extract_entities(patient_record) → dict`**
Parses a patient record and returns a dict with six entity lists (`symptoms`, `vitals`, `demographics`, `risk_factors`, `mechanisms_of_injury`, `clinical_attributes`) containing canonical `raw_symptom` strings that match node labels in the knowledge graph.

Extraction strategy by category:

| Category | Source field | Method |
|---|---|---|
| Symptoms | `chief_complaint` | Substring keyword match against `_SYMPTOM_KEYWORDS` |
| Vitals | `vitals` | Numeric threshold comparisons (HR, BP, temperature) |
| Demographics | `demographics` | Age threshold and sex + age range checks |
| Risk factors | `medical_history` | Substring keyword match against `_RISK_FACTOR_KEYWORDS` |
| Mechanisms of injury | `mechanism_of_injury` | Multi-keyword AND match against `_MOI_KEYWORDS` |
| Clinical attributes | `pain_onset`, `clinical_attributes` | Regex for onset (`sudden`/`gradual`) + `_CLINICAL_ATTR_KEYWORDS` |

#### Stage 3 — Graph Retrieval

**`_upsert_test(recommended_tests, test_name, test_node, guideline, trace_entry) → None`**
Internal helper that inserts a new test entry into the `recommended_tests` accumulator dict, or updates an existing one by merging guideline sources, triggering entities, and appending to the reasoning trace.

**`get_triage_context(extracted_entities, kg_path) → dict`**
Loads the serialised knowledge graph and traverses it to identify recommended diagnostic tests:

1. Flattens all extracted entities into a single list.
2. Matches entities to graph nodes via case-insensitive substring comparison against the node label (the part after `": "`).
3. For each matched node, follows outgoing edges: `DIRECTLY_INDICATES_TEST` edges record the test in one hop; `INDICATES_CONDITION` edges are followed a second hop along `REQUIRES_TEST` edges to reach the test.
4. Builds a `reasoning_trace` for each recommended test recording the triggering entity, traversal path, and guideline source.

Returns `matched_graph_nodes` (sorted list) and `recommended_tests` (dict keyed by test name).

#### Stage 4 — LLM Prompt Assembly

**`assemble_llm_prompt(patient_record, triage_context) → str`**
Formats patient demographics, vitals, medical history, matched KG nodes, and evidence-graded recommended tests into a structured system prompt. The prompt instructs the LLM to confirm or deprioritise each recommended test, identify additional tests not in the graph, assign a CTAS triage level (1–5), flag time-critical findings, and return output in a strict JSON schema.

---

## Graph Schema Summary

| Node prefix | `type` attribute | Key attributes |
|---|---|---|
| `Symptom:` | `Symptom` | `ebi_open_code`, `snomed_ca_code` |
| `Vital:` | `Vital_Sign_Threshold` | `ebi_open_code`, `snomed_ca_code` |
| `Demographic:` | `Demographic_Factor` | `ebi_open_code`, `snomed_ca_code` |
| `Risk Factor:` | `Risk_Factor` | `ebi_open_code`, `snomed_ca_code` |
| `Attribute:` | `Clinical_Attribute` | `ebi_open_code`, `snomed_ca_code` |
| `MOI:` | `Mechanism_of_Injury` | `ebi_open_code`, `snomed_ca_code` |
| `Condition:` | `Condition` | `icd10_code` |
| `Test:` | `Diagnostic_Test` | `guideline_source`, `guideline_url`, `loinc_code` |
| `Treatment:` | `Treatment` | `rxcui` |
| `Adverse Event:` | `Adverse_Event` | — |
