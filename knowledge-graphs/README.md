
# Medical Triage Knowledge Graph

A pipeline to build, enrich, audit, and query a multi-ontology knowledge graph for Emergency Department triage. Clinical guidelines from ACR, AHA/ACC, NICE, and CTAS are encoded as structured rules that map symptoms, vitals, demographics, risk factors, mechanisms of injury, and clinical attributes to specific diagnostic tests and treatments.

---

## Repository Layout

```
knowledge-graphs/
├── guideline_rules.json               # Curated triage rules (source of truth)
│
├── ── Core Pipeline ──────────────────────────────────────────────────────────
├── build_kg.py                        # Builds the NetworkX graph → .pkl
├── enrich_from_clingraph.py           # Enriches graph with ClinGraph data
├── enrich_from_mimic_demo.py          # Enriches edges with MIMIC-IV Demo counts
├── clean_kg.py                        # Post-processing data quality corrections
├── visualize_kg.py                    # Renders the graph → interactive HTML
│
├── ── Quality Assurance ──────────────────────────────────────────────────────
├── audit_guidelines.py                # 4-pass retroactive guideline auditor
├── kg_audit_agent.py                  # 5-persona multi-agent graph auditor
├── kg_fact_checker.py                 # Edge evidence validation (PMC, ClinicalTrials)
│
├── ── Utilities ──────────────────────────────────────────────────────────────
├── triage_extraction_pipeline.py      # GraphRAG pipeline for patient triage
├── delete_nodes.py                    # Interactive node / pathway deletion tool
├── filter_guidelines.py               # Filter EPFL guidelines by diagnostic test
├── test_umls.py                       # UMLS API key validator
│
├── ── Config ─────────────────────────────────────────────────────────────────
├── requirements.txt                   # Python dependencies (Python ≥ 3.10)
├── .env                               # API credentials (not committed)
│
├── ── Generated Artifacts ────────────────────────────────────────────────────
├── triage_knowledge_graph.pkl             # Base graph (output of build_kg.py)
├── triage_knowledge_graph_enriched.pkl    # + ClinGraph enrichment (canonical)
├── triage_knowledge_graph_mimic_enriched.pkl  # + MIMIC edge counts
├── triage_knowledge_graph_clean.pkl       # + quality corrections
├── triage_knowledge_graph.html            # Interactive visualisation
├── audit_report.json                      # Output of kg_audit_agent.py
│
└── ── Subdirectories ─────────────────────────────────────────────────────────
    ├── vector_db/                     # Local ChromaDB + MedEmbed embedding scripts
    └── data/                          # External datasets (ClinGraph, MIMIC, guidelines)
```

---

## PKL Hierarchy

Scripts operate on different versions of the graph. The canonical working graph is `triage_knowledge_graph_enriched.pkl`.

```
build_kg.py
    → triage_knowledge_graph.pkl               (base graph, ~18 KB)
        │
enrich_from_clingraph.py
    → triage_knowledge_graph_enriched.pkl      (+ ClinGraph nodes/edges, ~92 KB) ← canonical
        │
enrich_from_mimic_demo.py
    → triage_knowledge_graph_mimic_enriched.pkl (+ MIMIC edge co-occurrence counts)
        │
clean_kg.py
    → triage_knowledge_graph_clean.pkl          (+ data quality corrections)
```

Most scripts default to `triage_knowledge_graph_enriched.pkl`. Use `--kg-path` to override.

---

## Prerequisites

```bash
pip install -r requirements.txt
```

Requires Python 3.10 or later.

---

## Environment Setup

Create a `.env` file inside `knowledge-graphs/` (copy `.env.example` if provided):

```text
# LLM provider: "nebius" or "azure"
LLM_PROVIDER=nebius
NEBIUS_API_KEY=your_nebius_key_here
NEBIUS_MODEL=meta-llama/Llama-3.3-70B-Instruct-fast

UMLS_API_KEY=your_umls_api_key_here
INFOWAY_CLIENT_ID=your_infoway_client_id_here
INFOWAY_CLIENT_SECRET=your_infoway_client_secret_here

# Local ChromaDB + MedEmbed (no API key required)
MEDEMBED_MODEL=abhinand/MedEmbed-large-v0.1
CHROMA_PATH=chroma_db
```

| Variable | Source | Used for |
|---|---|---|
| `NEBIUS_API_KEY` | [Nebius AI Studio](https://studio.nebius.ai/) | LLM calls in audit_guidelines.py, kg_audit_agent.py |
| `UMLS_API_KEY` | [UMLS account](https://uts.nlm.nih.gov/uts/login) | SNOMED CT US, LOINC, ICD-10-CM lookups |
| `INFOWAY_CLIENT_ID/SECRET` | [Canada Health Infoway](https://tss.infoway-inforoute.ca/) | SNOMED CT CA lookups |
| `MEDEMBED_MODEL` | HuggingFace (auto-downloaded) | Local embedding model for ChromaDB |

The EBI OLS4, NLM RxNorm, OpenFDA, Europe PMC, and ClinicalTrials.gov APIs require no credentials.

---

## Run Order

Run all scripts from inside the `knowledge-graphs/` directory.

### Step 1 — Build the base graph
```bash
python build_kg.py
# → triage_knowledge_graph.pkl
```

### Step 2 — Enrich with ClinGraph
```bash
python enrich_from_clingraph.py
# → triage_knowledge_graph_enriched.pkl  (canonical working graph)
```

### Step 3 — (Optional) Enrich edges with MIMIC-IV Demo counts
```bash
python enrich_from_mimic_demo.py
# → triage_knowledge_graph_mimic_enriched.pkl
```

### Step 4 — (Optional) Apply data quality corrections
```bash
python clean_kg.py
# → triage_knowledge_graph_clean.pkl
```

### Step 5 — Visualise
```bash
python visualize_kg.py
# → triage_knowledge_graph.html
```

### Step 6 — Audit guidelines (retroactive rule cleanup)
```bash
python audit_guidelines.py            # full 4-pass audit
python audit_guidelines.py --dry-run  # preview without modifying guideline_rules.json
python audit_guidelines.py --skip-verify --skip-normalise  # specificity check only
```

### Step 7 — Multi-agent graph audit
```bash
python kg_audit_agent.py                         # full A→B→C→D triad audit
python kg_audit_agent.py --persona-e-only        # node specificity check only (fast, no LLM)
python kg_audit_agent.py --resume                # resume from checkpoint after interruption
python kg_audit_agent.py --acuity HIGH           # audit only HIGH-acuity triads
python kg_audit_agent.py --dry-run               # report only, no files written
```

### Step 8 — Build semantic vector index
```bash
cd vector_db/
python build_vector_db_chroma.py               # embed all KG nodes into local ChromaDB
python build_vector_db_chroma.py --force-rebuild  # re-index from scratch
python build_vector_db_chroma.py --query "chest pain diaphoresis" --n 5  # semantic search
```

### Step 9 — Run the GraphRAG triage pipeline
```bash
python triage_extraction_pipeline.py                       # all 4 mock patients, full output
python triage_extraction_pipeline.py --output entities     # entity extraction only
python triage_extraction_pipeline.py --output context      # graph retrieval results only
python triage_extraction_pipeline.py --output prompt       # assembled LLM prompts only
python triage_extraction_pipeline.py --kg-path triage_knowledge_graph_enriched.pkl
```

---

## Node & Edge Management

### Delete a specific node
```bash
python delete_nodes.py --node "Symptom: Pediatric Assessment"
python delete_nodes.py --node "Symptom: Pediatric Assessment" --dry-run  # preview first
```

### Delete an entire test pathway (and its exclusively-owned nodes)
```bash
python delete_nodes.py "ECG"
python delete_nodes.py          # interactive menu
```

### Prune nodes with no path to any Diagnostic_Test
```bash
python delete_nodes.py --clean
python delete_nodes.py --clean --dry-run   # preview only
```

### Browse graph contents
```bash
python delete_nodes.py --list       # list all Test nodes
python delete_nodes.py --list-all   # list every node, grouped by type
```

---

## How the Knowledge Graph Was Built

The graph is constructed by `build_kg.py` in a nine-step pipeline. Each step enriches a shared pandas DataFrame (`df_rules`) before the final graph object is assembled and serialised.

### Step 0 — Curated Ground-Truth Rules

Everything starts with `guideline_rules.json`, a hand-authored list of clinical rules drawn from four evidence-based guidelines. Each rule encodes one clinical observation (a symptom, vital sign threshold, demographic factor, risk factor, clinical attribute, or mechanism of injury), the condition it suggests, and the diagnostic test that condition warrants. Three rules additionally carry a `treatment` field for first-line pharmacological management.

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

Six APIs are queried to attach standardised medical codes to each row. Lookups are deduplicated where possible to minimise network calls.

| Step | API | Credential | Enriches | Column(s) added |
|---|---|---|---|---|
| 2 | EMBL-EBI OLS4 (HP/MONDO/EFO) | None | `raw_symptom` | `ebi_name`, `ebi_code`, `synonyms` |
| 3 | Canada Health Infoway FHIR (SNOMED CT CA) | OAuth2 client credentials | `raw_symptom` | `infoway_name`, `infoway_code` |
| 4 | UMLS REST API (LOINC) | `UMLS_API_KEY` | `test` (unique values only) | `loinc_code` |
| 5 | UMLS REST API (ICD-10-CM) | `UMLS_API_KEY` | `condition` (unique values only) | `icd10_code` |
| 6 | NLM RxNorm + OpenFDA | None | `treatment` (where present) | `rxcui`, `adverse_event` |

---

### Steps 7–9 — Evidence Weighting via Literature and Trial APIs

| Step | API | Query pair | Column added | Used on edge |
|---|---|---|---|---|
| 7 | Europe PMC REST search | `raw_symptom` ↔ `condition` | `literature_weight` | `INDICATES_CONDITION` |
| 8 | Europe PMC REST search | `condition` ↔ `test` | `test_literature_weight` | `REQUIRES_TEST` |
| 9 | ClinicalTrials.gov v2 | `condition` ↔ `treatment` | `trial_count` | `RECOMMENDS_TREATMENT` |

---

### Final Step — Graph Assembly

`build_unified_medical_kg(df)` iterates the fully-enriched DataFrame and constructs the NetworkX `DiGraph`:

```
[Primary]──INDICATES_CONDITION──▶[Condition]──REQUIRES_TEST──▶[Test]
    │                                  │
    └──DIRECTLY_INDICATES_TEST─────────┘
                                       │
                               RECOMMENDS_TREATMENT
                                       │
                                  [Treatment]──HAS_ADVERSE_EVENT──▶[Adverse Event]
```

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

---

## File Reference

### `guideline_rules.json`

The single source of truth for all curated triage rules.

| Field | Type | Required | Description |
|---|---|---|---|
| `raw_symptom` | string | yes | Free-text clinical entity |
| `node_type` | string | yes | KG node category (see table above) |
| `condition` | string | yes | Target diagnosis |
| `test` | string | yes | Recommended diagnostic test |
| `source` | string | yes | Guideline (`AHA/ACC`, `ACR`, `NICE`, `CTAS`) |
| `treatment` | string | no | First-line treatment |

Comment-only entries (`{"_comment": "..."}`) are used as section separators and ignored by the loader.

---

### `audit_guidelines.py`

Retroactive 4-pass audit of `guideline_rules.json`. Rules are verified and optionally updated in-place.

| Pass | What it does | Requires LLM |
|---|---|---|
| 0 | Specificity check — flag nodes spanning ≥N tests as non-specific | No |
| 1 | Name normalisation — canonicalise via EBI OLS4 + UMLS | No |
| 2 | Evidence grounding — Europe PMC + web co-occurrence | No |
| 3 | LLM verification — remove implausible/fabricated rules | Yes |

```bash
python audit_guidelines.py --specificity-threshold 3   # default
python audit_guidelines.py --skip-specificity          # skip Pass 0
python audit_guidelines.py --skip-verify               # skip Pass 3 (no LLM needed)
```

Output: modified `guideline_rules.json` + `nonspecific_rules.json` sidecar.

---

### `kg_audit_agent.py`

5-persona multi-agent audit of compiled graph edges. Each triad (finding → condition → test) is reviewed by four AI personas, with a fifth pure-code persona for node specificity.

| Persona | Role | Requires LLM |
|---|---|---|
| E | Node Specificity Analyst — flags generic/multi-pathway nodes | No |
| A | Protocol & Triage Enforcer — vector DB retrieval + LLM review | Yes |
| B | Rapid Diagnostics Researcher — PMC + Semantic Scholar | No |
| C | ED Attending — pure LLM clinical reasoning | Yes |
| D | ED Medical Director — LLM synthesis of A+B+C | Yes |

```bash
python kg_audit_agent.py --persona-e-only              # run only Persona E (fast)
python kg_audit_agent.py --specificity-threshold 2     # stricter node specificity
python kg_audit_agent.py --resume                      # resume from cache checkpoint
python kg_audit_agent.py --acuity HIGH                 # filter to HIGH-acuity triads only
python kg_audit_agent.py --threshold 0.6               # confidence flag threshold
```

Output: `audit_report.json` + updated `.pkl` with `audit_metadata` edge attributes.

---

### `clean_kg.py`

Post-processing corrections applied to `triage_knowledge_graph_enriched.pkl`:

- Fix incorrect LOINC codes (MTHU*/LP*/LA* → authoritative panel codes)
- Fix incorrect ICD-10 codes (WHO overrides)
- Reclassify mis-typed ClinGraph nodes (R-codes, M25/M54/M79 → Symptom)
- Prune ClinGraph nodes with no path to any Test node

```bash
python clean_kg.py
python clean_kg.py --base-pkl triage_knowledge_graph_enriched.pkl --out-pkl triage_knowledge_graph_clean.pkl
```

---

### `delete_nodes.py`

Interactive tool for removing nodes or pathways from the canonical PKL. Always creates a timestamped backup before writing.

See [Node & Edge Management](#node--edge-management) section above for usage.

---

### `kg_fact_checker.py`

Validates edges against PMC and ClinicalTrials.gov evidence thresholds:

- `INDICATES_CONDITION` → ≥5 Europe PMC papers
- `REQUIRES_TEST` → ≥5 Europe PMC papers
- `RECOMMENDS_TREATMENT` → ≥1 ClinicalTrials.gov trial

Falls back to OLS4 synonyms stored on nodes if primary terms return too few results.

```bash
python kg_fact_checker.py
python kg_fact_checker.py --kg-path triage_knowledge_graph_enriched.pkl --verbose
```

---

### `triage_extraction_pipeline.py`

A four-stage GraphRAG pipeline that processes a synthetic patient triage record and produces a structured LLM prompt.

Stages:
1. **Synthea Mocker** — Generate 4 synthetic patients (one per pathway)
2. **Entity Extractor** — Parse patient data into KG node categories
3. **Graph Retrieval** — Traverse KG from matched entities to test nodes
4. **LLM Prompt Assembly** — Format context into a structured clinical prompt

```bash
python triage_extraction_pipeline.py --output entities
python triage_extraction_pipeline.py --output context
python triage_extraction_pipeline.py --output prompt
python triage_extraction_pipeline.py --output all
python triage_extraction_pipeline.py --kg-path triage_knowledge_graph_enriched.pkl
```

---

### `vector_db/`

See [`vector_db/README.md`](vector_db/README.md) for full documentation of the local ChromaDB + MedEmbed embedding pipeline.

Quick start:
```bash
cd vector_db/
python build_vector_db_chroma.py                  # index all KG nodes
python build_vector_db_chroma.py --query "fever"  # semantic search
```
