# Diagnotix — AI-Powered Emergency Triage Decision Support

Diagnotix is a GraphRAG-based clinical decision-support system for Emergency Department triage. It combines a multi-ontology knowledge graph — built from AHA/ACC, ACR, NICE, and CTAS guidelines — with a Qdrant-backed vector database of medical guidelines and a live Semantic Scholar feed to give ED clinicians grounded, evidence-cited reasoning at the point of care.

---

## Repository Structure

```
mlmd-noise/
├── diagnotix/                          # Full-stack web application
│   ├── backend/                        # FastAPI (Python)
│   │   ├── main.py                     # App entry point, CORS, router registration
│   │   ├── routers/
│   │   │   ├── chat.py                 # POST /api/chat
│   │   │   ├── graph.py                # GET /api/graph, GET /api/tests, POST /api/add_test
│   │   │   └── kg.py                   # POST /api/add_test (duplicate, not registered)
│   │   ├── services/
│   │   │   ├── chat_service.py         # RAG + Semantic Scholar + Kimi-K2.5 chat
│   │   │   ├── graph_service.py        # KG serialisation for the frontend
│   │   │   ├── kg_service.py           # LLM-driven KG expansion pipeline
│   │   │   └── semantic_scholar.py     # Semantic Scholar API client
│   │   └── models/
│   │       └── schemas.py              # Pydantic request / response models
│   ├── frontend/                       # React 19 + TypeScript + Vite
│   │   └── src/
│   │       ├── App.tsx                 # Root component (graph view + chat panel)
│   │       ├── components/             # KGForm, chat UI components
│   │       └── services/api.ts         # fetch wrappers for the backend API
│   └── requirements.txt
│
├── knowledge-graphs/                   # KG construction & GraphRAG pipeline
│   ├── guideline_rules.json            # Curated triage rules (source of truth)
│   ├── build_kg.py                     # Builds NetworkX DiGraph → .pkl
│   ├── visualize_kg.py                 # Renders graph → interactive HTML (PyVis)
│   ├── triage_extraction_pipeline.py   # GraphRAG triage pipeline (4 mock patients)
│   ├── audit_guidelines.py             # Evidence auditing against live sources
│   ├── kg_fact_checker.py              # Fact-checking KG nodes via literature
│   ├── enrich_from_mimic_demo.py       # Enrich KG with MIMIC-IV demo patient data
│   ├── enrich_from_clingraph.py        # Enrich KG from ClinGraph dataset
│   ├── filter_guidelines.py            # Pre-filter raw guideline dataset
│   ├── clean_kg.py                     # Remove / correct KG nodes
│   ├── delete_nodes.py                 # Bulk node deletion utility
│   ├── vector_db/
│   │   └── build_vector_db.py          # Qdrant Cloud indexer + RAG query helper
│   ├── data/
│   │   ├── guidelines/                 # Raw guideline source documents
│   │   ├── guidelines_filtered/        # Pre-filtered Arrow / Parquet dataset
│   │   ├── mimic_demo_data/            # MIMIC-IV demo patient records
│   │   └── clingraph_data/             # ClinGraph structured clinical data
│   └── requirements.txt
│
├── data-processing/                    # Standalone CSV mapping & enrichment tools
│   ├── csv_extract_potential_test.py
│   ├── csv_mapping.py
│   └── kg_enrichment_pipeline.py
│
├── comms/                              # Presentations & pitch materials
│   ├── class-presentation/             # LaTeX/Beamer slide deck
│   └── pitch/                          # Pitch deck templates & ideation notes
│
├── .env.example                        # All required environment variables
├── .gitignore
└── .github/
    └── workflows/
        └── build-vector-db.yml         # CI: index guidelines into Qdrant Cloud
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Backend API | FastAPI + Uvicorn (Python 3.12) |
| Frontend | React 19, TypeScript, Vite 6 |
| Graph visualisation (frontend) | react-force-graph-2d |
| Markdown rendering | react-markdown + remark-gfm |
| LLM — chat & query rewrite | Nebius AI Studio — `moonshotai/Kimi-K2.5-fast` (OpenAI-compatible SDK) |
| LLM — KG expansion | Nebius AI Studio — Llama 3.1 / Mistral / Qwen / DeepSeek (configurable via `NEBIUS_MODEL`) |
| LLM — alternate provider | Azure OpenAI (AzureCliCredential or API key) |
| Knowledge graph library | NetworkX `DiGraph`, serialised as `.pkl` |
| KG visualisation (standalone) | PyVis + vis.js |
| Vector DB | Qdrant Cloud (cosine similarity, 4096-dim, collection: `guidelines`) |
| Embeddings | Nebius AI Studio — `intfloat/e5-mistral-7b-instruct` |
| Graph DB (optional) | Neo4j (bolt, configurable via `NEO4J_URI`) |
| Object storage | Nebius S3 (`diagnotix-kg` bucket, `eu-north1`) |
| Data format | Apache Parquet (pre-filtered guideline chunks) |
| CI/CD | GitHub Actions (manual `workflow_dispatch`) |

---

## APIs & Data Sources

### AI / Inference

| Service | Auth | Used for |
|---|---|---|
| **Nebius AI Studio** | `NEBIUS_API_KEY` | LLM completions (chat, KG expansion, query rewrite) and embeddings (`e5-mistral-7b-instruct`) |
| **Azure OpenAI** | `ENDPOINT_URL` + `DEPLOYMENT_NAME` + `az login` | Alternate LLM provider for KG expansion and the audit pipeline |

### Vector & Graph Storage

| Service | Auth | Used for |
|---|---|---|
| **Qdrant Cloud** | `QDRANT_URL` + `QDRANT_API_KEY` | Vector store for guideline RAG — stores chunked guideline passages (chunk size 2 000 chars, 200-char overlap) |
| **Nebius S3 Object Storage** | `S3_ACCESS_KEY_ID` + `S3_SECRET_ACCESS_KEY` | KG artifact storage (`.pkl` files, parquet) |

### Medical Literature & Research

| Service | Auth | Used for |
|---|---|---|
| **Semantic Scholar Academic Graph API** | `SEMANTIC_SCHOLAR_API_KEY` (optional — raises rate limit) | Live research paper abstracts injected into the chat system prompt |
| **Europe PMC REST** | None | Literature co-occurrence counts between symptom↔condition and condition↔test pairs (evidence weights on KG edges) |
| **ClinicalTrials.gov v2** | None | Trial counts for condition–treatment edges in the KG |
| **Bing Search** | `BING_SEARCH_KEY` | Guideline web search used by the evidence audit pipeline |

### Medical Ontologies & Terminologies

| Service | Auth | Used for |
|---|---|---|
| **EMBL-EBI OLS4** | None | HP / MONDO / EFO ontology codes and synonyms for clinical entity nodes |
| **Canada Health Infoway FHIR** | OAuth2 `INFOWAY_CLIENT_ID` + `INFOWAY_CLIENT_SECRET` | SNOMED CT CA codes for clinical entity nodes |
| **UMLS REST API** | `UMLS_API_KEY` | LOINC codes (diagnostic tests), ICD-10-CM codes (conditions), SNOMED CT US |
| **NLM RxNorm** | None | RxCUI drug identifiers for treatment nodes |
| **OpenFDA Drug Events** | None | MedDRA adverse reaction terms for treatment → adverse event edges |

### Clinical Datasets

| Dataset | Access | Used for |
|---|---|---|
| **MIMIC-IV Demo** | Kaggle credentials (`kaggle` Python package) | Real patient data for KG enrichment (`enrich_from_mimic_demo.py`) |
| **ClinGraph** | Bundled in `data/clingraph_data/` | Structured clinical graph data for KG enrichment |

---

## Diagnostic Pathways

The knowledge graph encodes triage rules for four diagnostic pathways drawn from four clinical guidelines:

| Pathway | Guidelines | Target conditions |
|---|---|---|
| ECG | AHA/ACC, CTAS | Acute MI, Arrhythmia, Tachycardia, Cardiogenic Shock |
| Testicular Ultrasound | ACR, NICE | Testicular Torsion, Epididymitis |
| Arm X-Ray | ACR | Arm / Wrist / Shoulder Fracture |
| Appendix Ultrasound | ACR, NICE | Acute Appendicitis, Ectopic Pregnancy, Ovarian Torsion |

---

## Quick Start

### 1. Environment

```bash
cp .env.example .env
# Fill in at minimum: NEBIUS_API_KEY, QDRANT_URL, QDRANT_API_KEY
```

### 2. Build the Knowledge Graph

```bash
cd knowledge-graphs
pip install -r requirements.txt
python build_kg.py          # builds triage_knowledge_graph_enriched.pkl
python visualize_kg.py      # renders triage_knowledge_graph.html
```

### 3. Index Guidelines into Qdrant

```bash
cd knowledge-graphs/vector_db
python build_vector_db.py                        # index full dataset
python build_vector_db.py --source cma           # gauge run (CMA only)
python build_vector_db.py --query "ST elevation" # test a query
```

The CI workflow (`.github/workflows/build-vector-db.yml`) can also run this step against a pre-filtered parquet release artifact.

### 4. Start the Backend

```bash
cd diagnotix
pip install -r requirements.txt
pip install -r ../knowledge-graphs/requirements.txt
uvicorn backend.main:app --reload --port 8000
# Verify: curl http://localhost:8000/health
```

### 5. Start the Frontend

```bash
cd diagnotix/frontend
npm install
npm run dev
# Open http://localhost:5173
```

---

## Environment Variables

All variables are documented in `.env.example`. A summary:

| Variable | Required | Description |
|---|---|---|
| `LLM_PROVIDER` | Yes | `nebius` or `azure` |
| `NEBIUS_API_KEY` | Yes | Nebius AI Studio API key |
| `NEBIUS_MODEL` | Yes | Model ID for KG expansion (default: `meta-llama/Meta-Llama-3.1-8B-Instruct`) |
| `NEBIUS_EMBEDDING_MODEL` | Yes | Embedding model (default: `intfloat/e5-mistral-7b-instruct`) |
| `ENDPOINT_URL` | Azure only | Azure OpenAI endpoint URL |
| `DEPLOYMENT_NAME` | Azure only | Azure OpenAI deployment name |
| `QDRANT_URL` | Yes | Qdrant Cloud cluster URL |
| `QDRANT_API_KEY` | Yes | Qdrant Cloud API key |
| `UMLS_API_KEY` | KG build | UMLS API key (LOINC + ICD-10-CM + SNOMED CT US) |
| `INFOWAY_CLIENT_ID` | KG build | Canada Health Infoway OAuth2 client ID |
| `INFOWAY_CLIENT_SECRET` | KG build | Canada Health Infoway OAuth2 client secret |
| `SEMANTIC_SCHOLAR_API_KEY` | Optional | Raises Semantic Scholar rate limit |
| `BING_SEARCH_KEY` | Audit pipeline | Bing Search API key |
| `NEO4J_URI` | Optional | Neo4j bolt URI (default: `bolt://neo4j:7687`) |
| `NEO4J_USER` / `NEO4J_PASSWORD` | Optional | Neo4j credentials |
| `KG_PKL_FILE` | Yes | PKL file to load (default: `triage_knowledge_graph_enriched.pkl`) |
| `S3_ENDPOINT_URL` | Storage | Nebius S3 endpoint |
| `S3_ACCESS_KEY_ID` / `S3_SECRET_ACCESS_KEY` | Storage | Nebius S3 credentials |
| `S3_BUCKET` | Storage | S3 bucket name (default: `diagnotix-kg`) |
