# Guidelines Vector Database

Two scripts for building and querying a semantic vector index. Both live in this folder and share the same ChromaDB / Qdrant store path.

| Script | What gets embedded | Embedding model | Vector store |
|--------|-------------------|----------------|--------------|
| `build_vector_db_chroma.py` | KG nodes from `triage_knowledge_graph_enriched.pkl` | MedEmbed / MedEIR (local) | ChromaDB (local disk) |
| `build_vector_db.py` | `epfl-llm/guidelines` Arrow dataset chunks | Nebius e5-mistral-7b-instruct (cloud) | Qdrant Cloud *(legacy)* |

`build_vector_db_chroma.py` is the active script. It embeds every node in the enriched knowledge graph — Symptoms, Conditions, Vital Sign Thresholds, Demographic Factors, Risk Factors, Clinical Attributes, and Mechanisms of Injury — into a local ChromaDB collection called `kg_nodes`.

---

## `build_vector_db_chroma.py`

### Dependencies

```bash
pip install chromadb sentence-transformers torch
```

### Configuration (`.env`)

```ini
# Embedding model — pick one:
#   abhinand/MedEmbed-small-v0.1    384-dim  ~340 MB  fastest
#   abhinand/MedEmbed-base-v0.1     768-dim  ~440 MB  balanced
#   abhinand/MedEmbed-large-v0.1   1024-dim  ~1.3 GB  best quality (default)
#   Thakkar-AI/MedEIR               768-dim            IR-tuned alternative
MEDEMBED_MODEL=abhinand/MedEmbed-large-v0.1

# Local path for the ChromaDB on-disk store
CHROMA_PATH=./knowledge-graphs/vector_db/chroma_db

# KG pickle to embed (default: triage_knowledge_graph_enriched.pkl)
KG_PKL_FILE=triage_knowledge_graph_enriched.pkl
```

No Qdrant or Nebius credentials needed. Models download from HuggingFace into `~/.cache/huggingface/` on first run.

### Device selection

Auto-detected in priority order: **MPS** (Apple Silicon) → **CUDA** (NVIDIA GPU) → **CPU**.

---

### Usage

Run from `knowledge-graphs/vector_db/` or supply a full path with `--graph-pkl`.

**Build the index:**
```bash
python build_vector_db_chroma.py
```

**Rebuild from scratch (drops existing collection):**
```bash
python build_vector_db_chroma.py --force-rebuild
```

**Preview node texts without writing to ChromaDB:**
```bash
python build_vector_db_chroma.py --dry-run
```

**Use a different PKL:**
```bash
python build_vector_db_chroma.py --graph-pkl ../../knowledge-graphs/triage_knowledge_graph_enriched.pkl
```

**Query:**
```bash
python build_vector_db_chroma.py --query "chest pain with diaphoresis" --n 5
```

**Query filtered by node type:**
```bash
python build_vector_db_chroma.py --query "wrist fracture after fall" --type Symptom
python build_vector_db_chroma.py --query "diabetes mellitus" --type Condition
```

Valid `--type` values: `Symptom`, `Condition`, `Vital_Sign_Threshold`, `Demographic_Factor`, `Risk_Factor`, `Clinical_Attribute`, `Mechanism_of_Injury`.

Indexing is resumable — already-indexed nodes are detected from ChromaDB metadata and skipped on re-runs.

---

### What gets embedded

Each KG node is converted to a short natural-language passage before embedding:

```
Symptom: Chest Pain. Associated tests: ECG. SNOMED: 29857009. ICD-10: R07.9.
```

The passage includes:
- Node type + label
- Associated diagnostic tests (from `test_evidence` attribute)
- Ontology codes: SNOMED CT CA, ICD-10, ICD-10-CA, EBI/OLS

Only the 7 embeddable node types are indexed (Test nodes and edges are excluded).

---

### Python API

```python
from build_vector_db_chroma import query_kg_nodes

results = query_kg_nodes(
    query_text="sudden scrotal pain",
    n_results=5,
    node_type="Symptom",   # optional
)

for r in results:
    print(r["score"], r["node_id"], r["snomed_ca_code"])
```

Each result dict contains: `score`, `node_id`, `label`, `node_type`, `text`, `snomed_ca_code`, `icd10_code`, `icd10ca_code`, `ebi_open_code`.

---

### Switching models

Change `MEDEMBED_MODEL` in `.env`. If the new model has a different output dimension, delete `chroma_db/` and re-index:

```bash
rm -rf knowledge-graphs/vector_db/chroma_db
python build_vector_db_chroma.py
```
