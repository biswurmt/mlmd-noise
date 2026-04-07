# Guidelines Vector Database

Two parallel scripts for building and querying a semantic vector index of medical guidelines. Both expose the same `query_guidelines()` API so they are interchangeable as backends.

| Script | Embedding model | Vector store | Requires |
|--------|----------------|--------------|---------|
| `build_vector_db.py` | Nebius e5-mistral-7b-instruct (4096-dim, cloud) | Qdrant Cloud | `NEBIUS_API_KEY`, `QDRANT_URL`, `QDRANT_API_KEY` |
| `build_vector_db_chroma.py` | MedEmbed / MedEIR (local, no key) | ChromaDB (local disk) | None — models download on first run |

---

## `build_vector_db_chroma.py` — Local Stack

### Dependencies

```bash
pip install chromadb sentence-transformers torch
```

> The Pylance "could not be resolved" warnings for `torch`, `sentence_transformers`, and `chromadb` disappear once the packages are installed into your active environment.

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
```

No Qdrant or Nebius credentials are needed. Models are downloaded from HuggingFace into `~/.cache/huggingface/` on first run; subsequent runs use the local cache.

### Usage

Run all commands from `knowledge-graphs/vector_db/`.

**Build the index (full dataset):**
```bash
python build_vector_db_chroma.py
```

**Gauge run (CMA source only, ~86 docs — good for testing):**
```bash
python build_vector_db_chroma.py --source cma
```

**Index a pre-filtered parquet file:**
```bash
python build_vector_db_chroma.py --parquet guidelines_remaining.parquet
```

**Query:**
```bash
python build_vector_db_chroma.py --query "ST elevation myocardial infarction" --n 5
```

**Query with diagnostic test filter:**
```bash
python build_vector_db_chroma.py --query "scrotal pain sudden onset" --test testicular_ultrasound --n 3
```

Indexing is resumable — already-indexed documents are skipped automatically on re-runs.

### Device selection

The script auto-detects the best available device in priority order: **MPS** (Apple Silicon) → **CUDA** (NVIDIA GPU) → **CPU**. No configuration needed.

### Switching models

Change `MEDEMBED_MODEL` in `.env` and re-run `build_index`. If the new model has a different embedding dimension, delete the existing `chroma_db/` folder first — ChromaDB will error if the stored dimension does not match the new model's output.

```bash
rm -rf knowledge-graphs/vector_db/chroma_db
python build_vector_db_chroma.py --source cma  # re-index
```

---

## Key differences from the Qdrant stack

### Symmetric embeddings
MedEmbed and MedEIR are **symmetric bi-encoders** — passages and queries are encoded identically. The asymmetric `"passage: "` / `"Instruct: ..."` prefixes used by e5-mistral are not needed and are not applied.

### `diagnostic_test` filtering
ChromaDB has no native substring operator (equivalent to Qdrant's `MatchText`). The `diagnostic_test` filter in `query_guidelines()` works by oversampling results (4×) and post-filtering in Python. Callers pass the same argument as before; behavior is equivalent.

### Text storage
Chunk text is stored in ChromaDB's native `documents` field rather than duplicated in the metadata payload. The `query_guidelines()` return dict uses the same keys as the Qdrant version (`matched_text`, `context`, `title`, `source`, `url`, `diagnostic_tests`, `score`).

### ChromaDB distance convention
ChromaDB returns cosine **distance** (0 = identical, 2 = opposite) for `hnsw:space: cosine`. The script converts this to **similarity** (`score = 1.0 − distance`) so returned scores are in the same [0, 1] range as the Qdrant stack.

---

## Python API

Both scripts expose the same function signature:

```python
from build_vector_db_chroma import query_guidelines

results = query_guidelines(
    query_text="wrist pain after fall on outstretched hand",
    n_results=5,
    diagnostic_test="arm_xray",   # optional substring filter
    context_window=1,              # chunks before/after each hit to include
)

for r in results:
    print(r["score"], r["title"])
    print(r["context"])
```
