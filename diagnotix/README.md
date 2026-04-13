# Diagnotix — Dynamic Medical Knowledge Graph Builder

A full-stack web application for exploring and expanding the triage knowledge graph. The interface has two modes:

- **Navigate** — browse graph pathways, search nodes, filter by type, and add new diagnostic test pathways using an LLM pipeline.
- **Analyze** — per-pathway clinical chat backed by RAG retrieval, Semantic Scholar live abstracts, and Nebius Kimi-K2.5-fast.

## Architecture

```
diagnotix/
├── backend/                    # FastAPI (Python 3.12)
│   ├── main.py                 # App setup, CORS, router registration
│   ├── routers/
│   │   ├── chat.py             # POST /api/chat
│   │   ├── graph.py            # GET /api/tests · GET /api/graph · POST /api/add_test
│   │   └── kg.py               # POST /api/add_test (duplicate, not registered in main.py)
│   ├── services/
│   │   ├── chat_service.py     # Kimi-K2.5-fast chat with RAG + Semantic Scholar
│   │   ├── graph_service.py    # PKL → JSON-safe graph dict for the frontend
│   │   ├── kg_service.py       # LLM → rules → guideline_rules.json → PKL rebuild
│   │   └── semantic_scholar.py # Semantic Scholar Academic Graph API client
│   └── models/
│       └── schemas.py          # Pydantic request / response models
├── frontend/                   # React 19 + TypeScript + Vite 6
│   └── src/
│       ├── App.tsx             # Root component — sidebar + graph canvas + toast
│       ├── App.css             # Dark medical theme
│       ├── components/
│       │   ├── ChatBot.tsx     # Per-pathway chat thread (Analyze mode)
│       │   ├── ChatInput.tsx   # Text input for adding a new diagnostic test
│       │   ├── GraphCanvas.tsx # react-force-graph-2d visualisation
│       │   └── KGForm.tsx      # Diagnostic test input form with example chips
│       ├── services/
│       │   └── api.ts          # fetch wrappers for all backend endpoints
│       └── constants/
│           └── nodeColors.ts   # Node-type colour palette
└── requirements.txt            # Python dependencies
```

The backend imports `generate_knowledge_graph()` and `run_grounded_verify_pass()` directly from
`../knowledge-graphs/build_kg.py` and `../knowledge-graphs/audit_guidelines.py` — no code is duplicated.

---

## Setup

### 1. Create and activate a virtual environment

```bash
cd diagnotix
python3 -m venv venv
source venv/bin/activate
```

### 2. Install Python dependencies

```bash
pip install -r requirements.txt
pip install -r ../knowledge-graphs/requirements.txt   # needed for build_kg, audit_guidelines
```

### 3. Configure environment

The backend loads credentials from the **repo root `.env`** first, then from
`knowledge-graphs/.env`. Copy `.env.example` at the repo root and fill in at minimum:

```ini
LLM_PROVIDER=nebius            # "nebius" or "azure"

# Required for KG expansion (kg_service.py)
NEBIUS_API_KEY=...             # Nebius AI Studio key
NEBIUS_MODEL=meta-llama/Meta-Llama-3.1-70B-Instruct-fast

# Required for chat (chat_service.py)
# chat_service.py always uses Nebius Kimi-K2.5-fast regardless of LLM_PROVIDER
NEBIUS_API_KEY=...

# Azure OpenAI alternative for KG expansion (set LLM_PROVIDER=azure)
ENDPOINT_URL=https://your-resource.openai.azure.com/
DEPLOYMENT_NAME=your-deployment-name

# Optional: raises Semantic Scholar rate limit from ~100 req/5 min
SEMANTIC_SCHOLAR_API_KEY=...

# Optional: enables Bing Search grounding in rule generation
BING_SEARCH_KEY=...

# PKL file to serve (default: triage_knowledge_graph_enriched.pkl)
KG_PKL_FILE=triage_knowledge_graph_enriched.pkl
```

> **Azure provider note:** Azure requires `az login` before starting the backend.
> If you don't have Azure credentials, set `LLM_PROVIDER=nebius`.

### 4. Start the backend

```bash
cd diagnotix
source venv/bin/activate
uvicorn backend.main:app --reload --port 8000
```

Verify: `curl http://localhost:8000/health` → `{"status":"ok"}`

### 5. Start the frontend

In a separate terminal:

```bash
cd diagnotix/frontend
npm install
npm run dev
```

Open `http://localhost:5173` in your browser.

---

## Usage

### Navigate mode (default)

- **Pathway picker** — select any diagnostic test to filter the graph to its 1-hop cluster. "All Pathways" shows the full graph.
- **Node search** — type to find nodes by label; click a result to focus its 1-hop subgraph.
- **Node type legend** — click any type to toggle visibility.
- **Add Pathway** — enter a diagnostic test name (e.g. `CT Head`) and press Enter. The backend:
  1. Retrieves RAG grounding from Qdrant (if available) and live Semantic Scholar abstracts.
  2. Calls the configured LLM to generate 8–15 triage rules.
  3. Normalises, verifies, and regenerates rules in up to 3 convergence iterations.
  4. Appends surviving rules to `guideline_rules.json` and rebuilds the PKL.
  5. Returns only the new cluster (diff), which is appended to the canvas without a full reload.

Expect 1–4 minutes for the rule generation + graph rebuild pipeline.

### Analyze mode

Switch to **Analyze** in the sidebar to open per-pathway clinical chat threads. Each thread:
- Receives the current pathway's subgraph nodes and edges as LLM context.
- Retrieves relevant guideline passages from Qdrant (RAG) and live Semantic Scholar abstracts.
- Uses Nebius Kimi-K2.5-fast to generate cited, evidence-grounded responses.
- Preserves conversation history per pathway for multi-turn reasoning.

---

## API Reference

All routes are mounted under `/api`.

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Liveness check |
| `GET` | `/api/tests` | List all `Diagnostic_Test` nodes `[{id, label}]` |
| `GET` | `/api/graph?pathway=<name>` | Full graph or 1-hop pathway subgraph |
| `POST` | `/api/add_test` | Run LLM pipeline; return new node/edge diff |
| `POST` | `/api/chat` | Clinical chat with RAG + Semantic Scholar |

### `POST /api/add_test`

```json
// Request
{ "diagnostic_test": "CT Head" }

// Response 200 OK
{
  "success": true,
  "diagnostic_test": "CT Head",
  "new_rules_added": 11,
  "new_nodes": [...],
  "new_edges": [...],
  "total_nodes": 98,
  "total_edges": 247,
  "message": "Added 11 rules for 'CT Head'. Graph now has 98 nodes and 247 edges."
}
```

### `POST /api/chat`

```json
// Request
{
  "message": "What are the key ECG findings for acute MI?",
  "history": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}],
  "context": {
    "nodes": [...],
    "edges": [...],
    "pathway": "ECG"
  }
}

// Response 200 OK
{ "content": "..." }
```

---

## Troubleshooting

**`ModuleNotFoundError` on backend start:**
- Install both `requirements.txt` files (diagnotix + knowledge-graphs).
- Run `pip install beautifulsoup4 openai azure-identity` if issues persist.

**`ValueError: NEBIUS_API_KEY not set` (when LLM_PROVIDER=nebius):**
- Add `NEBIUS_API_KEY` to the repo root `.env`.

**`ValueError: ENDPOINT_URL / DEPLOYMENT_NAME missing` (when LLM_PROVIDER=azure):**
- Add placeholder values if you don't have Azure credentials, or switch to `LLM_PROVIDER=nebius`.

**Frontend port conflict:**
- Vite auto-increments to 5174+ if 5173 is in use. Check the `npm run dev` output.

**Graph pickle not loading:**
- Ensure `knowledge-graphs/triage_knowledge_graph_enriched.pkl` exists.
- Regenerate: `cd knowledge-graphs && python build_kg.py && python enrich_from_clingraph.py`

## Regenerate the graph visualisation

After adding pathways via the web app, you can also regenerate the standalone HTML:

```bash
cd knowledge-graphs
python visualize_kg.py
# Opens triage_knowledge_graph.html in your browser
```
