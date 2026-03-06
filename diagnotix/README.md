# Diagnotix — Dynamic Medical Knowledge Graph Builder

A full-stack web application that lets you expand the triage knowledge graph dynamically. Enter any diagnostic test, and Diagnotix uses Claude AI to extract clinical triage rules, grounds every node with multi-ontology codes, and rebuilds the NetworkX graph automatically.

## Architecture

```
diagnotix/
├── backend/                    # FastAPI (Python)
│   ├── main.py                 # App setup, CORS, router registration
│   ├── routers/kg.py           # POST /api/kg/expand
│   ├── services/kg_service.py  # LLM call + pipeline orchestration
│   └── models/schemas.py       # Pydantic request / response models
├── frontend/                   # React 19 + TypeScript + Vite
│   └── src/
│       ├── App.tsx             # Main component (idle / loading / success / error)
│       ├── App.css             # Custom CSS (dark medical theme)
│       ├── components/
│       │   └── KGForm.tsx      # Input form with example chips
│       └── services/
│           └── api.ts          # fetch wrapper for /api/kg/expand
└── requirements.txt            # Python dependencies
```

The backend imports `generate_knowledge_graph()` directly from
`../knowledge-graphs/build_kg.py` — no code is duplicated.

## Setup

### 1. Add your Anthropic API key

Add `ANTHROPIC_API_KEY` to `knowledge-graphs/.env`:

```
ANTHROPIC_API_KEY=sk-ant-...
INFOWAY_CLIENT_ID=...
INFOWAY_CLIENT_SECRET=...
```

### 2. Install Python dependencies

```bash
# From the project root, using the existing venv:
source .venv/bin/activate
pip install fastapi "uvicorn[standard]" anthropic
```

Or install everything from the diagnotix requirements file:

```bash
pip install -r diagnotix/requirements.txt
```

### 3. Start the backend

```bash
cd diagnotix
uvicorn backend.main:app --reload --port 8000
```

Verify: `curl http://localhost:8000/health` → `{"status":"ok"}`

### 4. Start the frontend

```bash
cd diagnotix/frontend
npm install
npm run dev
```

Open `http://localhost:5173` in your browser.

## Usage

1. Type a diagnostic test name in the input field (e.g. `CT Head`)
2. Click **Extract Guidelines and Build Graph**
3. Wait 1–3 minutes while the pipeline runs:
   - Claude generates 8–15 clinical triage rules
   - Rules are appended to `knowledge-graphs/guideline_rules.json`
   - `generate_knowledge_graph()` re-runs the full ontology grounding pipeline
   - The PKL is saved to `knowledge-graphs/triage_knowledge_graph.pkl`
4. The success screen shows the new rule count and updated graph statistics

## API Endpoint

```
POST /api/kg/expand
Content-Type: application/json

{ "diagnostic_test": "CT Head" }

→ 200 OK
{
  "success": true,
  "diagnostic_test": "CT Head",
  "new_rules_added": 11,
  "nodes": 98,
  "edges": 247,
  "message": "Added 11 rules for 'CT Head'. Graph rebuilt: 98 nodes, 247 edges."
}
```

## Visualising the Updated Graph

After a successful expansion, regenerate the HTML visualisation:

```bash
cd knowledge-graphs
python visualize_kg.py
# Opens triage_knowledge_graph.html in your browser
```
