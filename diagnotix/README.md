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

### 1. Create and activate venv

```bash
cd diagnotix
python3 -m venv venv
source venv/bin/activate
```

### 2. Install Python dependencies

Install diagnotix requirements:

```bash
pip install -r requirements.txt
```

Then install knowledge-graphs dependencies (needed by the backend to load audit_guidelines):

```bash
pip install -r ../knowledge-graphs/requirements.txt
```

### 3. Configure environment

Add credentials to `knowledge-graphs/.env`:

```
ANTHROPIC_API_KEY=sk-ant-...

# Azure OpenAI (required for audit pipeline)
ENDPOINT_URL=https://your-resource.openai.azure.com/
DEPLOYMENT_NAME=your-deployment-name

# Optional: Other ontology APIs
INFOWAY_CLIENT_ID=...
INFOWAY_CLIENT_SECRET=...
UMLS_API_KEY=...
```

The backend will not start without `ENDPOINT_URL` and `DEPLOYMENT_NAME` (even if unused by your workflow).

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

Open `http://localhost:5173` in your browser (or `5174+` if port is in use).

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

## Troubleshooting

**Backend fails to start with `ModuleNotFoundError`:**
- Ensure you've installed both `diagnotix/requirements.txt` AND `knowledge-graphs/requirements.txt`
- Run `pip install beautifulsoup4 openai azure-identity` explicitly if issues persist

**Backend fails with `ValueError: ENDPOINT_URL / DEPLOYMENT_NAME missing`:**
- Add placeholder values to `knowledge-graphs/.env` if you don't have Azure OpenAI credentials:
  ```
  ENDPOINT_URL=https://placeholder.openai.azure.com/
  DEPLOYMENT_NAME=placeholder
  ```

**Frontend port conflict:**
- If port 5173 is in use, Vite will automatically use 5174 (or higher)
- Check `npm run dev` output to see the actual port

**Graph pickle file not loading:**
- Ensure `knowledge-graphs/triage_knowledge_graph.pkl` exists
- Regenerate with: `cd knowledge-graphs && python build_kg.py`

## Visualising the Updated Graph

After a successful expansion, regenerate the HTML visualisation:

```bash
cd knowledge-graphs
python visualize_kg.py
# Opens triage_knowledge_graph.html in your browser
```
