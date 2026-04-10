# backend/routers

FastAPI `APIRouter` modules. All routers are mounted under the `/api` prefix by `main.py`.

## Files

### `chat.py` — `POST /api/chat`

Thin adapter that delegates to `chat_service.sync_chat` via `asyncio.to_thread` to avoid
blocking the event loop during the multi-step LLM call chain.

**Request:** `ChatRequest` — `{message, history[], context: {nodes, edges, pathway}}`
**Response:** `ChatResponse` — `{content: str}`

Raises `HTTP 500` on `ValueError` (missing API key, etc.) or `HTTP 502` on LLM errors.

---

### `graph.py` — `GET /api/tests`, `GET /api/graph`, `POST /api/add_test`

Handles all graph-related endpoints:

| Endpoint | Handler | Description |
|----------|---------|-------------|
| `GET /api/tests` | `get_tests()` | Returns `[{id, label}]` for every `Diagnostic_Test` node — used to populate the sidebar pathway picker without loading the full graph. |
| `GET /api/graph?pathway=<name>` | `get_graph()` | Returns `{nodes, edges}` for the full graph or, when `pathway` is supplied, the 1-hop subgraph centred on that test node. Returns an empty graph (not an error) if the PKL has not been built yet. |
| `POST /api/add_test` | `add_test()` | Runs the LLM pipeline for the given `diagnostic_test` and returns only the newly created nodes and edges (the diff) so the frontend can append them without a full reload. |

---

### `kg.py` — `POST /api/add_test` (not registered)

A duplicate `add_test` route that calls the same `kg_service.add_test`. This router is **not
included** in `main.py` — `graph.py` handles the endpoint. Kept for reference.
