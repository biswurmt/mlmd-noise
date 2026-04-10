# backend/models

Pydantic request and response schemas shared across routers and services.

## `schemas.py`

| Model | Direction | Used by | Fields |
|-------|-----------|---------|--------|
| `GraphData` | Response | `GET /api/graph` | `nodes: List[Dict]`, `edges: List[Dict]` |
| `AddTestRequest` | Request | `POST /api/add_test` | `diagnostic_test: str` |
| `AddTestResponse` | Response | `POST /api/add_test` | `success`, `diagnostic_test`, `new_rules_added`, `new_nodes`, `new_edges`, `total_nodes`, `total_edges`, `message` |
| `ChatMessage` | Nested | `ChatRequest` | `role: "user" \| "assistant"`, `content: str` |
| `ChatContext` | Nested | `ChatRequest` | `nodes`, `edges`, `pathway: Optional[str]` |
| `ChatRequest` | Request | `POST /api/chat` | `message: str`, `history: List[ChatMessage]`, `context: ChatContext` |
| `ChatResponse` | Response | `POST /api/chat` | `content: str` |

`AddTestResponse.new_nodes` / `new_edges` contain only the nodes and edges added in the current
pipeline run, so the frontend can append them to the existing visualisation without a full graph reload.
