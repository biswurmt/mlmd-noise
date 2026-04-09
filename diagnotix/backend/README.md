# diagnotix/backend

FastAPI application. Entry point is `main.py`.

```
backend/
├── main.py         # Creates FastAPI app, configures CORS, registers routers
├── routers/        # API endpoint handlers (see routers/README.md)
├── services/       # Business logic — LLM calls, graph loading, Semantic Scholar (see services/README.md)
└── models/         # Pydantic request/response schemas (see models/README.md)
```

`main.py` loads credentials from `knowledge-graphs/.env` and registers two routers:
- `graph_router` (`routers/graph.py`) — graph queries and add-test pipeline
- `chat_router` (`routers/chat.py`) — clinical chat

Start the server from the `diagnotix/` directory:

```bash
uvicorn backend.main:app --reload --port 8000
```
