import logging
import os
import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.routers import graph as graph_router

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Absolute path to the knowledge-graphs directory (used for S3 sync paths)
_KG_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "knowledge-graphs")
)
_PKL_PATH = os.path.join(_KG_DIR, "triage_knowledge_graph.pkl")
_RULES_PATH = os.path.join(_KG_DIR, "guideline_rules.json")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # On startup: pull PKL + rules from S3 if local copies are absent.
    # This lets a fresh Nebius container recover the latest graph automatically.
    if os.environ.get("S3_BUCKET"):
        if _KG_DIR not in sys.path:
            sys.path.insert(0, _KG_DIR)
        from s3_service import sync_from_s3_if_missing
        sync_from_s3_if_missing(_PKL_PATH, _RULES_PATH)
    yield
    # On shutdown: close Neo4j driver if it was opened.
    if os.environ.get("USE_NEO4J", "false").lower() == "true":
        from backend.services import neo4j_service
        neo4j_service.close()


app = FastAPI(title="Diagnotix API", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",  # Vite dev server
        "http://localhost:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(graph_router.router, prefix="/api", tags=["knowledge-graph"])


@app.get("/health")
async def health():
    return {"status": "ok"}
