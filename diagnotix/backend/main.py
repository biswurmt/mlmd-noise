import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.routers import graph as graph_router

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Diagnotix API", version="1.0.0")

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
