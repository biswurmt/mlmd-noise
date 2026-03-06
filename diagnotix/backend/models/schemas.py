from typing import Any, Dict, List, Optional

from pydantic import BaseModel


# ── Graph load (GET /api/graph) ───────────────────────────────────────────────

class GraphData(BaseModel):
    """Full graph payload returned by GET /api/graph."""
    nodes: List[Dict[str, Any]]
    edges: List[Dict[str, Any]]


# ── Add test (POST /api/add_test) ─────────────────────────────────────────────

class AddTestRequest(BaseModel):
    diagnostic_test: str


class AddTestResponse(BaseModel):
    """Response from POST /api/add_test.

    new_nodes / new_edges contain only the nodes and edges that did not exist
    in the graph before the pipeline was triggered, so the frontend can append
    them to the existing visualisation without a full reload.
    """
    success: bool
    diagnostic_test: str
    new_rules_added: int
    new_nodes: List[Dict[str, Any]]
    new_edges: List[Dict[str, Any]]
    total_nodes: int
    total_edges: int
    message: str
