"""graph.py
==========
Router exposing:
  GET  /api/tests     — list all Diagnostic_Test nodes (for the sidebar)
  GET  /api/graph     — load the PKL, optionally filtered to one pathway
  POST /api/add_test  — run the LLM pipeline and return the new cluster diff
"""

from typing import List

from fastapi import APIRouter, HTTPException, Query

from backend.models.schemas import AddTestRequest, AddTestResponse, GraphData
from backend.services.graph_service import load_graph_json, load_test_nodes
from backend.services.kg_service import add_test as svc_add_test

router = APIRouter()


@router.get("/tests", response_model=List[dict])
async def get_tests():
    """Return a sorted list of {id, label} objects for every Diagnostic_Test
    node in the graph.  Used by the frontend sidebar to populate the pathway
    picker without loading the full graph.
    """
    try:
        return load_test_nodes()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/graph", response_model=GraphData)
async def get_graph(pathway: str | None = Query(None)) -> GraphData:
    """Return nodes and edges from the serialised knowledge graph.

    If *pathway* is supplied, only the subgraph for that diagnostic test
    is returned (server-side filtering).  Pass nothing or "All Pathways"
    to receive the full graph.

    Returns an empty graph (no error) if the PKL has not been built yet so
    the frontend can still render an empty canvas on first load.
    """
    try:
        data = load_graph_json(pathway=pathway)
        return GraphData(**data)
    except FileNotFoundError:
        return GraphData(nodes=[], edges=[])
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/add_test", response_model=AddTestResponse)
async def add_test(request: AddTestRequest) -> AddTestResponse:
    """Run the full extraction pipeline for *diagnostic_test* and return only
    the newly created nodes/edges so the frontend can append them without a
    full graph reload.
    """
    try:
        return await svc_add_test(request.diagnostic_test)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
