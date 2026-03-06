"""graph.py
==========
Router exposing:
  GET  /api/graph     — load the current PKL and return all nodes + edges
  POST /api/add_test  — run the LLM pipeline and return the new cluster diff
"""

from fastapi import APIRouter, HTTPException

from backend.models.schemas import AddTestRequest, AddTestResponse, GraphData
from backend.services.graph_service import load_graph_json
from backend.services.kg_service import add_test as svc_add_test

router = APIRouter()


@router.get("/graph", response_model=GraphData)
async def get_graph() -> GraphData:
    """Return all nodes and edges from the serialised knowledge graph.

    Returns an empty graph (no error) if the PKL has not been built yet so
    the frontend can still render an empty canvas on first load.
    """
    try:
        data = load_graph_json()
        return GraphData(**data)
    except FileNotFoundError:
        return GraphData(nodes=[], edges=[])
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
