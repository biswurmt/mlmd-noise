from fastapi import APIRouter, HTTPException

from backend.models.schemas import ExpandRequest, ExpandResponse
from backend.services.kg_service import expand_knowledge_graph

router = APIRouter()


@router.post("/expand", response_model=ExpandResponse)
async def expand_kg(request: ExpandRequest):
    """Accept a diagnostic test name, invoke the LLM to generate triage rules,
    append them to guideline_rules.json, and rebuild the knowledge graph."""
    try:
        return await expand_knowledge_graph(request.diagnostic_test)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
