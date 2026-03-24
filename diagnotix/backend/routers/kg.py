from fastapi import APIRouter, HTTPException

from backend.models.schemas import AddTestRequest, AddTestResponse
from backend.services.kg_service import add_test

router = APIRouter()


@router.post("/add_test", response_model=AddTestResponse)
async def add_test_route(request: AddTestRequest):
    """Accept a diagnostic test name, invoke the LLM to generate triage rules,
    append them to guideline_rules.json, and rebuild the knowledge graph."""
    try:
        return await add_test(request.diagnostic_test)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
