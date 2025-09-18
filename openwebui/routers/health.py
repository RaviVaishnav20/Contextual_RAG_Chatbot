from fastapi import APIRouter
from utils.helpers import check_ollama_health, check_postgres_health, check_phoenix_health, check_rag_pipeline_health
from openwebui.models import HealthStatus
from datetime import datetime
import asyncio

router = APIRouter()

@router.get("/")
async def health_check():
    return {"api": "healthy", "timestamp": datetime.now().isoformat()}

@router.get("/detailed")
async def detailed_health_check():
    results = await asyncio.gather(
        check_ollama_health(),
        check_postgres_health(),
        check_phoenix_health(),
        check_rag_pipeline_health(),
        return_exceptions=True
    )
    return {"services": [r.dict() if not isinstance(r, Exception) else str(r) for r in results]}

@router.get("/ollama")
async def ollama(): return (await check_ollama_health()).dict()

@router.get("/postgres")
async def postgres(): return (await check_postgres_health()).dict()

@router.get("/phoenix")
async def phoenix(): return (await check_phoenix_health()).dict()

@router.get("/rag")
async def rag(): return (await check_rag_pipeline_health()).dict()
