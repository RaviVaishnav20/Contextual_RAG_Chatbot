import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]  # go up 1 more level (project root)

if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
    
from fastapi import FastAPI
from pydantic import BaseModel
from agentic_rag.rag import get_rag_response, get_relevant_chunks
from agentic_rag.crew import AgenticRag # Import AgenticRag

app = FastAPI()

class Query(BaseModel):
    query: str

@app.post("/rag")
async def rag_endpoint(query: Query):
    response = await get_rag_response(query.query)
    return response

@app.post("/agentic_rag")
async def crewai_endpoint(query: Query):
    agentic_rag = AgenticRag()
    response = agentic_rag.run_crew_with_context(query.query)
    return response

@app.post("/relevant_chunks")
async def relevant_chunks_endpointt_chunks(query: Query):
    return await get_relevant_chunks(query.query)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)