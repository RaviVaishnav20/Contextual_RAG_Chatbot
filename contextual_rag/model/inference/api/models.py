from typing import Optional, Dict, Any, List, Literal, Tuple
from pydantic import BaseModel, Field
from contextual_rag.application.rag.rag_model import RerankedOutput
# Request/Response models
class Query(BaseModel):
    query: str
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    evaluate: bool = Field(default=False, description="Whether to run RAGAS evaluation and tracing")
    reference_answer: Optional[str] = Field(default="", description="Reference answer for evaluation")

class RAGResponse(BaseModel):
    retrieved_text: list
    llm_response: str
    sources: list
    response_time: float
    session_id: str
    query_id: str
    evaluation: Optional[Dict[str, Any]] = None
    phoenix_trace_id: Optional[str] = None

class AgenticResponse(BaseModel):
    response: str
    response_time: float
    session_id: str
    query_id: str
    evaluation: Optional[Dict[str, Any]] = None
    phoenix_trace_id: Optional[str] = None

class ChunksResponse(BaseModel):
    retrieved_text: List[RerankedOutput]
    response_time: float
    session_id: str
    query_id: str

class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str

class ChatCompletionRequest(BaseModel):
    model: Optional[str] = Field(default="contextual-rag")
    messages: List[ChatMessage]

class HealthStatus(BaseModel):
    service: str
    status: str
    details: Optional[Dict[str, Any]] = None
    response_time: Optional[float] = None
    error: Optional[str] = None
