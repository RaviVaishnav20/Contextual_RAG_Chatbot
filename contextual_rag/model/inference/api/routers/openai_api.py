from fastapi import APIRouter, HTTPException, BackgroundTasks
from contextual_rag.model.inference.api.models import ChatCompletionRequest, Query
from contextual_rag.model.inference.api.routers.rag import agentic_rag_endpoint
import time, traceback

router = APIRouter()

@router.get("/models")
async def list_models():
    now = int(time.time())
    return {"object": "list", "data": [{"id": "contextual-rag", "object": "model", "created": now}]}

@router.post("/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    try:
        user_msg = next((m for m in reversed(request.messages) if m.role == "user"), None)
        if not user_msg:
            raise HTTPException(status_code=400, detail="No user message found")
        
        query = Query(query=user_msg.content, evaluate=False)
        response = await agentic_rag_endpoint(query, BackgroundTasks())

        return {
            "id": f"chatcmpl-{response.query_id}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": request.model or "contextual-rag",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": response.response}}],
            "usage": {
                "prompt_tokens": len(user_msg.content.split()),
                "completion_tokens": len(response.response.split()),
                "total_tokens": len(user_msg.content.split()) + len(response.response.split())
            }
        }
    except Exception as e:
        print(f"Chat Completion Error: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail="Chat completion failed")
