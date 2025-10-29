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
        

        all_u_msg = [m for m in reversed(request.messages) if m.role == "user"]
        if len(all_u_msg) == 0:
            raise HTTPException(status_code=400, detail="No user message found")

        if len(all_u_msg) == 1:
            user_msg = all_u_msg[0]
            query = f"Query: {user_msg.content}"

        elif len(all_u_msg) > 3:
            # take the last 3 messages
            recent_msgs = all_u_msg[:3]
            user_msg = recent_msgs[0]
            conversation_text = "\n".join(m.content for m in recent_msgs[1:])
            query = f"""
            Query: {user_msg.content}

            Previous Conversation:
            {conversation_text}
            """
        else:
            user_msg = all_u_msg[0]
            conversation_text = "\n".join(m.content for m in all_u_msg[1:])
            query = f"""
            Query: {user_msg.content}

            Previous Conversation:
            {conversation_text}
            """


        query = Query(query=query, evaluate=True)
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
