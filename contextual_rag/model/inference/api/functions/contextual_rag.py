"""
Contextual RAG Function for Open WebUI
Integrates with the Contextual RAG ChatBot backend
"""

import requests
import json
import asyncio
from typing import Dict, Any, List, Optional
from pydantic import BaseModel

class Function:
    """
    Contextual RAG integration for Open WebUI
    Provides intelligent document search and retrieval
    """
    
    class Valves(BaseModel):
        rag_api_url: str = "http://rag_app:8000"
        default_mode: str = "agentic"  # "basic" or "agentic"
        enable_evaluation: bool = True
        auto_search_threshold: float = 0.7  # Threshold for automatic RAG search
        
    def __init__(self):
        self.valves = self.Valves()
    
    async def pipe(
        self, body: dict, __user__: dict, __event_emitter__=None, __task__: str = None
    ) -> str:
        """
        Process user messages and enhance with RAG when appropriate
        """
        messages = body.get("messages", [])
        if not messages:
            return body
        
        # Get the last user message
        last_message = messages[-1]
        if last_message.get("role") != "user":
            return body
            
        query = last_message.get("content", "")
        
        # Determine if we should use RAG
        should_use_rag = self._should_use_rag(query)
        
        if should_use_rag:
            if __event_emitter__:
                await __event_emitter__({
                    "type": "status",
                    "data": {"description": "Searching knowledge base...", "done": False}
                })
            
            try:
                # Perform RAG search
                rag_result = await self._perform_rag_search(query, __user__.get("id"))
                
                if rag_result.get("status") == "success":
                    # Enhance the message with RAG context
                    enhanced_content = self._format_rag_response(query, rag_result)
                    last_message["content"] = enhanced_content
                    
                    if __event_emitter__:
                        await __event_emitter__({
                            "type": "status", 
                            "data": {"description": "Knowledge retrieved successfully", "done": True}
                        })
                else:
                    if __event_emitter__:
                        await __event_emitter__({
                            "type": "status",
                            "data": {"description": "Knowledge search failed", "done": True}
                        })
                        
            except Exception as e:
                if __event_emitter__:
                    await __event_emitter__({
                        "type": "status",
                        "data": {"description": f"RAG error: {str(e)}", "done": True}
                    })
        
        return body
    
    def _should_use_rag(self, query: str) -> bool:
        """Determine if query should trigger RAG search"""
        rag_triggers = [
            "what is", "explain", "tell me about", "how to", "define",
            "describe", "find information", "search for", "look up",
            "procedure", "process", "standard", "policy", "guideline"
        ]
        
        query_lower = query.lower()
        return any(trigger in query_lower for trigger in rag_triggers)
    
    async def _perform_rag_search(self, query: str, user_id: str) -> Dict[str, Any]:
        """Perform RAG search using the backend API"""
        try:
            endpoint = "/agentic_rag" if self.valves.default_mode == "agentic" else "/rag"
            
            payload = {
                "query": query,
                "user_id": user_id,
                "evaluate": self.valves.enable_evaluation
            }
            
            response = requests.post(
                f"{self.valves.rag_api_url}{endpoint}",
                json=payload,
                timeout=60
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                return {"status": "error", "message": f"API error: {response.status_code}"}
                
        except requests.exceptions.Timeout:
            return {"status": "error", "message": "Search timed out"}
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    def _format_rag_response(self, original_query: str, rag_result: Dict[str, Any]) -> str:
        """Format RAG response for display"""
        answer = rag_result.get("response", rag_result.get("llm_response", ""))
        response_time = rag_result.get("response_time", 0)
        mode = rag_result.get("mode", self.valves.default_mode)
        
        formatted = f"""Based on your question: "{original_query}"

{answer}

---
📊 Search completed in {response_time:.2f}s using {mode} mode
🔍 Phoenix Trace: {rag_result.get('phoenix_trace_id', 'N/A')}"""

        if rag_result.get("evaluation"):
            eval_info = rag_result["evaluation"]
            if eval_info.get("status") == "evaluating":
                formatted += "\n📈 Quality evaluation in progress..."
        
        return formatted
