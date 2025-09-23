"""
Custom Functions for Open WebUI to integrate with Contextual RAG ChatBot
These functions will be automatically loaded by Open WebUI
"""

import json
import asyncio
import requests
from typing import Dict, Any, List, Optional
from datetime import datetime
import aiohttp

class Tools:
    def __init__(self):
        self.rag_base_url = "http://rag_app:8000"  # Docker service name
        self.phoenix_url = "http://phoenix:6006"
        
    def get_tools(self) -> List[Dict[str, Any]]:
        """Return available tools for Open WebUI"""
        return [
            {
                "name": "contextual_rag_search",
                "description": "Search through the contextual RAG knowledge base for relevant information",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The search query to find relevant information"
                        },
                        "mode": {
                            "type": "string",
                            "enum": ["basic", "agentic"],
                            "default": "agentic",
                            "description": "Search mode: basic RAG or agentic multi-agent search"
                        },
                        "evaluate": {
                            "type": "boolean",
                            "default": False,
                            "description": "Whether to run RAGAS evaluation on the response"
                        }
                    },
                    "required": ["query"]
                }
            },
            {
                "name": "get_relevant_chunks",
                "description": "Retrieve raw relevant document chunks without LLM generation",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The search query to find relevant chunks"
                        }
                    },
                    "required": ["query"]
                }
            },
            {
                "name": "view_phoenix_traces",
                "description": "Get Phoenix observability traces for the current session",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "limit": {
                            "type": "integer",
                            "default": 10,
                            "description": "Number of recent traces to retrieve"
                        }
                    }
                }
            }
        ]

    async def contextual_rag_search(
        self, 
        query: str, 
        mode: str = "agentic", 
        evaluate: bool = False,
        **kwargs
    ) -> Dict[str, Any]:
        """Search using contextual RAG with optional evaluation"""
        try:
            endpoint = "/agentic_rag" if mode == "agentic" else "/rag"
            
            payload = {
                "query": query,
                "user_id": kwargs.get("user_id"),
                "session_id": kwargs.get("session_id"),
                "evaluate": evaluate
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.rag_base_url}{endpoint}",
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=60)
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        
                        # Format response for Open WebUI
                        formatted_response = {
                            "status": "success",
                            "answer": result.get("response", result.get("llm_response", "")),
                            "context_used": result.get("context", result.get("retrieved_text", "")),
                            "response_time": result.get("response_time", 0),
                            "phoenix_trace_id": result.get("phoenix_trace_id"),
                            "mode": mode
                        }
                        
                        if evaluate and result.get("evaluation"):
                            formatted_response["evaluation"] = result["evaluation"]
                        
                        return formatted_response
                    else:
                        error_text = await response.text()
                        return {
                            "status": "error", 
                            "message": f"RAG API error: {response.status} - {error_text}"
                        }
                        
        except asyncio.TimeoutError:
            return {
                "status": "error",
                "message": "RAG search timed out. Please try again."
            }
        except Exception as e:
            return {
                "status": "error",
                "message": f"RAG search failed: {str(e)}"
            }

    async def get_relevant_chunks(self, query: str, **kwargs) -> Dict[str, Any]:
        """Get relevant document chunks without LLM processing"""
        try:
            payload = {
                "query": query,
                "user_id": kwargs.get("user_id"),
                "session_id": kwargs.get("session_id")
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.rag_base_url}/relevant_chunks",
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        
                        return {
                            "status": "success",
                            "chunks": result.get("retrieved_text", ""),
                            "response_time": result.get("response_time", 0),
                            "chunk_count": len(result.get("retrieved_text", "").split("--- Chunk"))
                        }
                    else:
                        error_text = await response.text()
                        return {
                            "status": "error",
                            "message": f"Chunks API error: {response.status} - {error_text}"
                        }
                        
        except Exception as e:
            return {
                "status": "error",
                "message": f"Chunks retrieval failed: {str(e)}"
            }

    async def view_phoenix_traces(self, limit: int = 10, **kwargs) -> Dict[str, Any]:
        """Retrieve Phoenix observability traces"""
        try:
            # This would need to be implemented based on Phoenix API
            # For now, return a placeholder
            return {
                "status": "success",
                "message": f"Phoenix traces available at: {self.phoenix_url}",
                "trace_url": f"{self.phoenix_url}/projects/contextual_rag_chatbot"
            }
        except Exception as e:
            return {
                "status": "error",
                "message": f"Phoenix traces retrieval failed: {str(e)}"
            }


# Global tools instance
tools = Tools()

def get_tools():
    """Main function called by Open WebUI"""
    return tools.get_tools()

async def call_tool(name: str, parameters: Dict[str, Any], **kwargs) -> Dict[str, Any]:
    """Main function to execute tools called by Open WebUI"""
    if name == "contextual_rag_search":
        return await tools.contextual_rag_search(**parameters, **kwargs)
    elif name == "get_relevant_chunks":
        return await tools.get_relevant_chunks(**parameters, **kwargs)
    elif name == "view_phoenix_traces":
        return await tools.view_phoenix_traces(**parameters, **kwargs)
    else:
        return {
            "status": "error",
            "message": f"Unknown tool: {name}"
        }

# Additional Open WebUI specific functions
class Function:
    def __init__(self):
        self.type = "function"
    
    class Valves:
        rag_api_url: str = "http://rag_app:8000"
        phoenix_url: str = "http://phoenix:6006"
        enable_evaluation: bool = True
        default_mode: str = "agentic"
        
    def __init__(self):
        self.valves = self.Valves()
    
    async def pipe(self, body: dict, **kwargs) -> dict:
        """Process messages through RAG system"""
        messages = body.get("messages", [])
        if not messages:
            return body
        
        # Get the last user message
        user_message = next((msg for msg in reversed(messages) if msg.get("role") == "user"), None)
        if not user_message:
            return body
        
        query_text = user_message.get("content", "")
        
        # Check if user wants to use RAG
        if any(trigger in query_text.lower() for trigger in ["search", "find", "what is", "explain", "tell me about"]):
            try:
                # Use RAG to enhance the response
                rag_result = await tools.contextual_rag_search(
                    query=query_text,
                    mode=self.valves.default_mode,
                    evaluate=self.valves.enable_evaluation
                )
                
                if rag_result["status"] == "success":
                    # Add RAG context to the conversation
                    enhanced_message = f"""Based on the knowledge base search:

{rag_result['answer']}

Source Context: {rag_result['context_used'][:500]}...

Response Time: {rag_result['response_time']:.2f}s | Mode: {rag_result['mode']}"""
                    
                    # Replace the user message with enhanced version
                    user_message["content"] = enhanced_message
                    
                    # Add metadata for tracking
                    if "metadata" not in body:
                        body["metadata"] = {}
                    body["metadata"]["rag_enhanced"] = True
                    body["metadata"]["phoenix_trace"] = rag_result.get("phoenix_trace_id")
                    
            except Exception as e:
                # If RAG fails, add a note but don't break the conversation
                user_message["content"] += f"\n\n[Note: Knowledge base search unavailable: {str(e)}]"
        
        return body


# Export for Open WebUI
__all__ = ["get_tools", "call_tool", "Function"]