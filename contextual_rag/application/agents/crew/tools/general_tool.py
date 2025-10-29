from crewai.tools import BaseTool
from typing import Type
from pydantic import BaseModel, Field, field_validator
import re
from contextual_rag.infrastructure.config_manager import ConfigManager
from contextual_rag.application.networks.llm import generate_content

class GeneralToolInput(BaseModel):
    """Input schema for General Conversation Tool."""
    query: object = Field(..., description="General query or conversational input.")

    @field_validator("query", mode="before")
    @classmethod
    def coerce_query_to_string(cls, value):
        if isinstance(value, dict) and "description" in value:
            return value["description"]
        return str(value)

class GeneralConversationTool(BaseTool):
    name: str = "GeneralConversationTool"
    description: str = """Handle general conversational queries and greetings. Use this tool for:
    - Greetings and social interactions
    - General questions and explanations
    - Follow-up questions (crew memory handles context)
    - Conversational queries that don't require document search"""
    args_schema: Type[BaseModel] = GeneralToolInput

    def __init__(self):
        super().__init__()
    
    # PART 1: GREETING HANDLER
    def _is_greeting(self, query: str) -> bool:
        """Check if query is a greeting or simple social interaction"""
        greeting_patterns = [
            r'^\s*(hi|hello|hey|good morning|good afternoon|good evening)\s*$',
            r'^\s*(thank you|thanks|ok|okay|alright|bye|goodbye)\s*$',
            r'^\s*(how are you|what\'s up)\s*[\?]?\s*$',
        ]
        query_lower = query.lower().strip()
        return any(re.search(pattern, query_lower) for pattern in greeting_patterns)
    
    def _handle_greeting(self, query: str) -> str:
        """Handle greetings and simple social interactions"""
        query_lower = query.lower().strip()
        
        # Greetings
        if re.search(r'^\s*(hi|hello|hey)\s*$', query_lower):
            return "Hello! How can I help you today?"
        elif re.search(r'^\s*good\s+(morning|afternoon|evening)\s*$', query_lower):
            return "Good day! What can I assist you with?"
        elif re.search(r'^\s*(how are you|what\'s up)\s*[\?]?\s*$', query_lower):
            return "I'm doing well, thank you for asking! How can I help you today?"
        
        # Acknowledgments
        elif re.search(r'^\s*(thank you|thanks)\s*$', query_lower):
            return "You're welcome! Is there anything else you'd like to know?"
        elif re.search(r'^\s*(ok|okay|alright)\s*$', query_lower):
            return "Great! What would you like to know or discuss?"
        
        # Goodbyes
        elif re.search(r'^\s*(goodbye|bye)\s*$', query_lower):
            return "Goodbye! Feel free to ask me anything anytime."
        
        return ""

    def _get_llm_config(self):
        """Get LLM configuration when needed"""
        cm = ConfigManager()
        crew_cfg = cm.get_crewai_config() or {}
        primary_provider = crew_cfg.get('model', {}).get('primary_provider', 'ollama')
        primary_model = crew_cfg.get('model', {}).get('primary_model_name', 'llama3:8b')
        return primary_provider, primary_model

    # PART 2: GENERAL QUERY HANDLER
    async def _handle_general_query(self, query: str) -> str:
        """Handle general queries using LLM (crew memory provides context)"""
        try:
            primary_provider, primary_model = self._get_llm_config()
            
            prompt = f"""You are a helpful AI assistant in a friendly conversation. You can engage in general conversation, answer questions, and provide explanations, but you should NOT search through documents.

User question: "{query}"

Guidelines:
- Keep responses conversational and friendly
- For general knowledge questions, provide helpful explanations
- If the question needs specific document information, suggest document search
- Keep responses concise but informative (2-3 sentences)
- Don't make up specific facts or data
- The crew has memory enabled, so context from previous messages is automatically available

Respond naturally and helpfully:"""

            answer = generate_content(
                provider=primary_provider,
                model_name=primary_model,
                prompt=prompt
            )
            return answer
        except Exception as e:
            return f"I'd be happy to help with that, but I encountered an issue: {str(e)}. Please try rephrasing your question."

    async def _run(self, query: str) -> str:
        """Main method that routes to greeting or general query handler"""
        # Handle input coercion
        if isinstance(query, dict) and 'description' in query:
            query_string = query['description']
        else:
            query_string = str(query)
        
        try:
            # PART 1: Check if it's a greeting first
            if self._is_greeting(query_string):
                greeting_response = self._handle_greeting(query_string)
                if greeting_response:
                    return greeting_response
            
            # PART 2: Handle as general query (crew memory provides context)
            return await self._handle_general_query(query_string)
                
        except Exception as e:
            return f"I apologize, but I encountered an error: {str(e)}. Please try asking your question again."