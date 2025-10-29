from crewai.tools import BaseTool
from typing import Type
from pydantic import BaseModel, Field, field_validator  #ConfigDict
from contextual_rag.application.rag.rag import get_rag_answer

class RagToolInput(BaseModel):
    """Input schema for RAG Tool.

    Accepts either a plain string or a dict-shaped payload coming from
    hierarchical manager prompts (e.g., {"description": ..., "type": "str"}).
    Coerces input to string before passing to the tool runtime.
    """
    query: object = Field(..., description="Query to search the document.")

    @field_validator("query", mode="before")
    @classmethod
    def coerce_query_to_string(cls, value):
        if isinstance(value, dict) and "description" in value:
            return value["description"]
        return str(value)

class RagTool(BaseTool):
    name: str = "RagTool"
    description: str = "Search the document for the given query."
    args_schema: Type[BaseModel] = RagToolInput

    def __init__(self):
        super().__init__()
       
    
    async def _run(self, query: str) -> str:
        """Search the document for context and generate answer"""
        if isinstance(query, dict) and 'description' in query:
            query_string = query['description']
        else:
            query_string = str(query)
             
        try:
           results = await get_rag_answer(query_string)
        #    response = f"""#Answer: 
        #    \n
        #    {results.answer}
        #    \n\n
        #    # Context: 
        #    {'\n'.join(results.retrieved_contexts)}
        #     \n\n
        #     #Document Sources:
        #     {','.join(results.sources)}
        #    """
           return results.answer  
        except Exception as e:
            return f"Error in enhanced search: {str(e)}"
