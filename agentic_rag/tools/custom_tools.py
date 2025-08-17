import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]  # go up 1 more level (project root)

if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
    
from crewai.tools import BaseTool
from typing import Type
from pydantic import BaseModel, Field, ConfigDict

from rag_pipeline.embedding_storage import load_existing_index, query_embeddings, setup_ollama_embeddings
from config.config_manager import ConfigManager


class CustomRAGToolInput(BaseModel):
    """Input schema for DocumentSearchTool."""
    query: str = Field(..., description="Query to search the document.")

class CustomRAGTool(BaseTool):
    name: str = "CustomRAGTool"
    description: str = "Search the document for the given query."
    args_schema: Type[BaseModel] = CustomRAGToolInput
    
    model_config = ConfigDict(extra="allow")
    
    _last_retrieved_chunks: str = "" # Added to store the last retrieved chunks

    def __init__(self, config: ConfigManager):
        super().__init__()
        self.config = config
        setup_ollama_embeddings(config)
        self.index = load_existing_index(config)
    
    def _run(self, query: str) -> str:
        """Search the document with a query string."""
        if isinstance(query, dict) and 'description' in query:
            query_string = query['description']
        else:
            query_string = str(query)
        try:
            response = query_embeddings(self.index, query, self.config)
            
            if response and response.source_nodes:
                retrieved_chunks_list = []
                for i, node in enumerate(response.source_nodes):
                    source = node.metadata.get('source', 'Unknown')
                    chunk_id = node.metadata.get('chunk_id', 'N/A')
                    retrieved_chunks_list.append(
                        f"Source: {source} (Chunk {chunk_id})\n{node.text}\n"
                    )
                self._last_retrieved_chunks = "\n---\n".join(retrieved_chunks_list)
                return self._last_retrieved_chunks
            else:
                self._last_retrieved_chunks = "No relevant documents found in the knowledge base."
                return self._last_retrieved_chunks
                
        except Exception as e:
            self._last_retrieved_chunks = f"Error searching documents: {str(e)}"
            return self._last_retrieved_chunks

    def get_last_retrieved_chunks(self) -> str:
        return self._last_retrieved_chunks
   
# Test the implementation
def test_document_searcher():
    # Test file path
    config_path = "config.yaml"
    config = ConfigManager(config_path)
    # rag_config = config.get_rag_config()
    # Create instance
    searcher = CustomRAGTool(config)
    
    # Test search
    result = searcher._run("What is the purpose of benefit tracking?")
    print("Search Results:", result)

if __name__ == "__main__":
    test_document_searcher()