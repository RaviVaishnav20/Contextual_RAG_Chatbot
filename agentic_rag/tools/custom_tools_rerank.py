import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
    
from crewai.tools import BaseTool
from typing import Type
from pydantic import BaseModel, Field, ConfigDict

from rag_pipeline.embedding_storage import load_existing_index, query_embeddings, setup_ollama_embeddings
from config.config_manager import ConfigManager
from agentic_rag.ollama_reranker import OllamaReRanker

class CustomRAGToolInput(BaseModel):
    """Input schema for Enhanced RAG Tool with re-ranking."""
    query: str = Field(..., description="Query to search the document.")

class CustomRAGTool(BaseTool):
    name: str = "CustomRAGTool"
    description: str = "Search documents with Ollama-based re-ranking for better relevance."
    args_schema: Type[BaseModel] = CustomRAGToolInput
    
    model_config = ConfigDict(extra="allow")
    
    _last_retrieved_chunks: str = ""

    def __init__(self, config: ConfigManager, rerank_model: str = "llama3.1:8b"):
        super().__init__()
        self.config = config
        setup_ollama_embeddings(config)
        self.index = load_existing_index(config)
        self.reranker = OllamaReRanker(rerank_model)
    
    def _run(self, query: str) -> str:
        """Search with re-ranking."""
        if isinstance(query, dict) and 'description' in query:
            query_string = query['description']
        else:
            query_string = str(query)
             
        try:
            # Step 1: Get initial results (more than usual)
            rag_config = self.config.get_rag_config()
            original_top_k = rag_config.get('parameters', {}).get('similarity_top_k', 5)
            rag_config['parameters']['similarity_top_k'] = 15  # Get more initially
            
            response = query_embeddings(self.index, query_string, self.config)
            
            # Restore original setting
            rag_config['parameters']['similarity_top_k'] = original_top_k
            
            if response and response.source_nodes:
                # Step 2: Extract documents and metadata
                documents = []
                metadata_list = []
                
                for node in response.source_nodes:
                    documents.append(node.text)
                    metadata_list.append({
                        'source': node.metadata.get('source', 'Unknown'),
                        'chunk_id': node.metadata.get('chunk_id', 'N/A')
                    })
                
                # Step 3: Re-rank documents
                reranked_results = self.reranker.rerank(query_string, documents, top_k=5)
                
                # Step 4: Format results
                retrieved_chunks_list = []
                for i, (doc, score) in enumerate(reranked_results):
                    # Find original metadata
                    original_idx = documents.index(doc)
                    metadata = metadata_list[original_idx]
                    
                    chunk_text = f"Source: {metadata['source']} (Chunk {metadata['chunk_id']}) - Score: {score:.3f}\n{doc}\n"
                    retrieved_chunks_list.append(chunk_text)
                
                self._last_retrieved_chunks = "\n---\n".join(retrieved_chunks_list)
                return self._last_retrieved_chunks
            else:
                self._last_retrieved_chunks = "No relevant documents found."
                return self._last_retrieved_chunks
                
        except Exception as e:
            self._last_retrieved_chunks = f"Error in enhanced search: {str(e)}"
            return self._last_retrieved_chunks

    def get_last_retrieved_chunks(self) -> str:
        return self._last_retrieved_chunks

# Test function
def test_enhanced_tool():
    config = ConfigManager()
    tool = CustomRAGTool(config)
    
    result = tool._run("What is the purpose of benefit tracking?")
    print("Enhanced Search Results:")
    print(result)

if __name__ == "__main__":
    test_enhanced_tool()