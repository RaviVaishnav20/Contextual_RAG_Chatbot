from contextual_rag.infrastructure.config_manager import ConfigManager
from crewai.project import CrewBase, agent, task, crew
from crewai import Agent, Task, Crew, Process, LLM
from contextual_rag.application.agents.crew.tools.general_tool import GeneralConversationTool
from contextual_rag.application.agents.crew.tools.rag_tool import RagTool
from contextual_rag.settings import settings




cm = ConfigManager()
crew_cfg = cm.get_crewai_config() or {}

primary_model = crew_cfg.get('model', {}).get('primary_model_name', 'llama3:8b')

llm = LLM(model=f"ollama/{primary_model}", base_url=cm.get_ollama_host())

general_tool = GeneralConversationTool()
rag_tool = RagTool()

@CrewBase
class AgenticRag:

    def __init__(self):
        self.general_tool_instance = general_tool
        self.rag_tool_instance = rag_tool # Store the rag_tool instance
        crew_dir = settings.crew_dir
        self.agent_config = crew_dir/"config"/"agents.yaml"
        self.tasks_config = crew_dir/"config"/"tasks.yaml"
        self.agents_config = cm.load_yaml(self.agent_config)
        self.tasks_config = cm.load_yaml(self.tasks_config)

        # print(" self.agent_config")
        # print(self.agent_config)
    # def __init__(self, memory_type: str = "faiss", memory_path: str = settings.crew_memory_dir):
    #     self.general_tool_instance = general_tool
    #     self.rag_tool_instance = rag_tool
        
    #     # Initialize custom memory storage
    #     if memory_type.lower() == "faiss":
    #         self.memory_storage = LocalFAISSStorage(memory_path)
    #     elif memory_type.lower() == "pickle":
    #         self.memory_storage = PickleMemoryStorage(memory_path)
    #     else:
    #         raise ValueError(f"Unsupported memory type: {memory_type}")
        
    #     # Create custom short-term memory
    #     self.short_term_memory = ShortTermMemory(storage=self.memory_storage)
        
    #     crew_dir = settings.crew_dir
    #     self.agent_config = crew_dir/"config"/"agents.yaml"
    #     self.tasks_config = crew_dir/"config"/"tasks.yaml"
        
    #     print(f"🧠 Initialized {memory_type.upper()} memory storage at: {memory_path}")
    #     print(f"📊 Memory stats: {self.memory_storage.get_stats()}")

    # @agent
    # def general_agent(self) -> Agent:
    #     """Agent that handles general conversation queries directly"""
    #     return Agent(
    #         config=self.agents_config['general_agent'],
    #         verbose=True,
    #         tools=[self.general_tool_instance],
    #         llm=llm,
    #         max_retry_limit=1,
    #         max_iter=1,
    #     )


    @agent
    def retriever_agent(self) -> Agent:
        """Agent that retrieves relevant information for document queries"""
        return Agent(
            config=self.agents_config['retriever_agent'],
            verbose=True,
            tools=[self.general_tool_instance, self.rag_tool_instance],
            llm=llm,
            max_retry_limit=1,
            max_iter=1,
        )

    
    @agent
    def response_synthesizer_agent(self) -> Agent:
        return Agent(
            config=self.agents_config['response_synthesizer_agent'],
            verbose=True,
            llm=llm,
            max_retry_limit=1,
            max_iter=1,
        )

    
    # @task
    # def general_task(self) -> Task:
    #     """Task for general conversation queries"""
    #     return Task(
    #         config=self.tasks_config['general_task'],
    #     )

    @task
    def retrieval_task(self) -> Task:
        """Task for heneral converstion and retrieving knowledge from documents"""
        return Task(
            config=self.tasks_config['retrieval_task'],
        )

    @task
    def response_task(self) -> Task:
        """Final response synthesis task"""
        return Task(
            config=self.tasks_config['response_task'],
        )

    
    @crew
    def crew(self) -> Crew:
        """Creates the AgenticRag crew"""
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential, #Process.sequential,
            manager_llm=llm,
            # memory=True,  # Enable basic memory system
            # short_term_memory=self.short_term_memory,  # Use custom short-term memor
            verbose=True
        )
    
    def run_crew(self, query: str):
        """Runs the AgenticRag crew and returns the final response."""
        query_string = query['description'] if isinstance(query, dict) else str(query)
        
        # Simple routing logic
        # if self.is_general_query(query_string):
        #     tasks = [self.general_task(), self.response_task()]
        #     agents = [self.general_agent(), self.response_synthesizer_agent()]
        # else:
        #     tasks = [self.retrieval_task(), self.response_task()]
        #     agents = [self.retriever_agent(), self.response_synthesizer_agent()]
        final_response = self.crew().kickoff(inputs={'query': query_string})
        return final_response

    # def run_crew(self, query: str):
    #     """Runs the AgenticRag crew and returns the final response"""
    #     if isinstance(query, dict) and 'description' in query:
    #         query_string = query['description']
    #     else:
    #         query_string = str(query)
        
    #     inputs = {'query': query}
        
    #     # Save the query to memory before processing
    #     self.memory_storage.save(
    #         value=f"User Query: {query_string}",
    #         metadata={"type": "user_query", "timestamp": datetime.now().isoformat()},
    #         agent="user"
    #     )
        
    #     # Run the crew
    #     final_response = self.crew().kickoff(inputs=inputs)
        
    #     # Save the response to memory
    #     self.memory_storage.save(
    #         value=f"Crew Response: {str(final_response)}",
    #         metadata={"type": "crew_response", "query": query_string},
    #         agent="crew"
    #     )
        
    #     return final_response
    
    # def search_memory(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
    #     """Search crew memory"""
    #     return self.memory_storage.search(query, limit=limit)
    
    # def get_memory_stats(self) -> Dict[str, Any]:
    #     """Get memory statistics"""
    #     return self.memory_storage.get_stats()
    
    # def reset_memory(self):
    #     """Reset all memory data"""
    #     self.memory_storage.reset()
    
    # def export_memory(self, filepath: str):
    #     """Export memory for backup"""
    #     if hasattr(self.memory_storage, 'export_memories'):
    #         self.memory_storage.export_memories(filepath)