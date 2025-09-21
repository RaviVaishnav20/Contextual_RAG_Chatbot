from contextual_rag.infrastructure.config_manager import ConfigManager
# from contextual_rag.infrastructure.llm import generate_content
from crewai.project import CrewBase, agent, task, crew
# from crewai_tools import SerperDevTool
from crewai import Agent, Task, Crew, Process
from contextual_rag.application.agents.crew.tools.rag_tool import RagTool
from contextual_rag.settings import settings
from langchain_ollama import ChatOllama
# import os
# from dotenv import load_dotenv
# load_dotenv()

cm = ConfigManager()
crew_cfg = cm.get_crewai_config() or {}

primary_provider = crew_cfg.get('model', {}).get('primary_provider', 'ollama')
primary_model = crew_cfg.get('model', {}).get('primary_model_name', 'llama3:8b')
fallback_provider = crew_cfg.get('model', {}).get('fallback_provider', 'gemini')
fallback_model = crew_cfg.get('model', {}).get('fallback_model_name', 'gemini-2.5-flash')

# crewai_config = config.get_crewai_config()
# model_name = crewai_config.get('model', {}).get('model_name', 'gemma3')
# os.getenv("SERPER_API_KEY")
llm = ChatOllama(
    model= f"ollama/{primary_model}", base_url=cm.get_ollama_host()
)

rag_tool = RagTool()
# web_search_tool = SerperDevTool()

# agent_config = cm.get_agents_config()
# tasks_config = cm.get_tasks_config()
@CrewBase
class AgenticRag:

    def __init__(self):
        self.rag_tool_instance = rag_tool # Store the rag_tool instance
        crew_dir = settings.crew_dir
        self.agent_config = crew_dir/"config"/"agents.yaml"
        self.tasks_config = crew_dir/"config"/"tasks.yaml"

        print(" self.agent_config")
        print(self.agent_config)
       

    @agent
    def retriever_agent(self) -> Agent:
        return Agent(
            config=self.agents_config['retriever_agent'],
            verbose=True,
            tools=[
                self.rag_tool_instance, # Use the stored instance
                # web_search_tool
            ],
            llm=llm,
            max_retry_limit=1,
            max_iter=2
        )
    
    @agent
    def response_synthesizer_agent(self) -> Agent:
        return Agent(
            config=self.agents_config['response_synthesizer_agent'],
            verbose=True,
            llm=llm,
            max_retry_limit=1,
            max_iter=1
        )
    
    @task
    def retrieval_task(self) -> Task:
        return Task(
            config=self.tasks_config['retrieval_task'],
            
        )
    
    @task
    def response_task(self) -> Task:
        return Task(
            config=self.tasks_config['response_task'],
        )
    
    @crew
    def crew(self) -> Crew:
        """Creates the AgenticRag crew"""
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            manager_llm=llm,
            verbose=True
        )
    
    def run_crew_with_context(self, query: str) -> dict:
        """Runs the AgenticRag crew and returns the final response."""
        if isinstance(query, dict) and 'description' in query:
            query_string = query['description']
        else:
            query_string = str(query)
        inputs = {'query': query}
        final_response = self.crew().kickoff(inputs=inputs)
        return final_response
