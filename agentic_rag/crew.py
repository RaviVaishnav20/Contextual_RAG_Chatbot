import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]  # go up 1 more level (project root)

if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from crewai.project import CrewBase, agent, task, crew
from crewai_tools import SerperDevTool
from crewai import Agent, Task, Crew, Process
from agentic_rag.tools.custom_tools import CustomRAGTool
from config.config_manager import ConfigManager
from langchain_ollama import ChatOllama
import os
from dotenv import load_dotenv

load_dotenv()
os.getenv("SERPER_API_KEY")
llm = ChatOllama(
    model=f"ollama/gemma3", base_url=os.getenv("OLLAMA_HOST", "http://localhost:11434")
)
config_path = ROOT/"config"/"config.yaml"
config = ConfigManager(config_path)
rag_tool = CustomRAGTool(config)
web_search_tool = SerperDevTool()


@CrewBase
class AgenticRag:

    

    def __init__(self):
        self.rag_tool_instance = rag_tool # Store the rag_tool instance
        agent_config = ROOT/"agentic_rag"/"config"/"agents.yaml"
        tasks_config = ROOT/"agentic_rag"/"config"/"tasks.yaml"

    @agent
    def retriever_agent(self) -> Agent:
        return Agent(
            config=self.agents_config['retriever_agent'],
            verbose=True,
            tools=[
                self.rag_tool_instance, # Use the stored instance
                web_search_tool
            ],
            llm=llm,
            max_retry_limit=2,
            max_iter=2
        )
    
    @agent
    def response_synthesizer_agent(self) -> Agent:
        return Agent(
            config=self.agents_config['response_synthesizer_agent'],
            verbose=True,
            llm=llm,
            max_retry_limit=2,
            max_iter=2
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
            verbose=True
        )
    
    def run_crew_with_context(self, query: str) -> dict:
        """Runs the AgenticRag crew and returns the retrieved context and final response."""
        inputs = {'query': query}
        final_response = self.crew().kickoff(inputs=inputs)
        retrieved_context = self.rag_tool_instance.get_last_retrieved_chunks()
        return {"context": retrieved_context, "response": final_response}
