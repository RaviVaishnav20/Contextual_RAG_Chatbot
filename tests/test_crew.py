from contextual_rag.application.agents.crew.crew import AgenticRag

if __name__=="__main__":
    query = "Definitions, scope of application, and delegation of powers"
    agentic_rag = AgenticRag()

    reponse = agentic_rag.run_crew_with_context(query)
    print(reponse)