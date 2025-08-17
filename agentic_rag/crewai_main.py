import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]  # go up 1 more level (project root)

if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
import warnings

from agentic_rag.crew import AgenticRag
import asyncio
warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")

def run():
    """
    Run the crew
    """
    inputs = {
        'query': 'What is benefit tracking related to procurement?'#'Who is Elon mask ?'
    }
    result = AgenticRag().crew().kickoff(inputs=inputs)
    print(result)

if __name__ == "__main__":
    run()