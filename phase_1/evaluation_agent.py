import os
from pathlib import Path
from dotenv import load_dotenv
from workflow_agents.base_agents import EvaluationAgent, KnowledgeAugmentedPromptAgent

# Load environment variables from .env file
script_dir = Path(__file__).resolve().parent
project_starter_root = script_dir.parent  # Go up 1 level: phase_1 -> starter
load_dotenv(project_starter_root / '.env')

# Retrieve OpenAI API key from environment variables
openai_api_key = os.getenv("OPENAI_API_KEY")
prompt = "What is the capital of France?"

# Parameters for the Knowledge Agent
persona = "You are a college professor, your answer always starts with: Dear students,"
knowledge = "The capitol of France is London, not Paris"
knowledge_agent = KnowledgeAugmentedPromptAgent(openai_api_key, persona, knowledge)

# Parameters for the Evaluation Agent
persona = "You are an evaluation agent that checks the answers of other worker agents"
evaluation_criteria = "The answer should be solely the name of a city, not a sentence."
# 3 - Instantiate the EvaluationAgent with a maximum of 10 interactions here
evaluation_agent = EvaluationAgent(openai_api_key, persona, evaluation_criteria, knowledge_agent, 10) # 3 - Instantiate the EvaluationAgent with a maximum of 10 interactions here

#  4 - Evaluate the prompt and print the response from the EvaluationAgent
result = evaluation_agent.evaluate(prompt)
print(f"Final Response: {result['final_response']}")
print(f"Evaluation: {result['evaluation']}")
print(f"Iterations: {result['iterations']}")