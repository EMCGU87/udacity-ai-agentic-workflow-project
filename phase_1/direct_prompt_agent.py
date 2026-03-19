# Test script for DirectPromptAgent class

from workflow_agents.base_agents import DirectPromptAgent
import os
from dotenv import load_dotenv

# Load environment variables from .env file
from pathlib import Path
script_dir = Path(__file__).resolve().parent
project_starter_root = script_dir.parent  # Go up 1 level: phase_1 -> starter
load_dotenv(project_starter_root / '.env')

# Load the OpenAI API key from the environment variables
openai_api_key = os.getenv("OPENAI_API_KEY")

prompt = "What is the Capital of France?"

# Instantiate the DirectPromptAgent as direct_agent
direct_agent = DirectPromptAgent(openai_api_key)
# Use direct_agent to send the prompt defined above and store the response
direct_agent_response = direct_agent.respond(prompt) 

# Print the response from the agent
print(direct_agent_response)

#  Print an explanatory message describing the knowledge source used by the agent to generate the response
print("Knowledge source: This DirectPromptAgent relies solely on the LLM's pre-trained general knowledge base. It does not use any external data sources, specialized knowledge, or context augmentation.")
