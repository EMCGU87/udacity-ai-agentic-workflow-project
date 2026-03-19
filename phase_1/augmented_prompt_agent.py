# Import the AugmentedPromptAgent class
import os
from pathlib import Path
from dotenv import load_dotenv
from workflow_agents.base_agents import AugmentedPromptAgent

# Load environment variables from .env file
script_dir = Path(__file__).resolve().parent
project_starter_root = script_dir.parent  # Go up 1 level: phase_1 -> starter
load_dotenv(project_starter_root / '.env')

# Retrieve OpenAI API key from environment variables
openai_api_key = os.getenv("OPENAI_API_KEY")

prompt = "What is the capital of France?"
persona = "You are a college professor; your answers always start with: 'Dear students,'"

# Instantiate an object of AugmentedPromptAgent with the required parameters
augmented_agent = AugmentedPromptAgent(openai_api_key, persona)

# Send the 'prompt' to the agent and store the response in a variable named 'augmented_agent_response'
augmented_agent_response = augmented_agent.respond(prompt)

# Print the agent's response
print(augmented_agent_response)

#  Add a comment explaining:
# - What knowledge the agent likely used to answer the prompt.
# - How the system prompt specifying the persona affected the agent's response.

# Knowledge source: This AugmentedPromptAgent uses the LLM's pre-trained general knowledge base 
# (same as DirectPromptAgent). It knows facts like "Paris is the capital of France" from its training data.
# The difference is that it adds a system prompt that instructs the agent to assume a specific persona.

# Persona effect: The system prompt added via the 'persona' parameter modified the agent's response style and format.
# Instead of giving a straightforward answer like "Paris", the agent was instructed to respond as a college professor
# starting with "Dear students,". This changed the tone, format, and delivery method of the answer while using
# the same underlying knowledge about France's capital city.