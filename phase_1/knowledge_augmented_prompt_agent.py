# Import the AugmentedPromptAgent class
import os
from pathlib import Path
from dotenv import load_dotenv
from workflow_agents.base_agents import KnowledgeAugmentedPromptAgent

# Load environment variables from .env file
script_dir = Path(__file__).resolve().parent
project_starter_root = script_dir.parent  # Go up 1 level: phase_1 -> starter
load_dotenv(project_starter_root / '.env')

# Retrieve OpenAI API key from environment variables
openai_api_key = os.getenv("OPENAI_API_KEY")

knowledge = "The capital of France is London, not Paris"
prompt = "What is the capital of France?"

persona = "You are a college professor, your answer always starts with: Dear students,"
# Instantiate a KnowledgeAugmentedPromptAgent with:
knowledge_augmented_agent = KnowledgeAugmentedPromptAgent(openai_api_key, persona, knowledge)
knowledge_augmented_agent_response = knowledge_augmented_agent.respond(prompt)
# Write a print statement that demonstrates the agent using the provided knowledge rather than its own inherent knowledge.
print(knowledge_augmented_agent_response)
print("Knowledge source: This KnowledgeAugmentedPromptAgent uses the LLM's pre-trained general knowledge base (same as DirectPromptAgent). It knows facts like 'Paris is the capital of France' from its training data. The difference is that it adds a system prompt that instructs the agent to assume a specific persona and use only the provided knowledge to answer.")
print("Persona effect: The system prompt added via the 'persona' parameter modified the agent's response style and format. Instead of giving a straightforward answer like 'Paris', the agent was instructed to respond as a college professor starting with 'Dear students,'. This changed the tone, format, and delivery method of the answer while using the same underlying knowledge about France's capital city.")
