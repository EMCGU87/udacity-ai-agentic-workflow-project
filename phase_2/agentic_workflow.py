import os
import sys
from pathlib import Path

from dotenv import load_dotenv

script_dir = Path(__file__).resolve().parent
project_starter_root = script_dir.parent  # phase_2 -> starter
phase_1_dir = project_starter_root / "phase_1"
if str(phase_1_dir) not in sys.path:
    sys.path.append(str(phase_1_dir))

from workflow_agents.base_agents import (  # noqa: E402
    ActionPlanningAgent,
    EvaluationAgent,
    KnowledgeAugmentedPromptAgent,
    RoutingAgent,
)


# Load OpenAI API key
load_dotenv(project_starter_root / ".env")
openai_api_key = os.getenv("OPENAI_API_KEY")


# Load Product Specification
with open(script_dir / "Product-Spec-Email-Router.txt", "r", encoding="utf-8") as f:
    product_spec = f.read()


# Instantiate Action Planning Agent
knowledge_action_planning = (
    "You are helping a Technical Program Manager coordinate product development.\n"
    "Break a project request into practical steps that should be routed to:\n"
    "1) Product Manager (user stories)\n"
    "2) Program Manager (features)\n"
    "3) Development Engineer (engineering tasks)\n"
    "Return concise, actionable steps."
)
action_planning_agent = ActionPlanningAgent(openai_api_key, knowledge_action_planning)


# Complete Product Manager knowledge (append product spec)
persona_product_manager = "You are a Product Manager."
knowledge_product_manager = (
    "You create user stories only. You do not create implementation tasks or feature architecture.\n"
    "Always format user stories as: As a [type of user], I want [an action or feature] so that [benefit/value].\n"
    "Use the product specification below as your source of truth.\n\n"
    + product_spec
)


# Instantiate Product Manager Knowledge Agent
product_manager_knowledge_agent = KnowledgeAugmentedPromptAgent(
    openai_api_key,
    persona_product_manager,
    knowledge_product_manager,
)


#  Instantiate Product Manager Evaluation Agent
persona_product_manager_eval = (
    "You are an evaluation agent that checks the answers of other worker agents"
)
evaluation_criteria_product_manager = (
    "The answer should be stories that follow the following structure: "
    "As a [type of user], I want [an action or feature] so that [benefit/value]."
)
product_manager_evaluation_agent = EvaluationAgent(
    openai_api_key,
    persona_product_manager_eval,
    evaluation_criteria_product_manager,
    product_manager_knowledge_agent,
    5,
)


# Instantiate Program Manager Knowledge Agent 
persona_program_manager = "You are a Technical Program Manager."
knowledge_program_manager = (
    "You define product features from user stories.\n"
    "Do not write implementation tasks.\n"
    "Use structured feature formatting."
)
program_manager_knowledge_agent = KnowledgeAugmentedPromptAgent(
    openai_api_key,
    persona_program_manager,
    knowledge_program_manager,
)


#  Instantiate Program Manager Evaluation Agent
persona_program_manager_eval = (
    "You are an evaluation agent that checks the answers of other worker agents"
)
evaluation_criteria_program_manager = (
    "The answer should be product features that follow the following structure: "
    "Feature Name: A clear, concise title that identifies the capability\n"
    "Description: A brief explanation of what the feature does and its purpose\n"
    "Key Functionality: The specific capabilities or actions the feature provides\n"
    "User Benefit: How this feature creates value for the user"
)
program_manager_evaluation_agent = EvaluationAgent(
    openai_api_key,
    persona_program_manager_eval,
    evaluation_criteria_program_manager,
    program_manager_knowledge_agent,
    5,
)


# Instantiate Development Engineer Knowledge Agent
persona_dev_engineer = "You are a Development Engineer."
knowledge_dev_engineer = (
    "You convert feature definitions into concrete engineering tasks.\n"
    "Focus on technical implementation details, dependencies, acceptance criteria, and effort."
)
development_engineer_knowledge_agent = KnowledgeAugmentedPromptAgent(
    openai_api_key,
    persona_dev_engineer,
    knowledge_dev_engineer,
)


# Instantiate Development Engineer Evaluation Agent
persona_dev_engineer_eval = (
    "You are an evaluation agent that checks the answers of other worker agents"
)
evaluation_criteria_dev_engineer = (
    "The answer should be tasks following this exact structure: "
    "Task ID: A unique identifier for tracking purposes\n"
    "Task Title: Brief description of the specific development work\n"
    "Related User Story: Reference to the parent user story\n"
    "Description: Detailed explanation of the technical work required\n"
    "Acceptance Criteria: Specific requirements that must be met for completion\n"
    "Estimated Effort: Time or complexity estimation\n"
    "Dependencies: Any tasks that must be completed first"
)
development_engineer_evaluation_agent = EvaluationAgent(
    openai_api_key,
    persona_dev_engineer_eval,
    evaluation_criteria_dev_engineer,
    development_engineer_knowledge_agent,
    5,
)


#   Define support functions
def product_manager_support_function(query: str) -> str:
    response_from_knowledge_agent = product_manager_knowledge_agent.respond(query)
    validated = product_manager_evaluation_agent.evaluate(response_from_knowledge_agent)
    return validated["final_response"]


def program_manager_support_function(query: str) -> str:
    response_from_knowledge_agent = program_manager_knowledge_agent.respond(query)
    validated = program_manager_evaluation_agent.evaluate(response_from_knowledge_agent)
    return validated["final_response"]


def development_engineer_support_function(query: str) -> str:
    response_from_knowledge_agent = development_engineer_knowledge_agent.respond(query)
    validated = development_engineer_evaluation_agent.evaluate(response_from_knowledge_agent)
    return validated["final_response"]


# Instantiate Routing Agent + routes
routes = [
    {
        "name": "Product Manager",
        "description": (
            "Responsible for defining product personas and user stories only. "
            "Does not define features or implementation tasks."
        ),
        "func": lambda x: product_manager_support_function(x),
    },
    {
        "name": "Program Manager",
        "description": (
            "Responsible for converting stories into structured product features. "
            "Does not define engineering implementation tasks."
        ),
        "func": lambda x: program_manager_support_function(x),
    },
    {
        "name": "Development Engineer",
        "description": (
            "Responsible for converting features into engineering tasks, "
            "dependencies, acceptance criteria, and effort estimates."
        ),
        "func": lambda x: development_engineer_support_function(x),
    },
]
routing_agent = RoutingAgent(openai_api_key, routes)
routing_agent.agents = routes


# Implement workflow
workflow_prompt = (
    "Create a structured project plan for the Email Router product from the specification. "
    "First define user stories, then define product features, then define engineering tasks."
)
workflow_steps = action_planning_agent.extract_steps_from_prompt(workflow_prompt)
completed_steps = []

print("=== Workflow Steps ===")
for idx, step in enumerate(workflow_steps, start=1):
    print(f"\n[Step {idx}] {step}")
    result = routing_agent.route(step)
    completed_steps.append(result)
    print(f"[Result {idx}]\n{result}\n")

if completed_steps:
    print("=== Final Workflow Output ===")
    print(completed_steps[-1])
else:
    print("No workflow steps were produced.")
