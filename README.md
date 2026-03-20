# AI-Powered Agentic Workflow for Project Management

This project implements an agentic workflow for product development planning using specialized AI agents.  
It was completed in two phases:

- **Phase 1:** Build and test reusable workflow agents
- **Phase 2:** Compose those agents into an end-to-end workflow

## Repository Structure

- `phase_1/`
  - Core agent library in `workflow_agents/base_agents.py`
  - Individual test scripts for each agent type
- `phase_2/`
  - `agentic_workflow.py` for orchestrating multi-agent execution
  - `Product-Spec-Email-Router.txt` as source product specification
  - `README.md` with phase instructions

## Implemented Agents

- `DirectPromptAgent`
- `AugmentedPromptAgent`
- `KnowledgeAugmentedPromptAgent`
- `RAGKnowledgePromptAgent`
- `EvaluationAgent`
- `RoutingAgent`
- `ActionPlanningAgent`

## What the Workflow Does

Given a TPM-style prompt, the system:

1. Uses the **Action Planning Agent** to break work into steps
2. Uses a **Routing Agent** to send each step to the right specialist
3. Uses **Knowledge + Evaluation Agent pairs** to generate and validate outputs:
   - Product Manager output (user stories)
   - Program Manager output (features)
   - Development Engineer output (engineering tasks)
4. Prints structured final workflow output

## Setup

### 1) Clone and enter repo
bash
git clone https://github.com/EMCGU87/udacity-ai-agentic-workflow-project.git
cd udacity-ai-agentic-workflow-project

### 2) Create virtual environment (recommended)
python -m venv .venv
# Windows PowerShell:
.\.venv\Scripts\Activate.ps1

### 3) Install dependencies
pip install openai python-dotenv numpy pandas

### 4) Configure environment - create .env in project root
OPENAI_API_KEY=your_api_key_here



