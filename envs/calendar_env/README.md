---
title: Calendar Environment Server
emoji: 📅
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
app_port: 8004
base_path: /docs
tags:
  - openenv
---
# Calendar Environment

This environment exposes a Calendar Gym tools through the OpenEnv reset/step/state interface. The server runs a FastAPI app that serves the OpenEnv endpoints.

## Server Setup

### Docker (Recommended)

```bash
cd envs/calendar_env
docker build -t calendar-env:latest .
docker run --rm -p 8004:8004 calendar-env:latest
curl http://localhost:8004/health
```
On Server health success response will be:
`{"status":"healthy","service":"calendar-env"}`

### Without Docker

```bash
cd envs/calendar_env
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
uvicorn server.app:app --host 0.0.0.0 --port 8004
```

## Client Setup

### Quick Start (Demo)

For a quick demo, simply update `llm_api_key` in `scenario_config.json` and run:
```bash
python client.py --scenario scenario_config.json
```
The existing config includes a sample scenario for testing.

### Configure Scenario

To customize for your use case, edit `scenario_config.json` and update these fields:

**llm variables:**
- `llm_api_key` - Your OpenAI/Anthropic/Google API key (or set via env var)
- `llm_model` - Model name (e.g., `gpt-4o-mini`, `claude-3-5-sonnet-20241022`)
- `llm_provider` - Provider: `openai`, `anthropic`, or `google`

**Scenario Variables**
- `user_prompt` - Task for the agent to complete
- `system_prompt` - Instructions for agent behavior
- `context` - The auth headers for gym like (x-access-token)
- `seed_database_file` - Path to SQL file for custom data
- `verifiers` - SQL queries to validate task completion
- `expected_tools` - Tools agent should use (for tracking)

### Run Client

**Run scenario-based benchmark:**
```bash
python client.py --scenario scenario_config.json 
```

Output will be saved to `response_output/` folder with execution details, tool calls, and verification results.

**Notebook Evaluation:**
For interactive evaluation and testing, use the: [`Jupyter notebook`](client_notebooks/OpenEnv_and_mcp_Single_Gym_Client_Meta_Turing.ipynb)