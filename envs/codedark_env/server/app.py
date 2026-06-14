"""
CodeDark FastAPI Server

OpenEnv-compatible HTTP server for CodeDark environment.
Provides /reset, /step, /state, and /health endpoints.
"""

import os
from contextlib import asynccontextmanager
from typing import Optional

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

from ..models import (
    CodeDarkAction,
    CodeDarkObservation,
    CodeDarkState,
    ResetRequest,
    StepRequest,
    HealthResponse,
)
from .environment import CodeDarkEnvironment


# Global environment instance
_env: Optional[CodeDarkEnvironment] = None


def get_env() -> CodeDarkEnvironment:
    """Get or create environment instance."""
    global _env
    if _env is None:
        _env = CodeDarkEnvironment(
            data_dir=os.environ.get("CODEDARK_DATA_DIR"),
            tasks_path=os.environ.get("CODEDARK_TASKS_PATH"),
            max_turns=int(os.environ.get("CODEDARK_MAX_TURNS", "10")),
        )
    return _env


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup/shutdown."""
    # Startup: initialize environment
    get_env()
    yield
    # Shutdown: cleanup if needed
    global _env
    _env = None


# Create FastAPI app
app = FastAPI(
    title="CodeDark Environment",
    description="Multi-turn data analytics environment for RL agent training",
    version="0.1.0",
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", response_class=HTMLResponse)
async def root():
    """Landing page with API documentation."""
    env = get_env()
    return f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>CodeDark Environment</title>
        <style>
            body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                   max-width: 800px; margin: 50px auto; padding: 20px; background: #0d1117; color: #c9d1d9; }}
            h1 {{ color: #f0883e; }}
            h2 {{ color: #8b949e; border-bottom: 1px solid #30363d; padding-bottom: 8px; }}
            code {{ background: #161b22; padding: 2px 6px; border-radius: 4px; color: #7ee787; }}
            pre {{ background: #161b22; padding: 16px; border-radius: 8px; overflow-x: auto; }}
            .endpoint {{ background: #21262d; padding: 12px; margin: 8px 0; border-radius: 6px; }}
            .method {{ color: #7ee787; font-weight: bold; }}
            .path {{ color: #79c0ff; }}
            a {{ color: #58a6ff; }}
            .stats {{ display: flex; gap: 20px; margin: 20px 0; }}
            .stat {{ background: #21262d; padding: 16px; border-radius: 8px; text-align: center; }}
            .stat-value {{ font-size: 24px; color: #f0883e; font-weight: bold; }}
            .stat-label {{ color: #8b949e; font-size: 12px; }}
        </style>
    </head>
    <body>
        <h1>CodeDark Environment Server</h1>
        <p>OpenEnv-compatible multi-turn data analytics environment for RL agent training.</p>

        <div class="stats">
            <div class="stat"><div class="stat-value">{len(env.tasks)}</div><div class="stat-label">Tasks</div></div>
            <div class="stat"><div class="stat-value">750K</div><div class="stat-label">Bank Rows</div></div>
            <div class="stat"><div class="stat-value">500K</div><div class="stat-label">Road Rows</div></div>
            <div class="stat"><div class="stat-value">5</div><div class="stat-label">Tools</div></div>
        </div>

        <h2>API Endpoints</h2>
        <div class="endpoint"><span class="method">GET</span> <span class="path">/health</span> - Health check</div>
        <div class="endpoint"><span class="method">POST</span> <span class="path">/reset</span> - Reset for new episode</div>
        <div class="endpoint"><span class="method">POST</span> <span class="path">/step</span> - Execute action</div>
        <div class="endpoint"><span class="method">GET</span> <span class="path">/state</span> - Current state</div>
        <div class="endpoint"><span class="method">GET</span> <span class="path">/metadata</span> - Environment info</div>
        <div class="endpoint"><span class="method">GET</span> <span class="path">/schema</span> - Type schemas</div>
        <div class="endpoint"><span class="method">GET</span> <span class="path">/docs</span> - Interactive API docs</div>

        <h2>Quick Start</h2>
        <pre>
import requests

BASE = "https://albert-einstein-09-codedark.hf.space"

# Reset for new task
obs = requests.post(f"{{BASE}}/reset").json()
print(f"Task: {{obs['question']}}")

# Run Python code
obs = requests.post(f"{{BASE}}/step", json={{
    "tool": "run_python",
    "args": "&lt;code&gt;result = df.shape&lt;/code&gt;"
}}).json()
print(f"Result: {{obs['stdout']}}")

# Submit answer
obs = requests.post(f"{{BASE}}/step", json={{
    "tool": "submit_answer",
    "args": "&lt;answer&gt;42.5&lt;/answer&gt;"
}}).json()
print(f"Reward: {{obs['reward']}}")
        </pre>

        <h2>Tools</h2>
        <ul>
            <li><code>run_python</code> - Execute Python/pandas code</li>
            <li><code>read_notes</code> - Read saved notes</li>
            <li><code>save_note</code> - Save note for later</li>
            <li><code>clarify</code> - Ask clarifying question</li>
            <li><code>submit_answer</code> - Submit final answer</li>
        </ul>

        <h2>Links</h2>
        <p>
            <a href="/docs">Interactive API Docs</a> |
            <a href="https://github.com/vj-09/codeblue-env">GitHub</a> |
            <a href="https://www.analytics-rl.com">Leaderboard</a>
        </p>
    </body>
    </html>
    """


@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        environment="codedark",
        version="0.1.0",
    )


@app.post("/reset", response_model=CodeDarkObservation)
async def reset(request: ResetRequest = None):
    """Reset environment for a new episode.

    Args:
        request: Optional reset request with task_id and seed

    Returns:
        Initial observation
    """
    env = get_env()

    if request is None:
        request = ResetRequest()

    obs = env.reset(task_id=request.task_id, seed=request.seed)
    return obs


@app.post("/step", response_model=CodeDarkObservation)
async def step(request: StepRequest):
    """Execute an action and return observation.

    Args:
        request: Step request with tool and args

    Returns:
        Observation after action execution
    """
    env = get_env()

    # Validate tool
    valid_tools = ["run_python", "read_notes", "save_note", "clarify", "submit_answer"]
    if request.tool not in valid_tools:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid tool: {request.tool}. Valid tools: {valid_tools}",
        )

    action = CodeDarkAction(tool=request.tool, args=request.args)
    obs = env.step(action)
    return obs


@app.get("/state", response_model=CodeDarkState)
async def state():
    """Get current environment state.

    Returns:
        Current CodeDarkState
    """
    env = get_env()
    return env.state


@app.get("/metadata")
async def metadata():
    """Get environment metadata.

    Returns:
        Environment metadata dict
    """
    env = get_env()
    return {
        "name": "codedark",
        "version": "0.1.0",
        "description": "Multi-turn data analytics environment for RL agent training",
        "max_turns": env.max_turns,
        "max_clarifications": env.max_clarifications,
        "num_tasks": len(env.tasks),
        "tools": [
            {"name": "run_python", "description": "Execute Python/pandas code"},
            {"name": "read_notes", "description": "Read all saved notes"},
            {"name": "save_note", "description": "Save a note for later recall"},
            {"name": "clarify", "description": "Ask clarifying question (max 2)"},
            {"name": "submit_answer", "description": "Submit final answer"},
        ],
        "reward_structure": {
            "max_reward": 1.0,
            "components": [
                {"name": "correctness", "weight": 0.80},
                {"name": "efficiency", "weight": 0.10},
                {"name": "token_cost", "weight": 0.10},
            ],
        },
    }


@app.get("/schema")
async def schema():
    """Get environment schema for Action, Observation, State.

    Returns:
        JSON schemas for all types
    """
    return {
        "action": CodeDarkAction.model_json_schema(),
        "observation": CodeDarkObservation.model_json_schema(),
        "state": CodeDarkState.model_json_schema(),
    }


def main():
    """Run the server."""
    uvicorn.run(
        "codedark.server.app:app",
        host=os.environ.get("HOST", "0.0.0.0"),
        port=int(os.environ.get("PORT", "8000")),
        reload=os.environ.get("RELOAD", "false").lower() == "true",
    )


if __name__ == "__main__":
    main()
