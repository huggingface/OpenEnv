"""
CodeDark Data Models

Pydantic models for Action, Observation, and State following OpenEnv spec.
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Any, Literal


class CodeDarkAction(BaseModel):
    """
    Action for CodeDark environment.

    Agents send actions with a tool name and arguments.

    Tools available:
    - run_python: Execute Python code with pandas/numpy
    - read_notes: Read all saved notes
    - save_note: Save a note for later recall
    - clarify: Ask clarifying question (max 2 per episode)
    - submit_answer: Submit final answer (ends episode)
    """

    tool: Literal["run_python", "read_notes", "save_note", "clarify", "submit_answer"]
    args: str = ""  # Tool-specific arguments

    model_config = {
        "json_schema_extra": {
            "examples": [
                {"tool": "run_python", "args": "result = df['y'].mean() * 100"},
                {"tool": "read_notes", "args": ""},
                {"tool": "save_note", "args": "Average subscription rate is 11.26%"},
                {"tool": "clarify", "args": "What does Q1 mean in this context?"},
                {"tool": "submit_answer", "args": "11.26"},
            ]
        }
    }


class CodeDarkObservation(BaseModel):
    """
    Observation returned after each action.

    Contains execution results, environment state, and episode info.
    Reward is only populated when done=True.
    """

    # Execution results
    stdout: str = ""
    stderr: str = ""
    exit_code: int = 0

    # Turn tracking
    turn: int = 0
    max_turns: int = 10

    # Persistent state
    notes: List[str] = Field(default_factory=list)

    # Task info
    task_id: str = ""
    question: str = ""
    difficulty: str = ""  # L4, L5, L6
    dataset: str = ""  # bank, road

    # Episode status
    done: bool = False
    submitted: bool = False

    # Reward components (only set when done=True)
    reward: Optional[float] = None
    correctness: Optional[float] = None
    efficiency: Optional[float] = None

    # Additional metadata
    metadata: dict = Field(default_factory=dict)

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "stdout": "run_python Result:\n(45211, 17)",
                    "stderr": "",
                    "exit_code": 0,
                    "turn": 1,
                    "max_turns": 10,
                    "notes": [],
                    "task_id": "bank_hard_001",
                    "question": "What's the subscription rate for month='may'?",
                    "difficulty": "L5",
                    "dataset": "bank",
                    "done": False,
                    "submitted": False,
                    "reward": None,
                }
            ]
        }
    }


class CodeDarkState(BaseModel):
    """
    Internal state for CodeDark environment.

    Tracks episode progress, accumulated notes, and submission status.
    """

    episode_id: str = ""
    step_count: int = 0

    # Task info
    task_id: str = ""
    dataset: str = ""

    # Accumulated state
    notes: List[str] = Field(default_factory=list)
    turn_count: int = 0
    error_count: int = 0
    clarify_count: int = 0

    # Submission
    submitted: bool = False
    submitted_answer: Optional[Any] = None

    # For scoring
    expected_answer: Optional[Any] = None
    tolerance: float = 0.01


class ResetRequest(BaseModel):
    """Request body for /reset endpoint."""

    task_id: Optional[str] = None
    seed: Optional[int] = None


class StepRequest(BaseModel):
    """Request body for /step endpoint."""

    tool: str
    args: str = ""


class HealthResponse(BaseModel):
    """Response for /health endpoint."""

    status: str = "healthy"
    environment: str = "codedark"
    version: str = "0.1.0"
