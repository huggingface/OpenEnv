"""
CodeDark Environment

OpenEnv-compatible environment for multi-turn data analytics tasks.
Agents analyze CSV data using Python/Pandas tools and submit answers.
"""

import json
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from ..models import CodeDarkAction, CodeDarkObservation, CodeDarkState
from .tools import (
    run_python,
    read_notes,
    save_note,
    clarify,
    submit_answer,
    parse_tool_call,
)
from .scoring import compute_reward


class CodeDarkEnvironment:
    """CodeDark environment for multi-turn data analytics.

    Features:
    - Multi-turn agent evaluation
    - 5 tools: run_python, read_notes, save_note, clarify, submit_answer
    - Shaped rewards: correctness (80%) + efficiency (10%) + token cost (10%)
    - Supports bank and road datasets
    """

    def __init__(
        self,
        data_dir: Optional[str] = None,
        tasks_path: Optional[str] = None,
        max_turns: int = 10,
        max_clarifications: int = 2,
    ):
        """Initialize CodeDark environment.

        Args:
            data_dir: Path to directory containing CSV files
            tasks_path: Path to tasks.jsonl file
            max_turns: Maximum turns per episode (default: 10)
            max_clarifications: Maximum clarifications per episode (default: 2)
        """
        self.max_turns = max_turns
        self.max_clarifications = max_clarifications

        # Resolve paths
        if data_dir:
            self.data_dir = Path(data_dir)
        else:
            # Default to data/ relative to this file's parent
            self.data_dir = Path(__file__).parent.parent / "data"

        if tasks_path:
            self.tasks_path = Path(tasks_path)
        else:
            self.tasks_path = self.data_dir / "tasks" / "final_25_tasks.jsonl"

        # Load tasks
        self.tasks = self._load_tasks()
        self._tasks_by_id = {t["id"]: t for t in self.tasks}
        self._task_index = 0

        # Current episode state
        self._state: Optional[CodeDarkState] = None
        self._df: Optional[pd.DataFrame] = None
        self._current_task: Optional[Dict] = None

    def _load_tasks(self) -> List[Dict]:
        """Load tasks from JSONL file."""
        if not self.tasks_path.exists():
            return []

        tasks = []
        with open(self.tasks_path) as f:
            for line in f:
                if line.strip():
                    tasks.append(json.loads(line))
        return tasks

    def _load_data_for_task(self, task: Dict) -> Optional[pd.DataFrame]:
        """Load the appropriate CSV for a task.

        Args:
            task: Task dictionary with 'dataset' field

        Returns:
            DataFrame or None if not found
        """
        dataset = task.get("dataset", "bank")
        csv_path = self.data_dir / f"{dataset}.csv"

        if csv_path.exists():
            return pd.read_csv(csv_path)
        return None

    @property
    def state(self) -> CodeDarkState:
        """Return current environment state."""
        if self._state is None:
            self._state = CodeDarkState()
        return self._state

    def reset(
        self, task_id: Optional[str] = None, seed: Optional[int] = None
    ) -> CodeDarkObservation:
        """Reset environment for a new episode.

        Args:
            task_id: Specific task to load (optional)
            seed: Random seed for task selection (optional)

        Returns:
            Initial observation with task question
        """
        # Select task
        if task_id and task_id in self._tasks_by_id:
            task = self._tasks_by_id[task_id]
        elif self.tasks:
            if seed is not None:
                import random

                random.seed(seed)
                task = random.choice(self.tasks)
            else:
                # Round-robin through tasks
                task = self.tasks[self._task_index % len(self.tasks)]
                self._task_index += 1
        else:
            # No tasks loaded - return error observation
            return CodeDarkObservation(
                stderr="Error: No tasks loaded",
                exit_code=1,
                done=True,
            )

        self._current_task = task

        # Load data for this task
        self._df = self._load_data_for_task(task)
        if self._df is None:
            return CodeDarkObservation(
                stderr=f"Error: Could not load data for dataset '{task.get('dataset', 'bank')}'",
                exit_code=1,
                done=True,
            )

        # Initialize state
        self._state = CodeDarkState(
            episode_id=str(uuid.uuid4()),
            step_count=0,
            task_id=task["id"],
            dataset=task.get("dataset", "bank"),
            notes=[],
            turn_count=0,
            error_count=0,
            clarify_count=0,
            submitted=False,
            submitted_answer=None,
            expected_answer=task["golden"]["answer_value"],
            tolerance=task["golden"].get("tolerance", 0.01),
        )

        # Return initial observation
        return CodeDarkObservation(
            stdout=f"Task loaded. DataFrame shape: {self._df.shape}",
            turn=0,
            max_turns=self.max_turns,
            notes=[],
            task_id=task["id"],
            question=task["goal"],
            difficulty=task.get("level", "L5"),
            dataset=task.get("dataset", "bank"),
            done=False,
            submitted=False,
        )

    def step(self, action: CodeDarkAction) -> CodeDarkObservation:
        """Execute an action and return observation.

        Args:
            action: CodeDarkAction with tool name and args

        Returns:
            CodeDarkObservation with results
        """
        if self._state is None or self._current_task is None:
            return CodeDarkObservation(
                stderr="Error: Environment not reset. Call reset() first.",
                exit_code=1,
                done=True,
            )

        if self._state.submitted:
            return self._make_final_observation()

        # Increment turn
        self._state.turn_count += 1
        self._state.step_count += 1

        # Check turn limit
        if self._state.turn_count > self.max_turns:
            self._state.submitted = True
            return self._make_final_observation()

        # Parse tool-specific args
        parsed_content, parse_error = parse_tool_call(action.args, action.tool)

        if parse_error:
            self._state.error_count += 1
            return CodeDarkObservation(
                stderr=f"{action.tool} Error: {parse_error}",
                exit_code=1,
                turn=self._state.turn_count,
                max_turns=self.max_turns,
                notes=self._state.notes.copy(),
                task_id=self._state.task_id,
                question=self._current_task["goal"],
                difficulty=self._current_task.get("level", "L5"),
                dataset=self._state.dataset,
                done=False,
                submitted=False,
            )

        # Execute tool
        stdout, stderr, exit_code = "", "", 0

        if action.tool == "run_python":
            stdout, stderr, exit_code = run_python(parsed_content, self._df)

        elif action.tool == "read_notes":
            stdout, stderr, exit_code = read_notes(self._state.notes)

        elif action.tool == "save_note":
            stdout, stderr, exit_code = save_note(parsed_content, self._state.notes)

        elif action.tool == "clarify":
            stdout, stderr, exit_code, new_count = clarify(
                question=parsed_content,
                clarify_count=self._state.clarify_count,
                max_clarifications=self.max_clarifications,
                ambiguities=self._current_task.get("ambiguities", []),
                answer_type=self._current_task.get("golden", {}).get(
                    "answer_type", "scalar"
                ),
            )
            self._state.clarify_count = new_count

        elif action.tool == "submit_answer":
            stdout, stderr, exit_code, answer = submit_answer(parsed_content)
            if exit_code == 0:
                self._state.submitted = True
                self._state.submitted_answer = answer
                return self._make_final_observation()

        # Track errors
        if exit_code != 0:
            self._state.error_count += 1

        return CodeDarkObservation(
            stdout=stdout,
            stderr=stderr,
            exit_code=exit_code,
            turn=self._state.turn_count,
            max_turns=self.max_turns,
            notes=self._state.notes.copy(),
            task_id=self._state.task_id,
            question=self._current_task["goal"],
            difficulty=self._current_task.get("level", "L5"),
            dataset=self._state.dataset,
            done=False,
            submitted=False,
        )

    def _make_final_observation(self) -> CodeDarkObservation:
        """Create final observation with reward computation."""
        if self._state is None or self._current_task is None:
            return CodeDarkObservation(done=True)

        # Compute reward
        reward, correctness, efficiency, token_cost = compute_reward(
            submitted=self._state.submitted_answer,
            expected=self._state.expected_answer,
            tolerance=self._state.tolerance,
            turns=self._state.turn_count,
            max_turns=self.max_turns,
        )

        return CodeDarkObservation(
            stdout="[EPISODE COMPLETE]",
            turn=self._state.turn_count,
            max_turns=self.max_turns,
            notes=self._state.notes.copy(),
            task_id=self._state.task_id,
            question=self._current_task["goal"],
            difficulty=self._current_task.get("level", "L5"),
            dataset=self._state.dataset,
            done=True,
            submitted=self._state.submitted,
            reward=reward,
            correctness=correctness,
            efficiency=efficiency,
            metadata={
                "submitted_answer": self._state.submitted_answer,
                "expected_answer": self._state.expected_answer,
                "tolerance": self._state.tolerance,
                "error_count": self._state.error_count,
                "clarify_count": self._state.clarify_count,
                "token_cost_usd": token_cost,
            },
        )
