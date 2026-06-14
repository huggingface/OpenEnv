"""
CodeDark Client

HTTP client for interacting with CodeDark environment server.
Follows OpenEnv EnvClient pattern.
"""

from typing import Any, Dict, Optional
import requests


class CodeDarkEnv:
    """Client for CodeDark environment.

    Example usage:
        env = CodeDarkEnv("http://localhost:8000")
        obs = env.reset()
        print(f"Task: {obs['question']}")

        obs = env.step("run_python", "<code>result = df.shape</code>")
        print(f"Result: {obs['stdout']}")

        obs = env.step("submit_answer", "<answer>11.26</answer>")
        print(f"Reward: {obs['reward']}")
    """

    def __init__(self, base_url: str = "http://localhost:8000", timeout: int = 30):
        """Initialize client.

        Args:
            base_url: Server URL
            timeout: Request timeout in seconds
        """
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._session = requests.Session()

    def reset(
        self, task_id: Optional[str] = None, seed: Optional[int] = None
    ) -> Dict[str, Any]:
        """Reset environment for a new episode.

        Args:
            task_id: Specific task to load (optional)
            seed: Random seed for task selection (optional)

        Returns:
            Initial observation dict
        """
        payload = {}
        if task_id is not None:
            payload["task_id"] = task_id
        if seed is not None:
            payload["seed"] = seed

        response = self._session.post(
            f"{self.base_url}/reset",
            json=payload if payload else None,
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json()

    def step(self, tool: str, args: str = "") -> Dict[str, Any]:
        """Execute an action.

        Args:
            tool: Tool name (run_python, read_notes, save_note, clarify, submit_answer)
            args: Tool-specific arguments

        Returns:
            Observation dict
        """
        response = self._session.post(
            f"{self.base_url}/step",
            json={"tool": tool, "args": args},
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json()

    def state(self) -> Dict[str, Any]:
        """Get current environment state.

        Returns:
            State dict
        """
        response = self._session.get(
            f"{self.base_url}/state",
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json()

    def health(self) -> Dict[str, Any]:
        """Check server health.

        Returns:
            Health status dict
        """
        response = self._session.get(
            f"{self.base_url}/health",
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json()

    def metadata(self) -> Dict[str, Any]:
        """Get environment metadata.

        Returns:
            Metadata dict
        """
        response = self._session.get(
            f"{self.base_url}/metadata",
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json()

    def schema(self) -> Dict[str, Any]:
        """Get environment type schemas.

        Returns:
            Schema dict for action, observation, state
        """
        response = self._session.get(
            f"{self.base_url}/schema",
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json()

    # Convenience methods for common tools

    def run_python(self, code: str) -> Dict[str, Any]:
        """Execute Python code.

        Args:
            code: Python code to execute

        Returns:
            Observation dict
        """
        return self.step("run_python", f"<code>{code}</code>")

    def read_notes(self) -> Dict[str, Any]:
        """Read all saved notes.

        Returns:
            Observation dict
        """
        return self.step("read_notes", "")

    def save_note(self, content: str) -> Dict[str, Any]:
        """Save a note.

        Args:
            content: Note content

        Returns:
            Observation dict
        """
        return self.step("save_note", content)

    def clarify(self, question: str) -> Dict[str, Any]:
        """Ask a clarifying question.

        Args:
            question: Clarifying question

        Returns:
            Observation dict
        """
        return self.step("clarify", f"<question>{question}</question>")

    def submit_answer(self, answer: Any) -> Dict[str, Any]:
        """Submit final answer.

        Args:
            answer: Answer value

        Returns:
            Final observation with reward
        """
        return self.step("submit_answer", f"<answer>{answer}</answer>")

    def close(self):
        """Close the session."""
        self._session.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False
