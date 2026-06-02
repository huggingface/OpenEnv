import os
import sys
from pathlib import Path

import pytest

# Add the project root to the path for envs imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

try:
    import camel  # noqa: F401
except Exception:
    camel = None

from envs.tbench2_env.models import Tbench2Action
from envs.tbench2_env.server import tbench2_env_environment
from envs.tbench2_env.server.tbench2_env_environment import Tbench2Environment


class _FakeTerminalToolkit:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def shell_exec(self, **kwargs):
        return ""


def test_tbench2_reset_uses_default_task_id(monkeypatch, tmp_path: Path):
    """HTTP resets without kwargs should land on the default demo task."""
    task_dir = tmp_path / "headless-terminal"
    task_dir.mkdir()
    (task_dir / "instruction.md").write_text("Solve the terminal task.\n")

    monkeypatch.setattr(
        tbench2_env_environment,
        "_require_terminal_toolkit",
        lambda: _FakeTerminalToolkit,
    )

    env = Tbench2Environment(
        tasks_dir=str(tmp_path),
        output_dir=str(tmp_path / "runs"),
        default_task_id="headless-terminal",
    )

    observation = env.reset()

    assert observation.success is True
    assert observation.task_id == "headless-terminal"
    assert "terminal task" in observation.instruction


@pytest.mark.skipif(camel is None, reason="camel-ai not installed")
@pytest.mark.skipif(
    os.environ.get("TB2_ENABLE_TESTS", "0") != "1",
    reason="TB2_ENABLE_TESTS not enabled",
)
def test_tbench2_env_smoke():
    env = Tbench2Environment(tasks_dir=os.environ.get("TB2_TASKS_DIR"))
    obs = env.reset(task_id=os.environ.get("TB2_TASK_ID", "headless-terminal"))
    assert obs.instruction

    result = env.step(Tbench2Action(action_type="exec", command="pwd"))
    assert result.success
    assert result.output

    env.step(Tbench2Action(action_type="close"))
    env.close()
