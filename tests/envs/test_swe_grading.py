import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from mini_swe_env.grading import GradingError, grade_from_case_results
from mini_swe_env.harness import (
    HOME,
    SWESession,
    SWEAgentConfig,
    SWESessionFactory,
    _wrap_instruction,
)
from mini_swe_env.models import SWEGymTask, SWETask
from openenv.core.harness.sandbox.base import ExecResult


def _task() -> SWEGymTask:
    return SWEGymTask(
        instance_id="demo__task-1",
        repo="demo/repo",
        base_commit="deadbeef",
        problem_statement="Fix the bug.",
        version="1.0",
        patch="",
        test_patch="",
        FAIL_TO_PASS=["tests/test_a.py::test_fix"],
        PASS_TO_PASS=["tests/test_b.py::test_regression"],
    )


def test_grade_from_case_results_binary_mode_stays_sparse() -> None:
    grade = grade_from_case_results(
        _task(),
        {
            "tests/test_a.py::test_fix": True,
            "tests/test_b.py::test_regression": False,
        },
    )
    assert grade.case_fraction == pytest.approx(0.5)
    assert grade.reward == pytest.approx(0.0)
    assert grade.resolved is False


def test_grade_from_case_results_case_fraction_mode_is_dense() -> None:
    grade = grade_from_case_results(
        _task(),
        {
            "tests/test_a.py::test_fix": True,
            "tests/test_b.py::test_regression": False,
        },
        reward_mode="case_fraction",
    )
    assert grade.case_fraction == pytest.approx(0.5)
    assert grade.reward == pytest.approx(0.5)
    assert grade.resolved is False


def test_grade_from_case_results_rejects_unknown_reward_mode() -> None:
    with pytest.raises(GradingError):
        grade_from_case_results(_task(), {}, reward_mode="unknown")


def test_wrap_instruction_includes_optional_hints_block() -> None:
    wrapped = _wrap_instruction(
        "Fix the bug.",
        hints_text="Look at module.py around line 10.",
        workdir="/testbed",
    )
    assert "<maintainer_hints>" in wrapped
    assert "Look at module.py around line 10." in wrapped
    assert "You cannot continue working after submitting" in wrapped
    assert "Each bash tool call runs in a fresh shell." in wrapped
    assert "Test edits do not help." in wrapped
    assert "Do not use `git commit`, `git branch`, or `git push`." in wrapped
    assert "Start with the maintainer hints or issue description." in wrapped


def test_wrap_instruction_supports_fallback_only_grading_mode() -> None:
    wrapped = _wrap_instruction(
        "Fix the bug.",
        workdir="/testbed",
        answer_tool_enabled=False,
    )
    assert "There is no `answer` tool in this run." in wrapped
    assert "graded automatically when the session ends." in wrapped
    assert "If the `answer` tool is available" in wrapped


def test_swe_session_verify_uses_host_fallback_grader() -> None:
    task = SWETask(
        task_id="task-1",
        source="unit",
        instance_id="demo__task-1",
        repo="demo/repo",
        base_commit="deadbeef",
        instruction="Fix the bug.",
        setup=[],
        verify=[],
    )
    session = SWESession(
        swe_task=task,
        spec=SimpleNamespace(name="opencode"),
        sandbox=SimpleNamespace(),
        task=SimpleNamespace(instruction="Fix the bug."),
        config=SWEAgentConfig(sandbox_home=HOME, workdir="/testbed"),
    )

    def _grader(_sandbox, swe_task, *, home: str, workdir: str) -> tuple[float, bool]:
        assert swe_task.instance_id == "demo__task-1"
        assert home == HOME
        assert workdir == "/testbed"
        return 0.5, False

    session._fallback_grader = _grader
    result = session.verify(transcript=[])

    assert result.env_reward == pytest.approx(0.5)
    assert result.metrics["reward_source"] == "host_verify_fallback"
    assert result.metrics["resolved"] is False


def test_list_changed_test_paths_filters_to_test_like_files() -> None:
    class _Sandbox:
        def exec(self, cmd: str, *, cwd=None, timeout=None):  # type: ignore[no-untyped-def]
            if cmd == "git diff --name-only HEAD --":
                return ExecResult(
                    exit_code=0,
                    stdout=(
                        "moto/ssm/models.py\n"
                        "tests/test_ssm/test_ssm_boto3.py\n"
                        "pkg/widget_test.py\n"
                    ),
                    stderr="",
                )
            if cmd == "git ls-files --others --exclude-standard":
                return ExecResult(
                    exit_code=0,
                    stdout="notes.txt\ntesting/helpers/new_case.py\n",
                    stderr="",
                )
            raise AssertionError(f"unexpected command: {cmd}")

    paths = SWESessionFactory._list_changed_test_paths(_Sandbox(), workdir="/testbed")
    assert paths == [
        "pkg/widget_test.py",
        "testing/helpers/new_case.py",
        "tests/test_ssm/test_ssm_boto3.py",
    ]
