# SPDX-License-Identifier: BSD-3-Clause

"""Wire types for the Harbor environment.

Shared by the client and the server; neither side imports the other.
"""

from typing import Any, get_args, Literal, Union

from openenv.core.env_server.types import Action, Observation, State
from pydantic import Field


#: Actions an agent may take while solving the task.
AgentActionType = Literal["exec", "read", "write"]

#: Actions reserved for training orchestration. `evaluate` grades the episode and
#: `solve` runs the task's own reference solution; neither is part of the agent's
#: action space, mirroring OpenEnv's rule that simulation control stays on the
#: infrastructure side of the boundary.
ControlActionType = Literal["evaluate", "solve"]

#: Every action the server accepts. Composed from the two sets above rather than
#: restated, so the wire schema and the agent/orchestration split cannot drift:
#: a new action has to be classified as one or the other to exist at all.
HarborActionType = Union[AgentActionType, ControlActionType]

AGENT_ACTIONS: frozenset[str] = frozenset(get_args(AgentActionType))
CONTROL_ACTIONS: frozenset[str] = frozenset(get_args(ControlActionType))


class HarborAction(Action):
    """One interaction with a Harbor task sandbox.

    Args:
        action_type (`str`, *optional*, defaults to `"exec"`):
            One of `"exec"`, `"read"`, `"write"` (agent actions), or
            `"evaluate"`, `"solve"` (orchestration actions).
        command (`str`, *optional*):
            Shell command, for `action_type="exec"`.
        path (`str`, *optional*):
            Path relative to the working directory, for `"read"` and `"write"`.
        content (`str`, *optional*):
            File contents, for `"write"`.
        timeout_s (`float`, *optional*):
            Overrides the default command timeout. Ignored by `"evaluate"` and
            `"solve"`, which use the timeouts declared in `task.toml`.

    Examples:

    ```python
    HarborAction(action_type="exec", command="pytest -q")
    HarborAction(action_type="write", path="calc.py", content="def add(a, b):\\n    return a + b\\n")
    HarborAction(action_type="evaluate")
    ```
    """

    action_type: HarborActionType = Field(default="exec")
    command: str = Field(default="")
    path: str = Field(default="")
    content: str = Field(default="")
    timeout_s: float | None = Field(default=None, gt=0)


class HarborObservation(Observation):
    """What the sandbox reported back.

    Args:
        instruction (`str`):
            The task's `instruction.md`, repeated on every step so a stateless
            policy can always see the goal.
        output (`str`):
            Merged stdout/stderr of the command, file contents for `"read"`, or a
            short confirmation for `"write"`.
        action_type (`str`):
            The action this observation answers, or `"reset"`.
        success (`bool`):
            Whether the action itself completed. Unrelated to the reward: a
            failing test command is a successful `exec`.
        error (`str`):
            Why the action failed, when it did.
        exit_code (`int`, *optional*):
            Exit status, for actions that ran a command.
        timed_out (`bool`):
            Whether the command was killed by its timeout.
        task_id (`str`):
            Identifier of the running task.
        task_name (`str`):
            The task's `[task].name`, conventionally `<org>/<slug>`.
        workdir (`str`):
            Working directory inside the sandbox; `path` is relative to it.
        mode (`str`):
            Sandbox backend in use, `"local"` or `"docker"`.
        info (`dict`):
            Extra detail — the reward breakdown after `"evaluate"`, the task
            digest after `reset()`.
    """

    instruction: str = Field(default="")
    output: str = Field(default="")
    action_type: str = Field(default="")
    success: bool = Field(default=True)
    error: str = Field(default="")
    exit_code: int | None = Field(default=None)
    timed_out: bool = Field(default=False)
    task_id: str = Field(default="")
    task_name: str = Field(default="")
    workdir: str = Field(default="")
    mode: str = Field(default="")
    info: dict[str, Any] = Field(default_factory=dict)


class HarborState(State):
    """Server-side episode state, for training orchestration.

    Args:
        task_id (`str`):
            Identifier of the running task, empty before the first `reset()`.
        task_name (`str`):
            The task's `[task].name`.
        workdir (`str`):
            Working directory inside the sandbox.
        mode (`str`):
            Sandbox backend in use.
        available_tasks (`list[str]`):
            Task identifiers discovered in the catalog, truncated for large
            catalogs — see `task_count` for the true total.
        task_count (`int`):
            Number of tasks in the catalog.
        last_action_type (`str`):
            The most recent action.
        last_exit_code (`int`, *optional*):
            Exit status of the most recent command.
        evaluated (`bool`):
            Whether the verifier has run; the episode ends when it has.
        reward (`float`, *optional*):
            Reward reported by the verifier, once it has run.
    """

    task_id: str = Field(default="")
    task_name: str = Field(default="")
    workdir: str = Field(default="")
    mode: str = Field(default="")
    available_tasks: list[str] = Field(default_factory=list)
    task_count: int = Field(default=0)
    last_action_type: str = Field(default="")
    last_exit_code: int | None = Field(default=None)
    evaluated: bool = Field(default=False)
    reward: float | None = Field(default=None)
