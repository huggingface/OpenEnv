# SPDX-License-Identifier: BSD-3-Clause

"""Client for the Harbor environment."""

from __future__ import annotations

from typing import Any


# Support both in-repo and standalone imports
try:
    # In-repo imports (when running from OpenEnv repository)
    from openenv.core.client_types import StepResult
    from openenv.core.env_client import EnvClient

    from .models import HarborAction, HarborObservation, HarborState
except ImportError:
    # Standalone imports (when environment is standalone with openenv from pip)
    from models import HarborAction, HarborObservation, HarborState

    from openenv.core.client_types import StepResult
    from openenv.core.env_client import EnvClient


class HarborEnv(EnvClient[HarborAction, HarborObservation, HarborState]):
    """Drives a Harbor task server over the standard OpenEnv WebSocket API.

    The five actions map onto Harbor's phases: `run` / `read_file` / `write_file`
    are what an agent does, [`evaluate`] runs the task's verifier and ends the
    episode, and [`solve`] applies the task's reference solution.

    Examples:

    ```python
    import asyncio
    from harbor_env import HarborEnv

    async def main():
        env = HarborEnv(base_url="http://localhost:8000")
        start = await env.reset(task_id="fix-sum-bug")
        print(start.observation.instruction)

        await env.write_file("calc.py", "def total(xs):\\n    return sum(xs)\\n")
        result = await env.evaluate()
        print(result.reward, result.done)
        await env.close()

    asyncio.run(main())
    ```

    The same flow synchronously:

    ```python
    with HarborEnv(base_url="http://localhost:8000").sync() as env:
        env.reset(task_id="fix-sum-bug")
        env.step(HarborAction(action_type="exec", command="pytest -q"))
        print(env.step(HarborAction(action_type="evaluate")).reward)
    ```
    """

    # --- convenience wrappers ----------------------------------------------

    async def run(
        self, command: str, timeout_s: float | None = None
    ) -> StepResult[HarborObservation]:
        """Run a shell command in the task's working directory."""
        return await self.step(
            HarborAction(action_type="exec", command=command, timeout_s=timeout_s)
        )

    async def read_file(self, path: str) -> StepResult[HarborObservation]:
        """Read a file, relative to the task's working directory."""
        return await self.step(HarborAction(action_type="read", path=path))

    async def write_file(
        self, path: str, content: str
    ) -> StepResult[HarborObservation]:
        """Write a file, relative to the task's working directory."""
        return await self.step(
            HarborAction(action_type="write", path=path, content=content)
        )

    async def evaluate(self) -> StepResult[HarborObservation]:
        """Run the task's verifier and end the episode.

        The reward is whatever `tests/test.sh` wrote to `/logs/verifier`; the
        per-metric breakdown is in `observation.info`.
        """
        return await self.step(HarborAction(action_type="evaluate"))

    async def solve(self) -> StepResult[HarborObservation]:
        """Apply the task's reference solution — Harbor's oracle agent.

        Follow with [`evaluate`] to confirm the task grades at full reward. This
        is orchestration tooling for validating a task set, not part of an
        agent's action space.
        """
        return await self.step(HarborAction(action_type="solve"))

    # --- wire protocol ------------------------------------------------------

    def _step_payload(self, action: HarborAction) -> dict[str, Any]:
        return action.model_dump()

    def _parse_result(self, payload: dict[str, Any]) -> StepResult[HarborObservation]:
        data = dict(payload.get("observation", {}))
        data["reward"] = payload.get("reward")
        data["done"] = payload.get("done", False)
        observation = HarborObservation.model_validate(data)
        return StepResult(
            observation=observation,
            reward=observation.reward,
            done=observation.done,
        )

    def _parse_state(self, payload: dict[str, Any]) -> HarborState:
        return HarborState.model_validate(payload)
