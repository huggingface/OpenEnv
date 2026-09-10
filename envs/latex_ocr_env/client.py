# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
LaTeX OCR Environment client.

Gym-style client (reset/step over WebSocket) plus thin HTTP helpers for the
Task API so a trainer can enumerate and select dataset tasks.

Example:
    >>> with LatexOCREnv(base_url="http://localhost:8000") as env:
    ...     print(env.list_splits())            # ["train", "test"]
    ...     print(env.num_tasks("test"))        # 7595
    ...     result = env.reset(split="test", index=0)
    ...     img = result.observation.image_base64
    ...     # ... run a VLM to produce `latex` ...
    ...     result = env.step(LatexOCRAction(latex=latex))
    ...     print(result.reward, result.observation.target_latex)
"""

from __future__ import annotations

from typing import Any, Optional
from urllib.parse import urljoin

import requests

try:
    from openenv.core.client_types import StepResult
    from openenv.core.env_client import EnvClient
    from openenv.core.env_server.types import State
except ImportError:
    from core.client_types import StepResult
    from core.env_client import EnvClient
    from core.env_server.types import State

from .models import LatexOCRAction, LatexOCRObservation

ENV_NAME = "latex_ocr_env"


class LatexOCREnv(EnvClient[LatexOCRAction, LatexOCRObservation, State]):
    """Client for the LaTeX OCR environment."""

    def reset(
        self,
        split: str = "train",
        index: Optional[int] = None,
        seed: Optional[int] = None,
        **kwargs: Any,
    ) -> StepResult[LatexOCRObservation]:
        """Reset to a specific dataset task (or a random one if ``index`` is None)."""
        payload: dict[str, Any] = {"split": split}
        if index is not None:
            payload["index"] = index
        if seed is not None:
            payload["seed"] = seed
        payload.update(kwargs)
        return super().reset(**payload)

    # --- Gym-style (de)serialization required by EnvClient ---
    def _step_payload(self, action: LatexOCRAction) -> dict[str, Any]:
        return action.model_dump()

    def _parse_result(self, data: dict[str, Any]) -> StepResult[LatexOCRObservation]:
        obs_data = dict(data.get("observation", data))
        # Core serialization lifts reward/done to the top level and strips them from
        # the observation payload; merge them back so observation.reward/.done match
        # StepResult (consistent with other env clients).
        reward = data.get("reward", obs_data.get("reward"))
        done = data.get("done", obs_data.get("done"))
        obs_data["reward"] = reward
        obs_data["done"] = done
        obs = LatexOCRObservation(**obs_data)
        # `metadata` is what StepResult actually exposes. The previous version passed `info=`,
        # which raised TypeError on every call and dropped whatever the server sent. `info` is
        # still read as a fallback because the server used that name at one point.
        metadata = data.get("metadata") or data.get("info") or None
        return StepResult(observation=obs, reward=reward, done=done, metadata=metadata)

    def _parse_state(self, data: dict[str, Any]) -> State:
        return State(**data)

    # ------------------------------------------------------------------ #
    # Task API (HTTP)                                                     #
    # ------------------------------------------------------------------ #
    def _http_base(self) -> str:
        # Newer core exposes ``_base_url`` (http); older core stores only
        # ``_ws_url`` (ws://host/ws). Derive an http base that works for both.
        base = getattr(self, "_base_url", None)
        if not base:
            ws = getattr(self, "_ws_url", None)
            if not ws:
                raise RuntimeError("Task API requires an HTTP base URL")
            base = ws[:-3] if ws.endswith("/ws") else ws
        base = base.replace("wss://", "https://").replace("ws://", "http://")
        return base if base.endswith("/") else base + "/"

    def list_splits(self) -> list[str]:
        resp = requests.get(
            urljoin(self._http_base(), f"{ENV_NAME}/splits"), timeout=30
        )
        resp.raise_for_status()
        return [s["name"] for s in resp.json()]

    def num_tasks(self, split: str) -> int:
        resp = requests.post(
            urljoin(self._http_base(), f"{ENV_NAME}/num_tasks"),
            json={"split": split},
            timeout=60,
        )
        resp.raise_for_status()
        return int(resp.json()["num_tasks"])

    def get_task(self, split: str, index: int) -> dict[str, Any]:
        resp = requests.post(
            urljoin(self._http_base(), f"{ENV_NAME}/task"),
            json={"split": split, "index": index},
            timeout=60,
        )
        resp.raise_for_status()
        return resp.json()["task"]

    def get_task_range(
        self, split: str, start: int | None = None, stop: int | None = None
    ) -> list[dict[str, Any]]:
        resp = requests.post(
            urljoin(self._http_base(), f"{ENV_NAME}/task_range"),
            json={"split": split, "start": start, "stop": stop},
            timeout=120,
        )
        resp.raise_for_status()
        return resp.json()["tasks"]
