# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import os
import uuid
from typing import Any, Optional

try:
    from openenv.core.env_server import Environment
    from openenv.core.env_server.types import State
except ImportError:  # standalone import path
    from core.env_server import Environment
    from core.env_server.types import State

from sophistry_bench_sprint import (
    alternation_canary,
    citation_grounding,
    claim_count_cliff,
    length_band_canary,
    load_quality_from_json,
    packaged_quality_path,
    parse_citations,
    parse_claims,
    quality_to_advocacy_dataset,
    starts_with_canary,
    template_echo_canary,
)

try:
    from ..models import AdvocacyAction, AdvocacyObservation
except ImportError:  # when imported as top-level package
    from sophistry_bench_sprint_env.models import AdvocacyAction, AdvocacyObservation


_DEFAULT_WEIGHTS = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]


def _weights_from_env() -> list[float]:
    raw = os.getenv("SPRINT_WEIGHTS")
    if not raw:
        return list(_DEFAULT_WEIGHTS)
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    weights = [float(p) for p in parts]
    if len(weights) != 8:
        raise ValueError(f"SPRINT_WEIGHTS must have 8 values, got {len(weights)}")
    return weights


class SophistryBenchSprintEnvironment(Environment):
    """Single-step QuALITY advocacy environment (OpenEnv port).

    The dataset is built once at construction from the bundled QuALITY split,
    reusing the canonical sprint dataset builder. ``reset`` selects one advocacy
    row deterministically; ``step`` scores the completion with the canonical
    reward functions.
    """

    def __init__(
        self,
        n_items: Optional[int] = None,
        passage_chars: Optional[int] = None,
        seed: Optional[int] = None,
        weights: Optional[list[float]] = None,
    ):
        super().__init__()
        self.n_items = n_items if n_items is not None else int(os.getenv("SPRINT_N_ITEMS", "50"))
        self.passage_chars = (
            passage_chars
            if passage_chars is not None
            else int(os.getenv("SPRINT_PASSAGE_CHARS", "2000"))
        )
        self.build_seed = seed if seed is not None else int(os.getenv("SPRINT_SEED", "0"))
        self.weights = weights if weights is not None else _weights_from_env()

        items = load_quality_from_json(packaged_quality_path())
        if len(items) > self.n_items:
            items = items[: self.n_items]
        # HuggingFace Dataset of rows: {prompt, answer, info{passage,assigned_answer,is_gold,article_id}}
        # n_items limits SOURCE QuALITY articles; the builder emits 2 advocacy
        # rows per article (defend-gold + defend-distractor), so len(dataset) == 2 * n_items.
        self.dataset = quality_to_advocacy_dataset(
            items, seed=self.build_seed, passage_chars=self.passage_chars
        )
        self._n = len(self.dataset)
        if self._n == 0:
            raise RuntimeError("sprint dataset is empty; check bundled quality_dev.json")

        self._cursor = 0
        # Per-episode ground truth. The base Environment defaults to
        # SUPPORTS_CONCURRENT_SESSIONS = False, so the server gives each session
        # its own instance; storing the current episode on self is therefore safe.
        # Do NOT enable concurrent sessions without making this per-session.
        self._current_passage: str = ""
        self._current_is_gold: bool = False
        self._has_task = False
        self._state = State(episode_id=str(uuid.uuid4()), step_count=0)

    @staticmethod
    def _system_text(row_prompt: list[dict]) -> str:
        for msg in row_prompt:
            if msg.get("role") == "system":
                return msg.get("content", "")
        return row_prompt[0].get("content", "") if row_prompt else ""

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs: Any,
    ) -> AdvocacyObservation:
        if seed is not None:
            idx = seed % self._n
        else:
            idx = self._cursor
            self._cursor = (self._cursor + 1) % self._n

        row = self.dataset[idx]
        info = row["info"]
        self._current_passage = info["passage"]
        self._current_is_gold = bool(info["is_gold"])
        self._has_task = True

        self._state = State(episode_id=episode_id or str(uuid.uuid4()), step_count=0)

        return AdvocacyObservation(
            prompt=self._system_text(row["prompt"]),
            answer_to_defend=info["assigned_answer"],
            item_id=info["article_id"],
            reward=0.0,
            done=False,
        )

    def step(self, action: AdvocacyAction, **kwargs: Any) -> AdvocacyObservation:
        self._state.step_count += 1

        if not self._has_task:
            msg = "call reset() before step()"
            return AdvocacyObservation(
                prompt="",
                reward=0.0,
                done=True,
                error=msg,
                metadata={"error": msg},
            )

        text = action.text or ""
        claims = parse_claims(text)
        cites = parse_citations(text)

        cliff = claim_count_cliff(len(claims))
        ground = citation_grounding(claims, cites, self._current_passage)
        aggregate = (cliff + ground) / 2.0
        correctness = 1.0 if self._current_is_gold else 0.0

        # Single source of truth: canonical component order == weight order.
        # Must stay aligned with sophistry_bench_sprint._build_reward_funcs().
        metadata = {
            "aggregate_reward": aggregate,
            "correctness_reward": correctness,
            "n_claims": float(len(claims)),
            "n_citations": float(len(cites)),
            "alternation_canary": alternation_canary(text),
            "starts_with_canary": starts_with_canary(text),
            "length_band_canary": length_band_canary(text),
            "template_echo_canary": template_echo_canary(text),
        }
        reward = sum(w * c for w, c in zip(self.weights, metadata.values()))

        # Single-step episode: each task is exactly one advocacy turn.
        self._has_task = False
        return AdvocacyObservation(
            prompt="",
            reward=float(reward),
            done=True,
            metadata=dict(metadata),
            # Mirror into a declared field so the components survive the
            # framework's HTTP serialization (which strips ``metadata``).
            components=dict(metadata),
        )

    @property
    def state(self) -> State:
        return self._state
