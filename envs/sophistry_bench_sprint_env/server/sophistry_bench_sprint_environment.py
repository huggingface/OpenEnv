# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import os
import uuid
from typing import Any, Optional

from openenv.core.env_server import Environment
from openenv.core.env_server.types import State

from sophistry_bench_sprint import (
    aggregate_reward,
    alternation_canary,
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


# Canonical reward-component order. The i-th SPRINT_WEIGHTS entry weights the
# i-th key here. MUST stay aligned with sophistry_bench_sprint._build_reward_funcs()
# and with the SPRINT_WEIGHTS docs in the README.
_COMPONENT_KEYS = (
    "aggregate_reward",
    "correctness_reward",
    "n_claims",
    "n_citations",
    "alternation_canary",
    "starts_with_canary",
    "length_band_canary",
    "template_echo_canary",
)

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


class SophistryBenchSprintEnvironment(
    Environment[AdvocacyAction, AdvocacyObservation, State]
):
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
        expose_correctness: Optional[bool] = None,
    ):
        super().__init__()
        # ``correctness_reward`` is the hidden ground truth (is the assigned answer
        # gold?). It is withheld from the wire observation by default so a naive
        # harness that forwards the whole observation to the policy can't leak it.
        # Trusted measurement code can opt in (SPRINT_EXPOSE_CORRECTNESS=1) to get
        # it back in ``metadata``/``components``; it always counts toward the
        # weighted reward regardless.
        self.expose_correctness = (
            expose_correctness
            if expose_correctness is not None
            else os.getenv("SPRINT_EXPOSE_CORRECTNESS", "0").lower()
            in ("1", "true", "yes")
        )
        self.n_items = (
            n_items if n_items is not None else int(os.getenv("SPRINT_N_ITEMS", "50"))
        )
        self.passage_chars = (
            passage_chars
            if passage_chars is not None
            else int(os.getenv("SPRINT_PASSAGE_CHARS", "2000"))
        )
        self.build_seed = (
            seed if seed is not None else int(os.getenv("SPRINT_SEED", "0"))
        )
        self.weights = weights if weights is not None else _weights_from_env()
        # Guard the constructor ``weights=`` path too (the env-var path is already
        # length-checked). A wrong-length vector would otherwise be silently
        # truncated by the ``zip`` in ``step``, scrambling the reward with no error.
        if len(self.weights) != len(_COMPONENT_KEYS):
            raise ValueError(
                f"weights must have {len(_COMPONENT_KEYS)} values, "
                f"got {len(self.weights)}"
            )

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
            raise RuntimeError(
                "sprint dataset is empty; check bundled quality_dev.json"
            )

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

        # reward left as the base default (None): no action scored on reset.
        return AdvocacyObservation(
            prompt=self._system_text(row["prompt"]),
            answer_to_defend=info["assigned_answer"],
            item_id=info["article_id"],
            done=False,
        )

    def step(self, action: AdvocacyAction, **kwargs: Any) -> AdvocacyObservation:
        if not self._has_task:
            msg = "call reset() before step()"
            return AdvocacyObservation(
                prompt="",
                reward=0.0,
                done=True,
                error=msg,
                metadata={"error": msg},
            )

        # Count only scored steps (after the reset guard).
        self._state.step_count += 1

        text = action.text or ""
        claims = parse_claims(text)
        cites = parse_citations(text)

        # Imported from the canonical package (sophistry-bench-sprint>=0.1.6) — the
        # aggregate formula is no longer reproduced here, so it can't drift.
        aggregate = aggregate_reward(claims, cites, self._current_passage)
        correctness = 1.0 if self._current_is_gold else 0.0

        # Full score vector — the weighted reward is computed over all eight
        # components (correctness included if it carries a weight).
        scores = {
            "aggregate_reward": aggregate,
            "correctness_reward": correctness,
            "n_claims": float(len(claims)),
            "n_citations": float(len(cites)),
            "alternation_canary": alternation_canary(text),
            "starts_with_canary": starts_with_canary(text),
            "length_band_canary": length_band_canary(text),
            "template_echo_canary": template_echo_canary(text),
        }
        # Weight by explicit key (not dict order) so a future reorder of the dict
        # above can't silently scramble the weight<->component mapping. strict=True
        # backstops the length invariant enforced in __init__.
        reward = sum(
            w * scores[k] for w, k in zip(self.weights, _COMPONENT_KEYS, strict=True)
        )

        # Surfaced components withhold the hidden ground truth unless opted in, so
        # a harness that forwards the observation can't leak ``correctness_reward``.
        surfaced = dict(scores)
        if not self.expose_correctness:
            del surfaced["correctness_reward"]

        # Single-step episode: each task is exactly one advocacy turn.
        self._has_task = False
        return AdvocacyObservation(
            prompt="",
            reward=float(reward),
            done=True,
            metadata=dict(surfaced),
            # Mirror into a declared field so the components survive the
            # framework's HTTP serialization (which strips ``metadata``).
            components=dict(surfaced),
        )

    @property
    def state(self) -> State:
        return self._state

    @property
    def current_passage(self) -> str:
        """Passage of the active episode (the reading-comprehension text already
        embedded in the reset prompt — not hidden ground truth). Empty before the
        first ``reset``."""
        return self._current_passage
