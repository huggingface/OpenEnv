# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Reward selection and the per-agent seam.

Both decide things that are wrong silently rather than loudly. A reward picked by the wrong rule
trains the policy on the wrong objective and the run looks healthy throughout; a model name formatted
the way one harness expects and another does not resolves to a different model with no error.
"""

from __future__ import annotations

import pytest

rollout = pytest.importorskip("openenv.harbor.rollout")
seams = pytest.importorskip("openenv.harbor.seams")

_pick_reward = rollout._pick_reward


# --- reward selection -------------------------------------------------------
def test_no_rewards_means_ungraded_not_zero():
    """The verifier never ran. Scoring that as 0 makes a dead sandbox look like a wrong answer."""
    value, key = _pick_reward({})
    assert value is None and key == ""


def test_a_single_key_is_used_whatever_it_is_called():
    assert _pick_reward({"accuracy": 0.75}) == (0.75, "accuracy")


def test_a_key_named_reward_wins_when_several_exist():
    value, key = _pick_reward({"reward": 1.0, "steps": 12.0, "latency": 0.4})
    assert (value, key) == (1.0, "reward")


def test_ambiguous_rewards_raise_rather_than_guess():
    """Inventing a rule to combine keys is inventing reward semantics."""
    with pytest.raises(ValueError, match="reward_key"):
        _pick_reward({"correctness": 1.0, "style": 0.5})


def test_an_explicit_key_overrides_the_default_rule():
    assert _pick_reward({"reward": 1.0, "partial": 0.25}, "partial") == (
        0.25,
        "partial",
    )


def test_an_unknown_explicit_key_raises_and_names_the_options():
    with pytest.raises(ValueError, match="not in"):
        _pick_reward({"reward": 1.0}, "nope")


def test_zero_is_a_real_reward_and_is_not_treated_as_missing():
    """`0.0` is falsy; a truthiness check here would report a wrong answer as ungraded."""
    value, key = _pick_reward({"reward": 0.0})
    assert value == 0.0 and key == "reward"


def test_values_are_coerced_to_float():
    value, _ = _pick_reward({"reward": 1})
    assert isinstance(value, float)


# --- the model name a harness is given --------------------------------------
@pytest.mark.parametrize(
    "served,expected",
    [
        ("Qwen/Qwen3.5-9B", "Qwen3.5-9B"),  # the case that caused real breakage
        ("org/team/model", "model"),
        ("plain-model", "plain-model"),
    ],
)
def test_agent_facing_model_takes_the_leaf(served, expected):
    """A served id with a slash produces `provider/org/model` once a seam adds its prefix.

    gemini-cli requires exactly `provider/model_name`, cline-cli wants `provider:model`, and
    harnesses disagree about splitting on the first or the last slash, so the same string resolves
    to different models. Taking the leaf removes the ambiguity. Safe because the harness-facing name
    only has to route to the proxy: the proxy overwrites `model` with the served id upstream.
    """
    assert seams.agent_facing_model(served) == expected


def test_an_empty_served_model_raises_rather_than_producing_an_empty_name():
    """An empty name would be formatted into the seam's prefix and sent as `openai/`.

    The harness would then fail somewhere downstream with a confusing error, so this refuses at the
    point where the cause is still obvious.
    """
    with pytest.raises(ValueError, match="empty"):
        seams.agent_facing_model("")


def test_resolve_cannot_be_bypassed_by_a_seam():
    """Normalisation lives inside `resolve`, so no seam can format the raw served id by mistake."""
    for name in seams.SEAMS:
        seam = seams.get(name)
        model_name, kwargs, agent_env, proc_env = seam.resolve(
            base_url="http://proxy:8100", session="sess-123", model="Qwen/Qwen3.5-9B"
        )
        assert "Qwen/Qwen3.5-9B" not in model_name, f"{name} leaked the slashed id"
        assert isinstance(kwargs, dict)
        assert isinstance(agent_env, dict)
        assert isinstance(proc_env, dict)


def test_resolve_threads_the_session_id_through_as_the_key():
    """The API key is the capture session id: that is the whole multiplexing scheme."""
    seam = seams.get("opencode")
    _, kwargs, agent_env, proc_env = seam.resolve(
        base_url="http://proxy:8100", session="sess-abc", model="m"
    )
    blob = f"{kwargs}{agent_env}{proc_env}"
    assert "sess-abc" in blob
    assert "http://proxy:8100" in blob


def test_every_seam_declares_a_known_dialect():
    known = {"openai_chat", "openai_responses", "anthropic", "google"}
    for name in seams.SEAMS:
        assert seams.get(name).dialect in known, f"{name} has an unknown dialect"


def test_unknown_seam_raises():
    with pytest.raises((KeyError, ValueError)):
        seams.get("definitely-not-a-harness")


# --- concurrency ------------------------------------------------------------
def test_process_env_lock_is_usable_from_several_event_loops():
    """It guards `os.environ`, which is global to the process, not to an event loop.

    Rollouts arrive on several loops: the env server answers each request on its own, and a caller
    using `asyncio.run` per rollout creates another. An `asyncio.Lock` binds to the first loop that
    uses it and then raises "is bound to a different event loop" for every other one, which failed
    100% of concurrent rollouts while passing every sequential test.
    """
    import asyncio
    import threading

    lock = rollout._PROC_ENV_LOCK
    assert isinstance(lock, type(threading.Lock())), "must not be an asyncio primitive"

    errors: list[str] = []
    done: list[int] = []

    def worker(tag: int) -> None:
        async def body() -> None:
            with lock:
                await asyncio.sleep(0.005)

        try:
            asyncio.run(body())
            done.append(tag)
        except Exception as exc:  # noqa: BLE001 - the failure being pinned
            errors.append(f"{type(exc).__name__}: {exc}")

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, errors[:2]
    assert len(done) == 8


# --- ok must mean usable ----------------------------------------------------
def test_a_fatal_capture_finding_makes_the_rollout_unusable():
    """`ok=True` with zero model calls is the worst shape a result can take.

    A rollout that reached the verifier but produced no captured calls used to come back `ok=True`
    with zero trainable tokens, because only trace reconciliation could clear `ok` and it agreed:
    both sides were empty. One such rollout carried reward=1.0, so a trainer filtering on `ok` would
    have accepted a row containing nothing at all alongside a positive reward.
    """
    import asyncio
    from pathlib import Path

    result = asyncio.run(
        rollout.run_rollout(
            task_dir=Path("/definitely/not/a/task"),
            harness="opencode",
            sandbox="e2b",
            registry=_FakeRegistry(),
            intercept_url="http://127.0.0.1:9",
            model="m",
            trials_dir=Path("/tmp"),
        )
    )
    # The trial cannot even be built, so this must be a result rather than an exception.
    assert result.ok is False
    assert result.reward is None
    assert result.error


class _FakeSession:
    session_id = "sess-test"

    def __init__(self) -> None:
        self.graph = None


class _FakeRegistry:
    """Enough of `SessionRegistry` for `run_rollout` to mint and delete a session."""

    def create(self, session_id=None, **_kwargs):
        return _FakeSession()

    def delete(self, _session_id):
        return True
