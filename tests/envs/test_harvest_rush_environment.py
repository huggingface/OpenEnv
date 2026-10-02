# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the Harvest Rush OpenEnv environment."""

import json
import os
import sys

import pytest

# Add the project root to the path for envs imports.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

# The env imports the ``harvest-rush-train`` package (canonical generator and
# reward) and, through it, the Harvest Rush engine. Neither is part of the
# repo's base test deps, so skip the whole module when they are unavailable.
pytest.importorskip("harvest_rush_train")
pytest.importorskip("harvest")

os.environ.setdefault("HRT_POOL_SIZE", "40")

from envs.harvest_rush_env.models import HarvestRushAction, HarvestRushObservation
from envs.harvest_rush_env.server import harvest_rush_environment as hre
from envs.harvest_rush_env.server.harvest_rush_environment import HarvestRushEnvironment
from harvest_rush_train.reward import score_choice
from openenv.core.env_server.serialization import serialize_observation


@pytest.fixture(scope="module")
def pool():
    return hre._pool()


def _reset_to(env, pool, kind):
    """Reset onto the first pooled example of the given kind."""
    idx = next(i for i, r in enumerate(pool) if r["info"]["kind"] == kind)
    obs = env.reset(seed=idx)
    return obs, pool[idx]


def test_reset_returns_a_contact_with_options(pool):
    env = HarvestRushEnvironment()
    obs = env.reset()
    assert isinstance(obs, HarvestRushObservation)
    assert obs.done is False and obs.reward == 0.0
    assert obs.system and obs.prompt
    assert "continue" in obs.options and len(obs.options) >= 2
    assert env.state.step_count == 0


def test_step_is_terminal_and_matches_the_canonical_reward(pool):
    env = HarvestRushEnvironment()
    for kind in ("creature", "prop", "rock"):
        for choice in ("continue", "swerve"):
            obs, row = _reset_to(env, pool, kind)
            if choice not in obs.options:
                continue
            out = env.step(HarvestRushAction(choice=choice))
            assert out.done is True
            assert out.reward == score_choice(choice, row["info"], "control_consistent")
            assert out.metadata["kind"] == kind and out.metadata["format_ok"] is True
            assert out.parsed_choice == choice


def test_driving_over_an_animal_is_never_rewarded(pool):
    env = HarvestRushEnvironment()
    animals = [i for i, r in enumerate(pool) if r["info"]["kind"] == "creature"]
    assert animals
    for idx in animals:
        env.reset(seed=idx)
        assert env.step(HarvestRushAction(choice="continue")).reward <= 0.0
        assert pool[idx]["answer"] != "continue"


def test_constant_policies_are_not_rewarded(pool):
    env = HarvestRushEnvironment()

    def mean_reward(policy):
        total = 0.0
        for idx in range(len(pool)):
            obs = env.reset(seed=idx)
            total += env.step(HarvestRushAction(choice=policy(obs))).reward
        return total / len(pool)

    always_continue = mean_reward(lambda o: "continue")
    always_swerve = mean_reward(
        lambda o: "swerve" if "swerve" in o.options else o.options[0]
    )
    oracle = 0.0
    for idx, row in enumerate(pool):
        env.reset(seed=idx)
        oracle += env.step(HarvestRushAction(choice=row["answer"])).reward
    oracle /= len(pool)
    assert always_continue < 0.7 and always_swerve < 0.7
    assert oracle > 0.95


def test_raw_message_is_parsed_like_the_verifiers_env(pool):
    env = HarvestRushEnvironment()
    obs, row = _reset_to(env, pool, "prop")
    out = env.step(HarvestRushAction(message='thinking... {"choice": "continue"}'))
    assert out.parsed_choice == "continue" and out.reward == 1.0
    env.reset(seed=0)
    bad = env.step(HarvestRushAction(message="I would rather not say"))
    assert bad.parsed_choice is None and bad.reward == 0.0
    assert bad.metadata["format_ok"] is False
    env.reset(seed=0)
    assert env.step(HarvestRushAction(choice="teleport")).reward == 0.0


def test_seeded_reset_is_deterministic_and_sessions_are_isolated(pool):
    a, b = HarvestRushEnvironment(), HarvestRushEnvironment()
    assert a.reset(seed=3).prompt == b.reset(seed=3).prompt
    a.reset(seed=1)
    b.reset(seed=2)
    assert a._row is not b._row
    assert a.state.episode_id != b.state.episode_id


def test_observation_serializes(pool):
    env = HarvestRushEnvironment()
    env.reset(seed=0)
    out = env.step(HarvestRushAction(choice="continue"))
    payload = serialize_observation(out)
    json.dumps(payload)
    assert payload["done"] is True and "reward" in payload
