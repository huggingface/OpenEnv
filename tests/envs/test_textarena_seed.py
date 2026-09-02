# SPDX-License-Identifier: BSD-3-Clause

"""Reproducibility tests for TextArena environment seed handling.

TextArena chooses its episode/word with the global ``random`` RNG and does not
apply ``reset(seed=...)`` to that selection, so the OpenEnv wrapper seeds the
global RNGs itself. These tests pin that behaviour down: same seed -> same word,
seed drives selection, and seeding a reset does not disturb the global RNG stream.
"""

import random

import pytest

# Skip the whole module if the optional textarena dependency is not installed.
pytest.importorskip("textarena", reason="textarena is not installed")

from envs.textarena_env.server.environment import TextArenaEnvironment


def _secret_word(env: TextArenaEnvironment):
    """Best-effort extraction of the hidden Wordle word from the underlying env."""
    inner = env._ta_env
    while inner is not None:
        game_state = getattr(getattr(inner, "state", None), "game_state", None)
        if isinstance(game_state, dict) and "secret_word" in game_state:
            return game_state["secret_word"]
        inner = getattr(inner, "env", None)
    return None


@pytest.fixture(scope="module")
def env():
    return TextArenaEnvironment(env_id="Wordle-v0", num_players=1)


def test_same_seed_same_word(env):
    env.reset(seed=123)
    first = _secret_word(env)
    env.reset(seed=123)
    second = _secret_word(env)
    assert first is not None
    assert first == second


def test_same_seed_across_instances():
    a = TextArenaEnvironment(env_id="Wordle-v0", num_players=1)
    b = TextArenaEnvironment(env_id="Wordle-v0", num_players=1)
    a.reset(seed=7)
    b.reset(seed=7)
    assert _secret_word(a) is not None
    assert _secret_word(a) == _secret_word(b)


def test_seed_drives_word_selection(env):
    words = set()
    for seed in range(12):
        env.reset(seed=seed)
        words.add(_secret_word(env))
    # Combined with the determinism tests above, more than one distinct word
    # confirms the seed actually drives selection rather than being ignored.
    assert len(words) > 1


def test_unseeded_reset_still_works(env):
    observation = env.reset()
    assert observation.done is False
    assert _secret_word(env) is not None


def test_seed_restores_global_rng_state(env):
    random.seed(999)
    baseline = random.random()

    random.seed(999)
    env.reset(seed=1)  # a seeded reset must not disturb the global RNG stream
    after = random.random()

    assert baseline == after
