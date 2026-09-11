import random
import threading

import pytest
from textarena_env.models import TextArenaAction, TextArenaMessage
from textarena_env.server.environment import TextArenaEnvironment


def test_convert_messages_coalesces_consecutive_characters():
    env = object.__new__(TextArenaEnvironment)

    raw_messages = [
        (0, "[", "PROMPT"),
        (0, "GAME", "PROMPT"),
        (0, "]", "PROMPT"),
        (1, "A", "MESSAGE"),
        (1, "B", "MESSAGE"),
        (2, "!", "MESSAGE"),
    ]

    converted = env._convert_messages(raw_messages)

    assert converted == [
        TextArenaMessage(sender_id=0, content="[GAME]", category="PROMPT"),
        TextArenaMessage(sender_id=1, content="AB", category="MESSAGE"),
        TextArenaMessage(sender_id=2, content="!", category="MESSAGE"),
    ]


def test_wordle_reset_clears_accumulated_state():
    """Test that resetting Wordle environment clears accumulated observation state.

    This test verifies the workaround for TextArena's LLMObservationWrapper,
    which accumulates observations in self.full_observations across resets.
    """
    pytest.importorskip("textarena", reason="textarena not installed")
    env = TextArenaEnvironment(
        env_id="Wordle-v0",
        num_players=1,
    )

    # First episode
    obs1 = env.reset()
    prompt1_len = len(obs1.prompt)

    # Make a move to accumulate some state
    env.step(TextArenaAction(message="[CRANE]"))

    # Second episode - should NOT accumulate from first episode
    obs2 = env.reset()
    prompt2_len = len(obs2.prompt)

    # Make another move
    env.step(TextArenaAction(message="[STALE]"))

    # Third episode - should NOT accumulate from previous episodes
    obs3 = env.reset()
    prompt3_len = len(obs3.prompt)

    # All prompts should be the same length (no accumulation)
    assert prompt1_len == prompt2_len, (
        f"Episode 2 accumulated state: {prompt1_len} -> {prompt2_len}"
    )
    assert prompt2_len == prompt3_len, (
        f"Episode 3 accumulated state: {prompt2_len} -> {prompt3_len}"
    )

    # Verify the prompts are actually the same content
    assert obs1.prompt == obs2.prompt
    assert obs2.prompt == obs3.prompt


def _secret_word(env: TextArenaEnvironment) -> str:
    """Read the puzzle from TextArena's own game state, not from the prompt.

    Wordle's prompt is static instructions, so hashing the observation would
    report every episode as identical whether or not the seed took effect.
    """
    return env._ta_env.state.game_state["secret_word"]


def test_reset_seed_is_forwarded_to_textarena():
    """A fixed seed must give the same episode, and different seeds different ones."""
    pytest.importorskip("textarena", reason="textarena not installed")
    env = TextArenaEnvironment(env_id="Wordle-v0", num_players=1)

    seeded = []
    for _ in range(3):
        env.reset(seed=1234)
        seeded.append(_secret_word(env))

    env.reset(seed=999)
    other_seed = _secret_word(env)

    assert len(set(seeded)) == 1, f"seed=1234 produced {sorted(set(seeded))}"
    assert other_seed != seeded[0], "a different seed produced the same episode"


def test_reset_seed_is_reproducible_across_instances():
    """The same seed must survive constructing a fresh environment."""
    pytest.importorskip("textarena", reason="textarena not installed")
    first = TextArenaEnvironment(env_id="Wordle-v0", num_players=1)
    first.reset(seed=1234)

    second = TextArenaEnvironment(env_id="Wordle-v0", num_players=1)
    second.reset(seed=1234)

    assert _secret_word(first) == _secret_word(second)


def test_reset_without_seed_still_varies():
    """Forwarding the seed must not accidentally pin unseeded episodes."""
    pytest.importorskip("textarena", reason="textarena not installed")
    env = TextArenaEnvironment(env_id="Wordle-v0", num_players=1)

    words = set()
    for _ in range(8):
        env.reset()
        words.add(_secret_word(env))

    assert len(words) > 1, "unseeded resets should not be deterministic"


def test_seeded_reset_restores_global_rng_state():
    """A seeded reset must leave the process-global RNG where it found it."""
    pytest.importorskip("textarena", reason="textarena not installed")
    env = TextArenaEnvironment(env_id="Wordle-v0", num_players=1)

    random.seed(42)
    expected = random.random()

    random.seed(42)
    env.reset(seed=1234)

    assert random.random() == expected, "seeded reset leaked into the global RNG"


def test_seeded_reset_does_not_perturb_unseeded_session():
    """An unseeded session must get the same episode whether or not another
    session performed a seeded reset in between."""
    pytest.importorskip("textarena", reason="textarena not installed")
    unseeded = TextArenaEnvironment(env_id="Wordle-v0", num_players=1)
    seeded = TextArenaEnvironment(env_id="Wordle-v0", num_players=1)

    random.seed(42)
    unseeded.reset()
    baseline = _secret_word(unseeded)

    random.seed(42)
    seeded.reset(seed=1234)
    unseeded.reset()

    assert _secret_word(unseeded) == baseline


def test_concurrent_unseeded_reset_cannot_steal_a_seeded_draw(monkeypatch):
    """Force another session to reset in the gap between TextArena seeding the
    global RNG and drawing the episode. The seeded episode must be unaffected."""
    pytest.importorskip("textarena", reason="textarena not installed")
    seeded = TextArenaEnvironment(env_id="Wordle-v0", num_players=1)
    other = TextArenaEnvironment(env_id="Wordle-v0", num_players=1)

    seeded.reset(seed=1234)
    expected = _secret_word(seeded)

    window_open = threading.Event()
    other_done = threading.Event()
    original_seed = random.seed

    def seed_then_yield(*args, **kwargs):
        original_seed(*args, **kwargs)
        window_open.set()
        other_done.wait(timeout=0.5)

    monkeypatch.setattr(random, "seed", seed_then_yield)

    def unseeded_reset():
        window_open.wait(timeout=5)
        other.reset()
        other_done.set()

    thread = threading.Thread(target=unseeded_reset)
    thread.start()
    seeded.reset(seed=1234)
    thread.join(timeout=5)

    assert not thread.is_alive(), "the unseeded reset never completed"
    assert _secret_word(seeded) == expected, "another session stole the seeded draw"


@pytest.fixture
def fake_textarena(monkeypatch):
    """Exercise the wrapper in required CI without optional TextArena/NLTK data."""
    from types import SimpleNamespace

    from textarena_env.server import environment

    class Game:
        def reset(self, num_players, seed=None):
            random.seed(seed)
            self.draw = random.random()
            self.state = SimpleNamespace(turn=0, game_state={})

        def step(self, message):
            self.draw = random.random()
            return False, {}

        def get_observation(self):
            return 0, []

    monkeypatch.setattr(
        environment, "_TEXTARENA_MODULE", SimpleNamespace(make=lambda **kw: Game())
    )
    rng_state = random.getstate()
    yield lambda: TextArenaEnvironment(env_id="test", download_nltk=False)
    random.setstate(rng_state)


def test_seed_forwarding_without_optional_dependencies(fake_textarena):
    env = fake_textarena()
    before = random.getstate()
    env.reset(seed=0)
    assert env._ta_env.draw == random.Random(0).random()
    assert random.getstate() == before


def test_failed_seeded_reset_restores_rng(fake_textarena, monkeypatch):
    env = fake_textarena()
    before = random.getstate()

    def fail(**kwargs):
        random.seed(kwargs["seed"])
        raise RuntimeError("reset failed")

    monkeypatch.setattr(env._ta_env, "reset", fail)
    with pytest.raises(RuntimeError, match="reset failed"):
        env.reset(seed=1234)
    assert random.getstate() == before


@pytest.mark.parametrize("operation", ["reset", "step", "construct"])
def test_seed_window_excludes_other_session_operations(
    fake_textarena, monkeypatch, operation
):
    from concurrent.futures import ThreadPoolExecutor

    from textarena_env.server import environment

    seeded = fake_textarena()
    other = fake_textarena()
    window_open = threading.Event()
    attempted = threading.Event()
    entered = threading.Event()
    release = threading.Event()
    lock = threading.Lock()

    class ObservedLock:
        def __enter__(self):
            if window_open.is_set():
                attempted.set()
            lock.acquire()

        def __exit__(self, *args):
            lock.release()

    original_reset = seeded._ta_env.reset

    def paused_reset(**kwargs):
        original_reset(**kwargs)
        window_open.set()
        assert release.wait(5), "seed window was never released"

    def draw(*args, **kwargs):
        entered.set()
        other._ta_env.draw = random.random()
        return False, {}

    monkeypatch.setattr(environment, "_RNG_LOCK", ObservedLock())
    monkeypatch.setattr(seeded._ta_env, "reset", paused_reset)
    monkeypatch.setattr(
        other._ta_env, operation if operation != "construct" else "reset", draw
    )
    if operation == "construct":
        monkeypatch.setattr(
            environment._TEXTARENA_MODULE, "make", lambda **kw: other._ta_env
        )
    before = random.getstate()
    expected = random.Random()
    expected.setstate(before)
    expected_draw = expected.random()

    def run_other():
        if operation == "construct":
            fake_textarena()
        elif operation == "step":
            other.step(TextArenaAction(message="draw"))
        else:
            other.reset()

    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(seeded.reset, seed=1234)
        try:
            assert window_open.wait(5)
            second = pool.submit(run_other)
            assert attempted.wait(1), "operation did not acquire the RNG lock"
            assert not entered.is_set(), "operation entered the seeded RNG window"
        finally:
            release.set()
        first.result(timeout=5)
        second.result(timeout=5)
    assert other._ta_env.draw == expected_draw
    assert random.getstate() == expected.getstate()
