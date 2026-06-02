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
