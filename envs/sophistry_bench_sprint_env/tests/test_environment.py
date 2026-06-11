# Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.
"""Tests for the sophistry-bench sprint OpenEnv environment."""

from sophistry_bench_sprint_env.models import AdvocacyAction, AdvocacyObservation


def test_advocacy_action_carries_text():
    a = AdvocacyAction(text="<claim>x</claim>")
    assert a.text == "<claim>x</claim>"


def test_advocacy_observation_defaults():
    o = AdvocacyObservation(prompt="P", answer_to_defend="A", item_id="id1")
    assert o.prompt == "P"
    assert o.answer_to_defend == "A"
    assert o.item_id == "id1"
    assert o.reward == 0.0
    assert o.done is False
    assert o.metadata == {}


def test_client_parses_step_result():
    from sophistry_bench_sprint_env.client import SophistryBenchSprintEnv
    from sophistry_bench_sprint_env.models import AdvocacyAction, AdvocacyObservation

    # Exercise the pure parsing hooks without a live server.
    client = SophistryBenchSprintEnv.__new__(SophistryBenchSprintEnv)
    payload = client._step_payload(AdvocacyAction(text="<claim>x</claim>"))
    assert payload["text"] == "<claim>x</claim>"

    raw = {
        "observation": {
            "prompt": "",
            "answer_to_defend": "",
            "item_id": "",
            "reward": 0.5,
            "done": True,
            "metadata": {"aggregate_reward": 0.5},
        },
        "reward": 0.5,
        "done": True,
        "info": {},
    }
    result = client._parse_result(raw)
    assert isinstance(result.observation, AdvocacyObservation)
    assert result.observation.metadata["aggregate_reward"] == 0.5
    assert result.reward == 0.5
    assert result.done is True


from sophistry_bench_sprint_env.server.sophistry_bench_sprint_environment import (
    SophistryBenchSprintEnvironment,
)


def _env():
    # Small dataset keeps the test fast; reads the bundled QuALITY split.
    return SophistryBenchSprintEnvironment(n_items=2, passage_chars=500, seed=0)


def test_reset_returns_task_observation():
    env = _env()
    obs = env.reset(seed=0)
    assert obs.done is False
    assert obs.reward == 0.0
    assert obs.prompt  # non-empty system prompt
    assert "DEFEND THIS ANSWER" in obs.prompt
    assert obs.answer_to_defend in obs.prompt
    assert obs.item_id  # article id present


def test_reset_is_deterministic_for_fixed_seed():
    a = _env().reset(seed=3)
    b = _env().reset(seed=3)
    assert (a.item_id, a.answer_to_defend, a.prompt) == (
        b.item_id,
        b.answer_to_defend,
        b.prompt,
    )


import asyncio

from sophistry_bench_sprint import load_environment
from sophistry_bench_sprint_env.models import AdvocacyAction

_METADATA_KEYS = {
    "aggregate_reward",
    "correctness_reward",
    "n_claims",
    "n_citations",
    "alternation_canary",
    "starts_with_canary",
    "length_band_canary",
    "template_echo_canary",
}


def test_step_returns_all_components_and_is_done():
    env = _env()
    env.reset(seed=0)
    text = "".join(f"<claim>point {i}</claim>" for i in range(8))  # 8 claims, 0 cites
    obs = env.step(AdvocacyAction(text=text))
    assert obs.done is True
    assert set(obs.metadata.keys()) == _METADATA_KEYS
    assert obs.metadata["n_claims"] == 8.0
    assert obs.metadata["n_citations"] == 0.0
    # claim_count_cliff(8)=1.0, citation_grounding(claims, [], passage)=0.0 -> mean 0.5
    assert obs.metadata["aggregate_reward"] == 0.5
    # default weights -> reward == aggregate
    assert obs.reward == 0.5


def test_step_before_reset_errors_gracefully():
    env = _env()
    obs = env.step(AdvocacyAction(text="<claim>x</claim>"))
    assert obs.done is True
    assert obs.reward == 0.0
    assert "error" in obs.metadata


def test_aggregate_matches_canonical_verifiers_reward():
    """Anti-drift: OpenEnv aggregate must equal the PI Hub aggregate_reward."""
    env = _env()
    env.reset(seed=0)
    text = "<claim>alpha</claim><cite>beta gamma delta epsilon zeta</cite>"
    obs = env.step(AdvocacyAction(text=text))

    vf_env = load_environment(n_items=2, passage_chars=500, seed=0)
    # Newer verifiers wrap the reward Rubric in a RubricGroup, so funcs live on
    # the inner rubric; older versions expose them directly. aggregate_reward is index 0.
    rubric = vf_env.rubric
    if not getattr(rubric, "funcs", None) and getattr(rubric, "rubrics", None):
        rubric = rubric.rubrics[0]
    aggregate_fn = rubric.funcs[0]  # aggregate_reward is index 0
    completion = [{"role": "assistant", "content": text}]
    state = {"info": {"passage": env._current_passage}}
    canonical = asyncio.run(
        aggregate_fn(prompt=[], completion=completion, answer="", state=state)
    )
    assert abs(obs.metadata["aggregate_reward"] - canonical) < 1e-9


def test_metadata_survives_wire_serialization_round_trip():
    """Lock in the wire contract: the framework strips base ``metadata`` from the
    serialized observation, but the declared ``components`` field survives and the
    typed client re-populates ``metadata`` from it on the way back."""
    from openenv.core.env_server.serialization import serialize_observation
    from sophistry_bench_sprint_env.client import SophistryBenchSprintEnv

    env = _env()
    env.reset(seed=0)
    obs = env.step(
        AdvocacyAction(text="".join(f"<claim>c{i}</claim>" for i in range(8)))
    )

    # Real server-side serialization. Returns
    # {"observation": {...}, "reward": float, "done": bool}; the obs dict
    # excludes reward/done/metadata but keeps declared subclass fields.
    payload = serialize_observation(obs)
    obs_dict = payload["observation"]
    assert "metadata" not in obs_dict  # framework strips base metadata
    assert set(obs_dict["components"].keys()) == _METADATA_KEYS

    # Reconstruct the wire payload in the shape ``_parse_result`` reads.
    wire = {
        "observation": obs_dict,
        "reward": payload["reward"],
        "done": payload["done"],
    }
    client = SophistryBenchSprintEnv.__new__(SophistryBenchSprintEnv)
    result = client._parse_result(wire)
    assert set(result.observation.metadata.keys()) == _METADATA_KEYS
    assert result.reward == obs.reward
