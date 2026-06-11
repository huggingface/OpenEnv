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
