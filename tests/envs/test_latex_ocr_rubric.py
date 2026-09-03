# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for the LaTeX OCR reward rubric.

The rubric only depends on the standard library, so we load it directly from
its file to keep the test fast and free of the environment's runtime deps
(datasets, pillow, openenv server).
"""

import importlib.util
import pathlib
import sys

import pytest


_RUBRIC_PATH = (
    pathlib.Path(__file__).parents[2]
    / "envs"
    / "latex_ocr_env"
    / "server"
    / "rubric.py"
)
_spec = importlib.util.spec_from_file_location("latex_ocr_rubric", _RUBRIC_PATH)
rubric = importlib.util.module_from_spec(_spec)
# Register before exec so dataclasses in the module can resolve their module.
sys.modules[_spec.name] = rubric
_spec.loader.exec_module(rubric)


def test_levenshtein_basic():
    assert rubric.levenshtein("abc", "abc") == 0
    assert rubric.levenshtein("abc", "abd") == 1
    assert rubric.levenshtein("", "abc") == 3
    assert rubric.levenshtein("kitten", "sitting") == 3


def test_normalize_collapses_whitespace():
    assert rubric.normalize_latex("  x  ^  2  ") == "x ^ 2"
    assert rubric.normalize_latex("a\n\tb") == "a b"


def test_exact_match_scores_one():
    g = rubric.LatexOCRRubric().grade("x^2 + 1", "x^2 + 1")
    assert g.exact_match is True
    assert g.reward == pytest.approx(1.0)
    assert g.char_error_rate == pytest.approx(0.0)


def test_whitespace_only_diff_is_exact():
    # LaTeX spacing is cosmetic: "x ^ 2 + 1" == "x^2+1" after canonicalization.
    g = rubric.LatexOCRRubric().grade("x ^ 2 + 1", "x^2+1")
    assert g.exact_match is True
    assert g.reward == pytest.approx(1.0)


def test_partial_match_is_between_zero_and_exact_cap():
    g = rubric.LatexOCRRubric(exact_weight=0.2).grade("x^2 + 2", "x^2 + 1")
    assert g.exact_match is False
    # partial answers are capped at (1 - exact_weight)
    assert 0.0 < g.reward <= 0.8
    assert 0.0 < g.char_error_rate < 1.0


def test_empty_prediction_scores_zero():
    g = rubric.LatexOCRRubric().grade("", "x^2 + 1")
    assert g.reward == pytest.approx(0.0)
    assert g.char_error_rate == pytest.approx(1.0)


def test_empty_target_only_empty_prediction_correct():
    r = rubric.LatexOCRRubric()
    assert r.grade("", "").exact_match is True
    assert r.grade("x", "").exact_match is False


def test_default_exact_weight_is_point_four():
    # Default 0.4 -> edit 0.6 / exact 0.4; partial answers cap at 0.6.
    r = rubric.LatexOCRRubric()
    assert r.exact_weight == pytest.approx(0.4)
    g = r.grade("x^2 + 2", "x^2 + 1")  # one char off, not exact
    assert g.exact_match is False
    assert 0.0 < g.reward <= 0.6


def test_exact_weight_bounds_validated():
    with pytest.raises(ValueError):
        rubric.LatexOCRRubric(exact_weight=1.5)


def test_clean_latex_strips_fences_and_delimiters():
    # The rubric cleans raw completions itself; cleaning is idempotent.
    assert rubric.clean_latex("```latex\nx^2 + 1\n```") == "x^2 + 1"
    assert rubric.clean_latex("$x^2 + 1$") == "x^2 + 1"
    assert rubric.clean_latex("x^2 + 1") == "x^2 + 1"  # already clean -> unchanged


def test_raw_fenced_completion_scored_correctly():
    # A raw completion wrapped in a code fence still scores as an exact match.
    g = rubric.LatexOCRRubric().grade("```latex\nx^2 + 1\n```", "x^2 + 1")
    assert g.exact_match is True
    assert g.reward == pytest.approx(1.0)


def test_whitespace_padding_reward_hack_is_penalized():
    # The core hack: correct answer + a wall of trailing whitespace. Whitespace-insensitive
    # scoring alone would give this 1.0; the length guard must drive the reward down.
    target = r"\frac{x^2}{2} + c"
    r = rubric.LatexOCRRubric()
    clean = r.grade(target, target)
    padded = r.grade(target + "\n" * 500, target)
    assert clean.reward == pytest.approx(1.0)  # clean answer unaffected
    assert clean.length_factor == pytest.approx(1.0)
    assert padded.exact_match is True  # content is still exact...
    assert padded.reward < 0.1  # ...but the padding is punished
    assert padded.length_factor < 0.1


def test_normal_spacing_within_allowance_is_free():
    # Ordinary LaTeX spacing must not trip the guard.
    target = r"\frac{x^2}{2} + c"
    g = rubric.LatexOCRRubric().grade(r"\frac{x^2}{2}  +  c", target)
    assert g.length_factor == pytest.approx(1.0)
    assert g.reward == pytest.approx(1.0)


def test_length_guard_can_be_disabled():
    target = r"\frac{x^2}{2} + c"
    g = rubric.LatexOCRRubric(overlong_ratio=0).grade(target + "\n" * 500, target)
    assert g.length_factor == pytest.approx(1.0)  # guard off -> old (hackable) behavior
    assert g.reward == pytest.approx(1.0)


def test_zero_length_factor_skips_edit_distance(monkeypatch):
    def unexpected_edit_distance(*_args):
        raise AssertionError("edit distance must not run after reward reaches zero")

    monkeypatch.setattr(rubric, "levenshtein", unexpected_edit_distance)
    g = rubric.LatexOCRRubric(overlong_ratio=1.0, overlong_floor=1).grade(
        "x" * 100,
        "y",
    )

    assert g.reward == pytest.approx(0.0)
    assert g.length_factor == pytest.approx(0.0)
