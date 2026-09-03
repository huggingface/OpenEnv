# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for the LaTeX OCR Task API."""

import pytest
from latex_ocr_env.server import latex_ocr_environment as environment_module
from latex_ocr_env.server.latex_ocr_environment import LatexOCREnvironment


def test_stream_list_tasks_is_explicitly_unsupported(monkeypatch):
    monkeypatch.setattr(environment_module, "_split_total", lambda *_args: 150)
    env = LatexOCREnvironment(mode="stream")

    with pytest.raises(NotImplementedError, match="stream mode"):
        env.list_tasks("train")


def test_get_task_range_supports_python_slice_bounds(monkeypatch):
    monkeypatch.setattr(environment_module, "_load_split", lambda *_args: [None] * 5)
    env = LatexOCREnvironment(mode="materialize")

    assert [task["index"] for task in env.get_task_range("train", -2)] == [3, 4]
    assert [task["index"] for task in env.get_task_range("train", None, -1)] == [
        0,
        1,
        2,
        3,
    ]


@pytest.mark.parametrize(
    ("method_name", "args"),
    [
        ("num_tasks", ("validation",)),
        ("list_tasks", ("validation",)),
        ("get_task", ("validation", 0)),
        ("get_task_range", ("validation", 0, 1)),
    ],
)
def test_task_api_rejects_unconfigured_splits(monkeypatch, method_name, args):
    monkeypatch.setattr(environment_module, "_split_total", lambda *_args: -1)
    env = LatexOCREnvironment(mode="stream")

    with pytest.raises(ValueError, match="unknown split"):
        getattr(env, method_name)(*args)


def test_stream_task_range_rejects_oversized_requests(monkeypatch):
    monkeypatch.setattr(environment_module, "_split_total", lambda *_args: 20)
    monkeypatch.setattr(environment_module, "_STREAM_RANGE_CAP", 10)
    env = LatexOCREnvironment(mode="stream")

    with pytest.raises(ValueError, match="at most 10"):
        env.get_task_range("train", 0, 11)


def test_row_cap_is_the_answer_when_metadata_has_no_total(monkeypatch):
    """A missing `num_examples` gives -1, but a capped cursor stops at the cap.

    Reporting -1 there says "unknown" about a number we know exactly, and every progress field
    derived from it (`remaining`, `pct_done`) silently degrades with it.
    """
    monkeypatch.setattr(environment_module, "_split_total", lambda *_args: -1)
    monkeypatch.setenv("LATEX_OCR_MAX_ROWS", "50")
    env = LatexOCREnvironment(mode="stream")

    assert env.num_tasks("train") == 50


def test_row_cap_still_loses_to_a_smaller_real_total(monkeypatch):
    monkeypatch.setattr(environment_module, "_split_total", lambda *_args: 20)
    monkeypatch.setenv("LATEX_OCR_MAX_ROWS", "50")
    assert LatexOCREnvironment(mode="stream").num_tasks("train") == 20


def test_unknown_total_stays_unknown_without_a_cap(monkeypatch):
    monkeypatch.setattr(environment_module, "_split_total", lambda *_args: -1)
    monkeypatch.delenv("LATEX_OCR_MAX_ROWS", raising=False)
    assert LatexOCREnvironment(mode="stream").num_tasks("train") == -1


def test_step_reports_the_same_total_reset_did(monkeypatch):
    """`step` used `_split_total`, which ignores the cap, so the denominator moved mid-episode.

    A trainer reading progress saw `1/50` from reset and then `1/900` from step, for the same
    split, in the same episode.
    """
    monkeypatch.setattr(environment_module, "_split_total", lambda *_args: 900)
    monkeypatch.setenv("LATEX_OCR_MAX_ROWS", "50")
    env = LatexOCREnvironment(mode="stream")

    # One streamed row, without touching the network.
    monkeypatch.setattr(
        environment_module,
        "_encode_image",
        lambda _img: "",
        raising=False,
    )
    env._stream = iter(
        [{environment_module.IMAGE_COLUMN: None, environment_module.TEXT_COLUMN: "x^2"}]
    )
    env._stream_split = "train"
    env._cursor = 0
    env._exhausted = False

    obs_reset = env.reset(split="train")
    obs_step = env.step(environment_module.LatexOCRAction(latex="x^2"))

    assert obs_reset.total == 50
    assert obs_step.total == obs_reset.total, "the denominator changed mid-episode"
