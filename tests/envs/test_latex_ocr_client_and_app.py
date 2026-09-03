# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The client's step parsing, and the app's web-interface default.

Both cases are silent: the client dropped every step's metadata without raising, and the "Try it"
tab simply did not appear.
"""

import os

import pytest

client_module = pytest.importorskip("latex_ocr_env.client")

LatexOCREnv = client_module.LatexOCREnv


def _parse(payload):
    """Drive `_parse_result` without opening a connection."""
    env = object.__new__(LatexOCREnv)
    return LatexOCREnv._parse_result(env, payload)


def test_step_metadata_survives_parsing():
    """`StepResult` exposes `metadata`; the client used to pass `info=` and swallow the TypeError."""
    result = _parse(
        {
            "reward": 0.5,
            "done": True,
            "observation": {"split": "train", "index": 0},
            "metadata": {"exact_match": False, "mode": "stream"},
        }
    )

    assert result.metadata == {"exact_match": False, "mode": "stream"}
    assert result.reward == 0.5 and result.done is True


def test_a_step_without_metadata_parses_to_none():
    result = _parse({"reward": 0.0, "done": True, "observation": {"split": "train"}})
    assert result.metadata is None


def test_importing_the_app_enables_the_web_interface(monkeypatch):
    """`create_app` only mounts /web when this is set, and nothing else set it."""
    monkeypatch.delenv("ENABLE_WEB_INTERFACE", raising=False)
    pytest.importorskip("gradio")

    import importlib

    app_module = importlib.import_module("latex_ocr_env.server.app")
    importlib.reload(app_module)

    assert os.environ["ENABLE_WEB_INTERFACE"] == "true"


def test_an_explicit_false_is_respected(monkeypatch):
    monkeypatch.setenv("ENABLE_WEB_INTERFACE", "false")
    pytest.importorskip("gradio")

    import importlib

    app_module = importlib.import_module("latex_ocr_env.server.app")
    importlib.reload(app_module)

    assert os.environ["ENABLE_WEB_INTERFACE"] == "false"
