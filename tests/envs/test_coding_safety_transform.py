# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for coding_env safety transform false-positive handling."""

import os
import sys
from pathlib import Path

# Add the project root and src to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from envs.coding_env.models import CodeObservation
from envs.coding_env.server.transforms import CodeSafetyTransform


def _apply_safety_transform(code: str) -> CodeObservation:
    transform = CodeSafetyTransform()
    observation = CodeObservation(
        stdout="",
        stderr="",
        exit_code=0,
        metadata={"last_code": code},
    )
    transformed = transform(observation)
    assert isinstance(transformed, CodeObservation)
    return transformed


def test_blocks_real_dangerous_import():
    observation = _apply_safety_transform("import os\nprint('x')")
    assert observation.reward == -1.0
    assert "safety_violation" in observation.metadata


def test_blocks_builtin_open_call():
    observation = _apply_safety_transform("with open('f.txt') as f:\n    data = f.read()")
    assert observation.reward == -1.0
    assert "safety_violation" in observation.metadata


def test_does_not_flag_string_literal_with_dangerous_text():
    observation = _apply_safety_transform("print('import os')")
    assert observation.reward == 0.0
    assert "safety_violation" not in observation.metadata


def test_does_not_flag_user_defined_myopen_function():
    observation = _apply_safety_transform(
        "def myopen():\n    return 1\nresult = myopen()"
    )
    assert observation.reward == 0.0
    assert "safety_violation" not in observation.metadata


def test_does_not_flag_attribute_method_named_exec():
    observation = _apply_safety_transform(
        "class DB:\n"
        "    def exec(self, sql):\n"
        "        return sql\n"
        "db = DB()\n"
        "result = db.exec('SELECT 1')"
    )
    assert observation.reward == 0.0
    assert "safety_violation" not in observation.metadata
