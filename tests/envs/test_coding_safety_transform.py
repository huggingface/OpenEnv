# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for coding_env safety transform false-positive handling."""

from coding_env.models import CodeObservation
from coding_env.server.transforms import CodeQualityTransform, CodeSafetyTransform


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


def test_blocks_import_with_alias():
    observation = _apply_safety_transform("import os as operating_system")
    assert observation.reward == -1.0
    assert observation.metadata["safety_violation"] == "import os"


def test_blocks_subprocess_import():
    observation = _apply_safety_transform("import subprocess")
    assert observation.reward == -1.0
    assert observation.metadata["safety_violation"] == "import subprocess"


def test_blocks_from_subprocess_import():
    observation = _apply_safety_transform("from subprocess import run")
    assert observation.reward == -1.0
    assert observation.metadata["safety_violation"] == "import subprocess"


def test_blocks_from_os_path_import():
    observation = _apply_safety_transform("from os.path import join")
    assert observation.reward == -1.0
    assert observation.metadata["safety_violation"] == "import os"


def test_blocks_builtin_open_call():
    observation = _apply_safety_transform(
        "with open('f.txt') as f:\n    data = f.read()"
    )
    assert observation.reward == -1.0
    assert "safety_violation" in observation.metadata


def test_blocks_builtin_eval_call():
    observation = _apply_safety_transform("result = eval('1 + 1')")
    assert observation.reward == -1.0
    assert observation.metadata["safety_violation"] == "eval"


def test_blocks_builtin_exec_call():
    observation = _apply_safety_transform("exec('x = 1')")
    assert observation.reward == -1.0
    assert observation.metadata["safety_violation"] == "exec"


def test_blocks_builtin_import_call():
    observation = _apply_safety_transform("__import__('os')")
    assert observation.reward == -1.0
    assert observation.metadata["safety_violation"] == "__import__"


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


def test_quality_transform_handles_ast_recursion_error(monkeypatch):
    def raise_recursion_error(_code: str):
        raise RecursionError("pathologically nested code")

    monkeypatch.setattr("coding_env.server.transforms.ast.parse", raise_recursion_error)

    transform = CodeQualityTransform(concise_bonus=0.0, syntax_penalty=-0.2)
    observation = CodeObservation(
        stdout="",
        stderr="",
        exit_code=0,
        metadata={"last_code": "x = 1"},
    )

    transformed = transform(observation)

    assert isinstance(transformed, CodeObservation)
    assert transformed.reward == -0.2
