# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import openenv.core
from openenv.core.harness import PiContext as HarnessPiContext
from openenv.core.pi import PiContext


def test_pi_context_loads_project_instructions_and_skills(tmp_path):
    (tmp_path / "AGENTS.md").write_text("Project rules", encoding="utf-8")
    skill_dir = tmp_path / ".pi" / "skills" / "shell"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text("Use the shell skill", encoding="utf-8")

    context = PiContext(
        project_root=tmp_path,
        extra_system_prompts=["Extra context"],
    )

    assert context.loaded_files() == ["AGENTS.md", ".pi/skills/shell/SKILL.md"]
    message = context.messages()[0]
    assert message["role"] == "system"
    assert "Pi-compatible environment session" in message["content"]
    assert "Extra context" in message["content"]
    assert "Project rules" in message["content"]
    assert "Use the shell skill" in message["content"]


def test_pi_context_is_exported_from_pi_and_harness_modules():
    assert HarnessPiContext is PiContext
    assert "PiContext" not in openenv.core.__all__


def test_pi_context_truncates_long_files(tmp_path):
    (tmp_path / "AGENTS.md").write_text("abcdef", encoding="utf-8")

    message = PiContext(project_root=tmp_path, max_context_chars=3).messages()[0]

    assert "abc\n\n[truncated by PiContext]" in message["content"]
