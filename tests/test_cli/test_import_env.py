# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for deterministic OpenEnv environment import."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient
from openenv.cli.__main__ import app
from openenv.cli.importers.ors import detect_ors_dependencies, detect_ors_environments
from openenv.cli.importers.verifiers import detect_verifiers_environments
from openenv.core.env_server.mcp_types import CallToolAction, ListToolsAction
from typer.testing import CliRunner


runner = CliRunner()


def _write_fake_ors_sdk(root: Path) -> None:
    ors_dir = root / "ors"
    ors_dir.mkdir(parents=True)
    (ors_dir / "__init__.py").write_text(
        "from .environment import Environment, ListToolsOutput, Split, TextBlock, "
        "ToolOutput, ToolSpec\n",
        encoding="utf-8",
    )
    (ors_dir / "environment.py").write_text(
        """
class _Model:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def model_dump(self):
        return dict(self.__dict__)


class Split(_Model):
    pass


class ToolSpec(_Model):
    pass


class ListToolsOutput(_Model):
    pass


class TextBlock(_Model):
    def __init__(self, text, detail=None, type="text"):
        super().__init__(text=text, detail=detail, type=type)


class ToolOutput(_Model):
    pass


class _RunToolSuccess:
    ok = True

    def __init__(self, output):
        self.output = output


class RunToolOutput:
    def __init__(self, output):
        self.root = _RunToolSuccess(output)


class Environment:
    def __init__(self, task_spec=None, secrets=None):
        self.task_spec = task_spec or {}
        self.secrets = secrets or {}
        self.setup_called = False
        self.teardown_called = False

    def setup(self):
        self.setup_called = True

    def teardown(self):
        self.teardown_called = True
""".lstrip(),
        encoding="utf-8",
    )


def _write_single_fake_ors_env(root: Path) -> None:
    _write_fake_ors_sdk(root)
    (root / "demo_env.py").write_text(
        """
from ors import Environment, ListToolsOutput, Split, TextBlock, ToolOutput, ToolSpec


class DemoEnvironment(Environment):
    @classmethod
    def list_splits(cls):
        return [Split(name="train", type="train")]

    @classmethod
    def list_tasks(cls, split):
        return [{"id": "alpha", "goal": "answer"}]

    @classmethod
    def num_tasks(cls, split):
        return 1

    @classmethod
    def get_task(cls, split, index):
        return cls.list_tasks(split)[index]

    @classmethod
    def get_task_range(cls, split, start=None, stop=None):
        return cls.list_tasks(split)[slice(start, stop)]

    @classmethod
    def list_tools(cls):
        return ListToolsOutput(
            tools=[
                ToolSpec(
                    name="answer",
                    description="Submit an answer",
                    input_schema={"type": "object", "properties": {"value": {"type": "string"}}},
                )
            ]
        )

    def list_task_tools(self):
        return ListToolsOutput(
            tools=[
                ToolSpec(
                    name="hint",
                    description="Get a hint",
                    input_schema={"type": "object", "properties": {}},
                )
            ]
        )

    def get_prompt(self):
        return [TextBlock(text=f"Task: {self.task_spec['id']}")]

    def _call_tool(self, name, input):
        return __import__("ors.environment").environment.RunToolOutput(
            ToolOutput(
                blocks=[TextBlock(text=f"{name}:{input.get('value', '')}")],
                metadata={"tool": name},
                reward=1.0,
                finished=True,
            )
        )
""".lstrip(),
        encoding="utf-8",
    )


def _write_fake_verifiers_sdk(root: Path) -> None:
    verifiers_dir = root / "verifiers"
    verifiers_dir.mkdir(parents=True)
    (verifiers_dir / "__init__.py").write_text(
        """
class Environment:
    pass


class Rubric:
    async def score_rollout(self, state):
        answer = state.get("answer") or state.get("task", {}).get("answer")
        completion = state.get("completion") or []
        text = completion[-1].get("content", "") if completion else ""
        state["reward"] = 1.0 if answer and answer in text else 0.0
        state["metrics"] = {"contains_answer": state["reward"]}


class SingleTurnEnv(Environment):
    def __init__(self, dataset, eval_dataset=None, rubric=None):
        self._dataset = dataset
        self._eval_dataset = eval_dataset or dataset
        self.rubric = rubric or Rubric()

    def get_dataset(self):
        return self._dataset

    def get_eval_dataset(self):
        return self._eval_dataset
""".lstrip(),
        encoding="utf-8",
    )


def _write_single_fake_verifiers_env(root: Path) -> None:
    _write_fake_verifiers_sdk(root)
    (root / "vf_demo.py").write_text(
        """
import verifiers as vf


def load_environment() -> vf.Environment:
    train = [
        {"prompt": [{"role": "user", "content": "Say alpha"}], "answer": "alpha", "example_id": 0},
        {"prompt": [{"role": "user", "content": "Say beta"}], "answer": "beta", "example_id": 1},
    ]
    eval_rows = [
        {"prompt": [{"role": "user", "content": "Say gamma"}], "answer": "gamma", "example_id": 0}
    ]
    return vf.SingleTurnEnv(dataset=train, eval_dataset=eval_rows)
""".lstrip(),
        encoding="utf-8",
    )


def test_ors_detector_finds_environment_class_without_importing_source(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "envs").mkdir()
    (source / "envs" / "sample.py").write_text(
        """
from ors import Environment as ORSEnvironment

SIDE_EFFECT = 0


class SampleEnv(ORSEnvironment):
    pass
""".lstrip(),
        encoding="utf-8",
    )

    matches = detect_ors_environments(source)

    assert len(matches) == 1
    assert matches[0].class_name == "SampleEnv"
    assert matches[0].module_path == "envs.sample"
    assert matches[0].source_type == "ors"


def test_ors_detector_finds_openreward_environments_import_path(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "sample.py").write_text(
        """
from openreward.environments import Environment


class SampleEnv(Environment):
    pass
""".lstrip(),
        encoding="utf-8",
    )

    matches = detect_ors_environments(source)

    assert len(matches) == 1
    assert matches[0].class_name == "SampleEnv"
    assert detect_ors_dependencies(source) == ["openreward"]


def test_ors_detector_returns_no_matches_for_unrelated_source(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "plain.py").write_text("class Plain: pass\n", encoding="utf-8")

    assert detect_ors_environments(source) == []


def test_verifiers_detector_finds_load_environment_without_importing_source(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "demo.py").write_text(
        """
import verifiers as vf

SIDE_EFFECT = 0


def load_environment() -> vf.Environment:
    raise RuntimeError("should not import")
""".lstrip(),
        encoding="utf-8",
    )

    matches = detect_verifiers_environments(source)

    assert len(matches) == 1
    assert matches[0].source_type == "verifiers"
    assert matches[0].class_name == "load_environment"
    assert matches[0].module_path == "demo"


def test_import_command_requires_env_class_when_multiple_ors_classes(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "a.py").write_text(
        "from ors import Environment\nclass First(Environment): pass\n",
        encoding="utf-8",
    )
    (source / "b.py").write_text(
        "from ors import Environment\nclass Second(Environment): pass\n",
        encoding="utf-8",
    )

    result = runner.invoke(
        app,
        [
            "import",
            str(source),
            "--name",
            "imported_env",
            "--output-dir",
            str(tmp_path),
        ],
    )

    assert result.exit_code != 0
    assert "Multiple environment entrypoints" in result.output
    assert "env" in result.output
    assert "class" in result.output


def test_import_command_detects_ors_and_generates_working_wrapper(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    _write_single_fake_ors_env(source)
    output_dir = tmp_path / "out"

    with patch("openenv.cli.commands.import_env._generate_uv_lock", return_value=True):
        result = runner.invoke(
            app,
            [
                "import",
                str(source),
                "--name",
                "imported_env",
                "--output-dir",
                str(output_dir),
            ],
        )

    assert result.exit_code == 0, result.output
    env_dir = output_dir / "imported_env"
    assert (env_dir / "server" / "imported_env_environment.py").exists()
    assert (env_dir / "vendor" / "source" / "demo_env.py").exists()
    assert (env_dir / "vendor" / "source" / "ors" / "environment.py").exists()

    sys.path.insert(0, str(output_dir))
    try:
        from imported_env.server.imported_env_environment import (  # type: ignore
            ImportedEnvironment,
        )

        env = ImportedEnvironment()
        assert env.list_splits() == [{"name": "train", "type": "train"}]
        assert env.get_task("train", 0) == {"id": "alpha", "goal": "answer"}
        with pytest.raises(RuntimeError, match="reset"):
            ImportedEnvironment().step(
                CallToolAction(tool_name="answer", arguments={"value": "42"})
            )

        reset_obs = env.reset(split="train", index=0)
        assert reset_obs.metadata["task_spec"] == {"id": "alpha", "goal": "answer"}
        assert reset_obs.metadata["prompt"][0]["text"] == "Task: alpha"

        tools_obs = env.step(ListToolsAction())
        assert [tool.name for tool in tools_obs.tools] == ["answer", "hint"]

        call_obs = env.step(
            CallToolAction(tool_name="answer", arguments={"value": "42"})
        )
        assert call_obs.reward == 1.0
        assert call_obs.done is True
        assert call_obs.result["blocks"][0]["text"] == "answer:42"

        from imported_env.server.app import app as generated_app  # type: ignore

        client = TestClient(generated_app)
        assert client.get("/list_environments").json() == ["imported_env"]
        assert client.get("/imported_env/splits").status_code == 200
        mcp_tools = client.post(
            "/mcp",
            json={
                "jsonrpc": "2.0",
                "method": "tools/list",
                "params": {},
                "id": 1,
            },
        ).json()
        assert mcp_tools["result"]["tools"][0]["name"] == "answer"
        mcp_call = client.post(
            "/mcp",
            json={
                "jsonrpc": "2.0",
                "method": "tools/call",
                "params": {"name": "answer", "arguments": {"value": "42"}},
                "id": 2,
            },
        ).json()
        assert "reset" in mcp_call["error"]["message"]
    finally:
        sys.path.remove(str(output_dir))


def test_import_command_handles_source_module_matching_generated_package(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    _write_fake_ors_sdk(source)
    (source / "collision_env.py").write_text(
        """
from ors import Environment, Split


class CollisionEnvironment(Environment):
    @classmethod
    def list_splits(cls):
        return [Split(name="train", type="train")]
""".lstrip(),
        encoding="utf-8",
    )
    output_dir = tmp_path / "out"

    with patch("openenv.cli.commands.import_env._generate_uv_lock", return_value=True):
        result = runner.invoke(
            app,
            [
                "import",
                str(source),
                "--name",
                "collision_env",
                "--output-dir",
                str(output_dir),
            ],
        )

    assert result.exit_code == 0, result.output
    sys.path.insert(0, str(output_dir))
    try:
        from collision_env.server.collision_env_environment import (  # type: ignore
            CollisionEnvironment,
        )

        env = CollisionEnvironment()
        assert env.list_splits() == [{"name": "train", "type": "train"}]
    finally:
        sys.path.remove(str(output_dir))


def test_import_command_excludes_common_secret_files(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    _write_single_fake_ors_env(source)
    (source / ".env").write_text("TOKEN=secret\n", encoding="utf-8")
    (source / "secrets.yaml").write_text("token: secret\n", encoding="utf-8")
    (source / "private.pem").write_text("secret\n", encoding="utf-8")
    output_dir = tmp_path / "out"

    with patch("openenv.cli.commands.import_env._generate_uv_lock", return_value=True):
        result = runner.invoke(
            app,
            [
                "import",
                str(source),
                "--name",
                "secret_env",
                "--output-dir",
                str(output_dir),
            ],
        )

    assert result.exit_code == 0, result.output
    vendor_dir = output_dir / "secret_env" / "vendor" / "source"
    assert not (vendor_dir / ".env").exists()
    assert not (vendor_dir / "secrets.yaml").exists()
    assert not (vendor_dir / "private.pem").exists()


def test_import_command_uses_detected_ors_dependency(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "demo.py").write_text(
        """
from openreward.environments import Environment


class DemoEnvironment(Environment):
    pass
""".lstrip(),
        encoding="utf-8",
    )
    output_dir = tmp_path / "out"

    with patch("openenv.cli.commands.import_env._generate_uv_lock", return_value=True):
        result = runner.invoke(
            app,
            [
                "import",
                str(source),
                "--name",
                "openreward_env",
                "--output-dir",
                str(output_dir),
            ],
        )

    assert result.exit_code == 0, result.output
    requirements = (
        output_dir / "openreward_env" / "server" / "requirements.txt"
    ).read_text(encoding="utf-8")
    pyproject = (output_dir / "openreward_env" / "pyproject.toml").read_text(
        encoding="utf-8"
    )
    assert "openreward" in requirements
    assert "openreward" in pyproject
    assert "ors-sdk" not in requirements
    assert "ors-sdk" not in pyproject


def test_import_command_detects_verifiers_and_generates_working_wrapper(
    tmp_path: Path,
) -> None:
    source = tmp_path / "vf_source"
    source.mkdir()
    _write_single_fake_verifiers_env(source)
    output_dir = tmp_path / "out"

    with patch("openenv.cli.commands.import_env._generate_uv_lock", return_value=True):
        result = runner.invoke(
            app,
            [
                "import",
                str(source),
                "--name",
                "vf_imported_env",
                "--output-dir",
                str(output_dir),
            ],
        )

    assert result.exit_code == 0, result.output
    env_dir = output_dir / "vf_imported_env"
    assert (env_dir / "server" / "vf_imported_env_environment.py").exists()
    assert (env_dir / "vendor" / "vf_source" / "vf_demo.py").exists()
    assert (env_dir / "vendor" / "vf_source" / "verifiers" / "__init__.py").exists()

    sys.path.insert(0, str(output_dir))
    try:
        from vf_imported_env.server.vf_imported_env_environment import (  # type: ignore
            VfImportedEnvironment,
        )

        env = VfImportedEnvironment()
        assert env.list_splits() == [
            {"name": "train", "type": "train"},
            {"name": "eval", "type": "validation"},
        ]
        with pytest.raises(RuntimeError, match="reset"):
            VfImportedEnvironment().step(
                CallToolAction(tool_name="submit", arguments={"completion": "alpha"})
            )
        assert env.num_tasks("train") == 2
        assert env.get_task("train", 1)["answer"] == "beta"

        reset_obs = env.reset(split="train", index=0)
        assert reset_obs.metadata["prompt"][0]["content"] == "Say alpha"

        tools_obs = env.step(ListToolsAction())
        assert [tool.name for tool in tools_obs.tools] == ["submit"]

        call_obs = env.step(
            CallToolAction(tool_name="submit", arguments={"completion": "alpha"})
        )
        assert call_obs.reward == 1.0
        assert call_obs.done is True
        assert call_obs.result["reward"] == 1.0

        from vf_imported_env.server.app import app as generated_app  # type: ignore

        client = TestClient(generated_app)
        assert client.get("/list_environments").json() == ["vf_imported_env"]
        assert client.get("/vf_imported_env/splits").status_code == 200
    finally:
        sys.path.remove(str(output_dir))
