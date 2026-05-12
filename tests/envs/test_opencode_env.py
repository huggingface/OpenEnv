# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Smoke tests for ``opencode_env``.

The default suite runs in CI without any external dependencies (no E2B,
no LLM, no network). It covers:

  - Public API imports resolve.
  - The endpoint catalog (`vllm` / `openai` / `hf_router`) resolves
    explicit + env-var + default-value precedence correctly.
  - Pydantic models accept their expected shapes.
  - The `OpenCodeTask` coercion helper handles str / dict / `OpenCodeTask`.

A second class is marked ``@pytest.mark.integration`` and exercises the
deployed Space end-to-end. It only runs when ``E2B_API_KEY`` and at least
one endpoint credential are present and pytest is invoked with
``-m integration``.
"""

from __future__ import annotations

import os
import shlex
import sys

import pytest

# Make ``envs/`` importable when running from the repository root.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
_ENVS_DIR = os.path.join(_REPO_ROOT, "envs")
if _ENVS_DIR not in sys.path:
    sys.path.insert(0, _ENVS_DIR)


# ---------------------------------------------------------------------------
# Public API imports
# ---------------------------------------------------------------------------


def test_public_api_imports() -> None:
    """Top-level package re-exports the documented surface."""
    from opencode_env import (  # noqa: F401
        CommandResult,
        E2BSandboxBackend,
        OpenCodeConfig,
        OpenCodeEnv,
        OpenCodeSession,
        OpenCodeSessionFactory,
        OpenCodeState,
        OpenCodeTask,
        Provider,
        RolloutResult,
        RolloutTurn,
        SandboxBackend,
        SandboxHandle,
    )


def test_server_modules_import() -> None:
    """Server-side modules (FastAPI app, MCP env, catalog) import cleanly."""
    from opencode_env.server.app import app  # noqa: F401
    from opencode_env.server.catalog import (  # noqa: F401
        catalog_summary,
        ENDPOINT_KINDS,
        resolve_endpoint,
    )
    from opencode_env.server.opencode_environment import (  # noqa: F401
        OpenCodeEnvironment,
    )


# ---------------------------------------------------------------------------
# Endpoint catalog — pure logic, no network
# ---------------------------------------------------------------------------


def test_catalog_kinds() -> None:
    from opencode_env.server.catalog import ENDPOINT_KINDS

    assert ENDPOINT_KINDS == ("vllm", "openai", "hf_router")


def test_resolve_endpoint_explicit_args_win(monkeypatch: pytest.MonkeyPatch) -> None:
    """Explicit args beat env vars beat catalog defaults."""
    from opencode_env.server.catalog import resolve_endpoint

    monkeypatch.setenv("OPENAI_API_KEY", "from-env")
    r = resolve_endpoint(
        "openai",
        base_url="https://custom.example/v1",
        api_key="from-arg",
        model="gpt-from-arg",
    )
    assert r.kind == "openai"
    assert r.base_url == "https://custom.example/v1"
    assert r.api_key == "from-arg"
    assert r.model == "gpt-from-arg"
    assert r.disable_thinking_default is False


def test_resolve_endpoint_env_var_used_when_arg_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from opencode_env.server.catalog import resolve_endpoint

    monkeypatch.setenv("OPENAI_API_KEY", "key-from-env")
    monkeypatch.setenv("OPENAI_MODEL", "gpt-4o")
    r = resolve_endpoint("openai")
    assert r.api_key == "key-from-env"
    assert r.model == "gpt-4o"
    assert r.base_url == "https://api.openai.com/v1"


def test_resolve_endpoint_normalizes_v1_suffix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Base URL gets ``/v1`` appended if missing, otherwise left alone."""
    from opencode_env.server.catalog import resolve_endpoint

    monkeypatch.setenv("VLLM_URL", "https://my-vllm.example/")
    monkeypatch.setenv("VLLM_API_KEY", "x")
    r1 = resolve_endpoint("vllm")
    assert r1.base_url == "https://my-vllm.example/v1"

    monkeypatch.setenv("VLLM_URL", "https://my-vllm.example/v1")
    r2 = resolve_endpoint("vllm")
    assert r2.base_url == "https://my-vllm.example/v1"


def test_resolve_endpoint_unknown_kind_raises() -> None:
    from opencode_env.server.catalog import resolve_endpoint

    with pytest.raises(ValueError, match="unknown endpoint kind"):
        resolve_endpoint("bogus", base_url="x", api_key="y", model="z")


def test_resolve_endpoint_missing_creds_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from opencode_env.server.catalog import resolve_endpoint

    # Strip any inherited env vars.
    for k in ("OPENAI_API_KEY", "OPENAI_BASE_URL", "OPENAI_MODEL"):
        monkeypatch.delenv(k, raising=False)
    with pytest.raises(ValueError, match="no api_key"):
        resolve_endpoint("openai")


def test_catalog_summary_shape() -> None:
    from opencode_env.server.catalog import catalog_summary

    summary = catalog_summary()
    assert {entry["kind"] for entry in summary} == {"vllm", "openai", "hf_router"}
    for entry in summary:
        assert {
            "base_url_env",
            "api_key_env",
            "model_env",
            "configured",
        } <= entry.keys()


# ---------------------------------------------------------------------------
# Models + task coercion
# ---------------------------------------------------------------------------


def test_rollout_result_serializes_round_trip() -> None:
    from opencode_env import CommandResult, RolloutResult, RolloutTurn

    r = RolloutResult(
        task_id="t1",
        sandbox_id="sbx-1",
        reward=0.75,
        agent_exit_code=0,
        wall_s=12.5,
        mode="transparent_proxy",
        setup_results=[CommandResult(cmd="pip install pandas", exit_code=0)],
        verify_results=[CommandResult(cmd="pytest", exit_code=1, stderr="boom")],
        proxy_turns=[
            RolloutTurn(
                turn=1,
                finish_reason="stop",
                completion_tokens=["hi"],
                per_token_logps=[-0.1],
                latency_s=0.2,
            )
        ],
        files={"/home/user/workdir/x.py": "print('x')"},
    )
    blob = r.model_dump_json()
    rebuilt = RolloutResult.model_validate_json(blob)
    assert rebuilt.reward == 0.75
    assert rebuilt.verify_results[0].exit_code == 1
    assert rebuilt.proxy_turns[0].completion_tokens == ["hi"]


def test_opencode_task_coerce_str() -> None:
    from opencode_env import OpenCodeTask

    t = OpenCodeTask.coerce("write fizzbuzz.py")
    assert t.instruction == "write fizzbuzz.py"
    assert t.setup_shell is None
    assert t.upload_files == {}


def test_opencode_task_coerce_dict() -> None:
    from opencode_env import OpenCodeTask

    t = OpenCodeTask.coerce({"instruction": "x", "setup_shell": "pip install pandas"})
    assert t.instruction == "x"
    assert t.setup_shell == "pip install pandas"


def test_opencode_task_coerce_existing_passthrough() -> None:
    from opencode_env import OpenCodeTask

    src = OpenCodeTask(instruction="y")
    assert OpenCodeTask.coerce(src) is src


def test_opencode_task_coerce_rejects_unknown_type() -> None:
    from opencode_env import OpenCodeTask

    with pytest.raises(TypeError, match="Cannot coerce"):
        OpenCodeTask.coerce(42)  # type: ignore[arg-type]


def test_start_proxy_keeps_upstream_key_out_of_command() -> None:
    """The proxy API key must be passed via env, not shell argv."""
    from opencode_env import OpenCodeConfig, OpenCodeSessionFactory

    class FakeExecResult:
        exit_code = 0
        stdout = "ok"
        stderr = ""

    class FakeBgJob:
        def wait(self, timeout: float | None = None) -> int:
            return 0

        def kill(self) -> None:
            pass

    class FakeSandbox:
        sandbox_id = "fake-sandbox"

        def __init__(self) -> None:
            self.started_cmd: str | None = None
            self.started_envs: dict[str, str] | None = None
            self.written: dict[str, str] = {}

        def exec(self, *args, **kwargs) -> FakeExecResult:
            return FakeExecResult()

        def start_bg(self, cmd: str, *, envs=None, cwd=None) -> FakeBgJob:
            self.started_cmd = cmd
            self.started_envs = envs
            return FakeBgJob()

        def write_text(self, path: str, content: str) -> None:
            self.written[path] = content

        def read_text(self, path: str) -> str:
            return ""

        def exists(self, path: str) -> bool:
            return path in self.written

        def kill(self) -> None:
            pass

    class NoopInstallFactory(OpenCodeSessionFactory):
        def _exec_with_retry(self, *args, **kwargs):
            return FakeExecResult()

    secret = "sk-test '$(leak)"
    model = "provider/model'; touch /tmp/pwn #"
    config = OpenCodeConfig(
        base_url="https://example.test/v1?x='y",
        api_key=secret,
        model=model,
    )
    sandbox = FakeSandbox()
    factory = NoopInstallFactory(
        config=config,
        sandbox_backend=object(),  # unused by this protected-method test
        mode="transparent_proxy",
    )

    factory._start_proxy(sandbox)

    assert sandbox.started_cmd is not None
    assert sandbox.started_envs == {"OPENCODE_UPSTREAM_API_KEY": secret}
    assert secret not in sandbox.started_cmd
    assert "--upstream-api-key" not in sandbox.started_cmd

    argv = shlex.split(sandbox.started_cmd.split("&&", 1)[1].split(">", 1)[0].strip())
    assert argv[argv.index("--upstream-url") + 1] == config.base_url
    assert argv[argv.index("--model-override") + 1] == model


def test_interception_cli_reads_upstream_key_from_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from openenv.core.harness.sandbox import interception

    captured = {}

    def fake_serve(cfg) -> None:
        captured["cfg"] = cfg

    monkeypatch.setattr(interception, "serve", fake_serve)
    monkeypatch.setenv("OPENCODE_UPSTREAM_API_KEY", "sk-from-env")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "interception.py",
            "--upstream-url",
            "https://example.test/v1",
        ],
    )

    interception.main()

    assert captured["cfg"].upstream_api_key == "sk-from-env"


# ---------------------------------------------------------------------------
# Integration — only runs when E2B + endpoint creds are present and the
# user explicitly opts in via ``pytest -m integration``.
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_run_rollout_e2e_via_deployed_space() -> None:
    """End-to-end: hit the deployed Space, write binary_search.py, verify it.

    Requires:

      - ``E2B_API_KEY`` (the Space already has it as a secret)
      - ``OPENAI_API_KEY`` (used for the run; the test passes it explicitly so
        the Space doesn't need it as a secret)

    Skipped if either is missing.
    """
    if not os.environ.get("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY not set — required for the e2e test")

    import asyncio

    from opencode_env import OpenCodeEnv
    from opencode_env.client import _extract_text
    from opencode_env.models import RolloutResult

    SPACE = os.environ.get(
        "OPENCODE_ENV_SPACE", "https://adithyask-opencode-env.hf.space"
    )

    async def _go() -> RolloutResult:
        async with OpenCodeEnv(base_url=SPACE) as env:
            await env.reset()
            raw = await env.call_tool(
                "run_rollout",
                endpoint="openai",
                api_key=os.environ["OPENAI_API_KEY"],
                model="gpt-4o-mini",
                instruction=(
                    "Create binary_search.py exposing "
                    "def binary_search(arr, target) -> int. Use a relative path. "
                    "Standard binary search; return -1 if not found."
                ),
                setup=[],
                verify=[
                    "test -f /home/user/workdir/binary_search.py",
                    "python -c \"import sys; sys.path.insert(0, '/home/user/workdir'); "
                    "import binary_search; "
                    "assert binary_search.binary_search([1,2,3,4,5], 3) == 2; print('OK')\"",
                ],
                template="opencode-rl",
                agent_timeout_s=600,
            )
            return RolloutResult.model_validate_json(_extract_text(raw))

    result = asyncio.run(_go())
    assert result.reward == 1.0, (
        f"expected reward=1.0 got {result.reward}: {result.error}"
    )
    assert result.proxy_turns, "expected at least one captured LLM turn"
    assert any(f.endswith("/binary_search.py") for f in result.files), (
        f"expected binary_search.py in workdir, got {list(result.files)}"
    )
