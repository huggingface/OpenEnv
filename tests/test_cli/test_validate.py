# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for the openenv validate command and runtime validation utilities."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

from openenv.cli.__main__ import app
from openenv.cli._validation import validate_running_environment
from typer.testing import CliRunner


runner = CliRunner()


class _MockResponse:
    """Minimal mock response object for requests.get/post tests."""

    def __init__(self, status_code: int, payload: dict | None = None):
        self.status_code = status_code
        self._payload = payload

    def json(self) -> dict:
        if self._payload is None:
            raise ValueError("No JSON payload")
        return self._payload


def _write_minimal_valid_env(
    env_dir: Path,
    *,
    main_signature: str = "def main():",
    main_invocation: str = "main()",
) -> None:
    """Create a minimal local environment that passes local validation."""
    (env_dir / "server").mkdir(parents=True)

    (env_dir / "openenv.yaml").write_text(
        "spec_version: 1\nname: test_env\ntype: space\nruntime: fastapi\napp: server.app:app\nport: 8000\n"
    )
    (env_dir / "uv.lock").write_text("")
    (env_dir / "pyproject.toml").write_text(
        "[project]\n"
        'name = "test-env"\n'
        'version = "0.1.0"\n'
        'dependencies = ["openenv-core>=0.2.0"]\n'
        "\n"
        "[project.scripts]\n"
        'server = "server.app:main"\n'
    )
    (env_dir / "server" / "app.py").write_text(
        f"{main_signature}\n    return None\n\nif __name__ == '__main__':\n    {main_invocation}\n"
    )


def test_validate_running_environment_success() -> None:
    """Runtime validator returns passing criteria for a conforming server."""

    def _fake_get(url: str, timeout: float) -> _MockResponse:
        if url.endswith("/openapi.json"):
            return _MockResponse(
                200,
                {
                    "info": {"version": "1.0.0"},
                    "paths": {
                        "/health": {},
                        "/metadata": {},
                        "/schema": {},
                        "/mcp": {},
                        "/reset": {},
                        "/step": {},
                        "/state": {},
                    },
                },
            )
        if url.endswith("/health"):
            return _MockResponse(200, {"status": "healthy"})
        if url.endswith("/metadata"):
            return _MockResponse(200, {"name": "EchoEnv", "description": "Echo env"})
        if url.endswith("/schema"):
            return _MockResponse(
                200,
                {"action": {"type": "object"}, "observation": {}, "state": {}},
            )
        raise AssertionError(f"Unexpected GET url: {url}")

    def _fake_post(url: str, json: dict, timeout: float) -> _MockResponse:
        if url.endswith("/mcp"):
            return _MockResponse(
                200,
                {
                    "jsonrpc": "2.0",
                    "id": None,
                    "error": {"code": -32600, "message": "Invalid Request"},
                },
            )
        raise AssertionError(f"Unexpected POST url: {url}")

    with patch("openenv.cli._validation.requests.get", side_effect=_fake_get):
        with patch("openenv.cli._validation.requests.post", side_effect=_fake_post):
            report = validate_running_environment("http://localhost:8000")

    assert report["passed"] is True
    assert report["standard_version"] == "1.0.0"
    assert report["mode"] == "simulation"
    assert report["validation_type"] == "running_environment"
    assert report["summary"]["passed_count"] == 6
    assert report["summary"]["total_count"] == 6
    assert report["summary"]["failed_criteria"] == []


def test_validate_running_environment_failure() -> None:
    """Runtime validator marks report as failed when criteria fail."""

    def _fake_get(url: str, timeout: float) -> _MockResponse:
        if url.endswith("/openapi.json"):
            return _MockResponse(
                200,
                {
                    "info": {"version": "1.0.0"},
                    "paths": {
                        "/health": {},
                        "/metadata": {},
                        "/schema": {},
                        "/mcp": {},
                    },
                },
            )
        if url.endswith("/health"):
            return _MockResponse(200, {"status": "healthy"})
        if url.endswith("/metadata"):
            return _MockResponse(500, {"detail": "boom"})
        if url.endswith("/schema"):
            return _MockResponse(
                200,
                {"action": {"type": "object"}, "observation": {}, "state": {}},
            )
        raise AssertionError(f"Unexpected GET url: {url}")

    def _fake_post(url: str, json: dict, timeout: float) -> _MockResponse:
        if url.endswith("/mcp"):
            return _MockResponse(
                200,
                {
                    "jsonrpc": "2.0",
                    "id": None,
                    "error": {"code": -32600, "message": "Invalid Request"},
                },
            )
        raise AssertionError(f"Unexpected POST url: {url}")

    with patch("openenv.cli._validation.requests.get", side_effect=_fake_get):
        with patch("openenv.cli._validation.requests.post", side_effect=_fake_post):
            report = validate_running_environment("http://localhost:8000")

    assert report["passed"] is False
    metadata_checks = [c for c in report["criteria"] if c["id"] == "metadata_endpoint"]
    assert metadata_checks
    assert metadata_checks[0]["passed"] is False
    assert report["summary"]["passed_count"] == 5
    assert report["summary"]["total_count"] == 6
    assert report["summary"]["failed_criteria"] == ["metadata_endpoint"]


def test_validate_command_runtime_target_outputs_json() -> None:
    """CLI validates runtime targets and prints JSON report."""
    mock_report = {
        "target": "https://example.com",
        "validation_type": "running_environment",
        "standard_version": "1.0.0",
        "passed": True,
        "criteria": [],
    }

    with patch(
        "openenv.cli.commands.validate.validate_running_environment",
        return_value=mock_report,
    ) as mock_validate:
        result = runner.invoke(app, ["validate", "https://example.com"])

    assert result.exit_code == 0
    assert json.loads(result.output) == mock_report
    mock_validate.assert_called_once_with("https://example.com", timeout_s=5.0)


def test_validate_command_local_path_still_works(tmp_path: Path) -> None:
    """CLI local validation remains backward compatible."""
    env_dir = tmp_path / "test_env"
    _write_minimal_valid_env(env_dir)

    result = runner.invoke(app, ["validate", str(env_dir)])

    assert result.exit_code == 0
    assert "[OK]" in result.output


def test_validate_command_local_json_output(tmp_path: Path) -> None:
    """CLI can emit JSON report for local validation via --json."""
    env_dir = tmp_path / "test_env"
    _write_minimal_valid_env(env_dir)

    result = runner.invoke(app, ["validate", str(env_dir), "--json"])

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["validation_type"] == "local_environment"
    assert payload["passed"] is True
    assert payload["summary"]["passed_count"] == 1
    assert payload["summary"]["total_count"] == 1
    assert payload["summary"]["failed_criteria"] == []


def test_validate_command_accepts_main_call_with_arguments(tmp_path: Path) -> None:
    """Local validation accepts a guarded main(...) call with arguments."""
    env_dir = tmp_path / "test_env"
    _write_minimal_valid_env(
        env_dir,
        main_signature="def main(port: int = 8000):",
        main_invocation="main(port=8000)",
    )

    result = runner.invoke(app, ["validate", str(env_dir)])

    assert result.exit_code == 0
    assert "main() function not callable" not in result.output
    assert "[OK]" in result.output


def test_validate_command_rejects_nested_main_guard(tmp_path: Path) -> None:
    """Local validation requires the __main__ guard at module scope."""
    env_dir = tmp_path / "test_env"
    _write_minimal_valid_env(env_dir)
    (env_dir / "server" / "app.py").write_text(
        "def main():\n    return None\n\n"
        "def wrapper():\n"
        "    if __name__ == '__main__':\n"
        "        main()\n"
    )

    result = runner.invoke(app, ["validate", str(env_dir)])

    assert result.exit_code != 0
    assert "main() function not callable" in result.output


def test_validate_command_syntax_error_fallback_requires_dunder_main(
    tmp_path: Path,
) -> None:
    """Syntax-error fallback still requires the literal __main__ guard string."""
    env_dir = tmp_path / "test_env"
    _write_minimal_valid_env(env_dir)
    (env_dir / "server" / "app.py").write_text(
        "def main(:\n    return None\n\nif __name__ ==\n    main(\n"
    )

    result = runner.invoke(app, ["validate", str(env_dir)])

    assert result.exit_code != 0
    assert "main() function not callable" in result.output


def test_validate_command_rejects_mixed_path_and_url(tmp_path: Path) -> None:
    """CLI rejects mixing a local path argument with --url mode."""
    env_dir = tmp_path / "test_env"
    _write_minimal_valid_env(env_dir)

    result = runner.invoke(
        app,
        ["validate", str(env_dir), "--url", "http://localhost:8000"],
    )

    assert result.exit_code != 0
    assert "Cannot combine a local path argument with --url" in result.output
