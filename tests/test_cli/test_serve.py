# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for ``openenv serve``."""

from __future__ import annotations

import os
import socket
import subprocess
import sys
import time
from pathlib import Path
from unittest.mock import patch

import pytest
import requests
from openenv.cli.__main__ import app
from typer.testing import CliRunner


REPO_ROOT = Path(__file__).resolve().parents[2]
ECHO_ENV = REPO_ROOT / "envs" / "echo_env"
runner = CliRunner()


def _pick_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def test_serve_calls_uvicorn_with_echo_manifest() -> None:
    with patch("uvicorn.run") as mock_run:
        result = runner.invoke(
            app,
            [
                "serve",
                str(ECHO_ENV),
                "--port",
                "9911",
                "--host",
                "127.0.0.1",
            ],
            env={**os.environ, "PYTHONPATH": str(REPO_ROOT / "src")},
        )
    assert result.exit_code == 0, result.stdout
    mock_run.assert_called_once()
    (app_arg,), kwargs = mock_run.call_args
    assert app_arg == "server.app:app"
    assert kwargs["host"] == "127.0.0.1"
    assert kwargs["port"] == 9911
    assert kwargs["reload"] is False


def test_serve_rejects_invalid_env_dir() -> None:
    result = runner.invoke(
        app,
        ["serve", str(REPO_ROOT / "nonexistent_env_dir_xyz")],
        env={**os.environ, "PYTHONPATH": str(REPO_ROOT / "src")},
    )
    assert result.exit_code != 0


def test_serve_uses_manifest_port_when_omitted() -> None:
    with patch("uvicorn.run") as mock_run:
        result = runner.invoke(
            app,
            ["serve", str(ECHO_ENV), "--host", "127.0.0.1"],
            env={**os.environ, "PYTHONPATH": str(REPO_ROOT / "src")},
        )
    assert result.exit_code == 0, result.stdout
    _, kwargs = mock_run.call_args
    assert kwargs["port"] == 8000


@pytest.mark.integration
def test_serve_echo_env_health_subprocess() -> None:
    port = _pick_free_port()
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT / "src")
    cmd = [
        sys.executable,
        "-m",
        "openenv.cli",
        "serve",
        str(ECHO_ENV),
        "--port",
        str(port),
        "--host",
        "127.0.0.1",
    ]
    proc = subprocess.Popen(
        cmd,
        cwd=str(REPO_ROOT),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    try:
        deadline = time.time() + 60.0
        last_exc: Exception | None = None
        ok = False
        while time.time() < deadline:
            try:
                r = requests.get(f"http://127.0.0.1:{port}/health", timeout=2.0)
                if r.status_code == 200:
                    ok = True
                    break
            except Exception as exc:
                last_exc = exc
                if proc.poll() is not None:
                    out = proc.stdout.read() if proc.stdout else ""
                    pytest.fail(
                        f"serve process exited early (code={proc.returncode}): {out}"
                    )
                time.sleep(0.4)
        if not ok:
            out = proc.stdout.read() if proc.stdout else ""
            pytest.fail(f"/health never OK (last error={last_exc!r}): {out}")
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()
