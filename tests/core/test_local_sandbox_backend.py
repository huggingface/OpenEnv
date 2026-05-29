# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from pathlib import Path

from openenv.core.harness.sandbox.local_backend import LocalSandboxBackend


def test_local_sandbox_backend_basic_lifecycle(tmp_path):
    backend = LocalSandboxBackend(root_dir=str(tmp_path), create_virtualenv=False)
    sandbox = backend.create(envs={"OPENENV_TEST": "1"})

    root = Path(sandbox.sandbox_home).parent
    try:
        result = sandbox.exec(
            'printf "%s|%s|%s" "$HOME" "$TMPDIR" "$OPENENV_TEST"',
            cwd="/testbed",
        )
        assert result.exit_code == 0
        assert result.stdout == f"{sandbox.sandbox_home}|{sandbox.tmp_dir}|1"

        sandbox.write_text("/testbed/hello.txt", "hello\n")
        assert sandbox.read_text("/testbed/hello.txt") == "hello\n"
        assert sandbox.exists("/testbed/hello.txt")

        job = sandbox.start_bg("sleep 0.1", cwd="/testbed")
        assert job.wait(timeout=2.0) == 0
    finally:
        sandbox.kill()

    assert not root.exists()


def test_local_sandbox_backend_creates_virtualenv(tmp_path):
    backend = LocalSandboxBackend(root_dir=str(tmp_path), create_virtualenv=True)
    sandbox = backend.create()

    try:
        result = sandbox.exec(
            'python -c "import os,sys; print(sys.prefix); print(os.environ.get(\'VIRTUAL_ENV\', \'\'))"'
        )
        assert result.exit_code == 0
        lines = result.stdout.strip().splitlines()
        assert len(lines) == 2
        assert lines[0] == f"{sandbox.sandbox_home}/venv"
        assert lines[1] == f"{sandbox.sandbox_home}/venv"
    finally:
        sandbox.kill()


def test_local_sandbox_backend_creates_missing_root_dir(tmp_path):
    root_dir = tmp_path / "nested" / "sandboxes"
    backend = LocalSandboxBackend(root_dir=str(root_dir), create_virtualenv=False)
    sandbox = backend.create()

    try:
        assert root_dir.exists()
        assert Path(sandbox.sandbox_home).parent.parent == root_dir
    finally:
        sandbox.kill()


def test_local_sandbox_backend_inherits_host_cache_envs(tmp_path, monkeypatch):
    monkeypatch.setenv("PIP_CACHE_DIR", "/shared/pip-cache")
    monkeypatch.setenv("XDG_CACHE_HOME", "/shared/xdg-cache")

    backend = LocalSandboxBackend(root_dir=str(tmp_path), create_virtualenv=False)
    sandbox = backend.create()

    try:
        result = sandbox.exec(
            'printf "%s|%s" "$PIP_CACHE_DIR" "$XDG_CACHE_HOME"',
            cwd="/testbed",
        )
        assert result.exit_code == 0
        assert result.stdout == "/shared/pip-cache|/shared/xdg-cache"
    finally:
        sandbox.kill()
