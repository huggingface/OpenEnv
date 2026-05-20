# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for the Docker sandbox backend.

Tests marked ``@pytest.mark.docker`` require a running Docker daemon and
are skipped in CI when Docker is unavailable. They exercise the real
``docker run`` / ``docker exec`` / ``docker rm`` lifecycle.
"""

from __future__ import annotations

import subprocess
import time

import pytest

_DOCKER_AVAILABLE = False
try:
    subprocess.run(
        ["docker", "version"],
        capture_output=True,
        check=True,
        timeout=5,
    )
    _DOCKER_AVAILABLE = True
except Exception:
    pass

docker = pytest.mark.skipif(not _DOCKER_AVAILABLE, reason="Docker not available")


class TestDockerSandboxBackendUnit:
    """Unit tests that don't require Docker."""

    def test_import(self):
        from openenv.core.harness.sandbox.docker_backend import (
            DockerBgJob,
            DockerSandboxBackend,
            DockerSandboxHandle,
        )

        assert DockerSandboxBackend is not None
        assert DockerSandboxHandle is not None
        assert DockerBgJob is not None

    def test_exported_from_package(self):
        from openenv.core.harness.sandbox import (
            DockerBgJob,
            DockerSandboxBackend,
            DockerSandboxHandle,
        )

        assert DockerSandboxBackend is not None
        assert DockerSandboxHandle is not None
        assert DockerBgJob is not None

    def test_create_sandbox_backend_factory(self):
        from openenv.core.harness.sandbox import create_sandbox_backend

        assert callable(create_sandbox_backend)

    def test_create_sandbox_backend_unknown_raises(self):
        from openenv.core.harness.sandbox import create_sandbox_backend

        with pytest.raises(ValueError, match="Unknown sandbox backend"):
            create_sandbox_backend("bogus")  # type: ignore[arg-type]

    def test_create_adds_host_gateway_and_supports_image_override(self, monkeypatch):
        import openenv.core.harness.sandbox.docker_backend as docker_backend

        calls: list[list[str]] = []

        def _fake_run(cmd, *args, **kwargs):
            calls.append(list(cmd))
            if cmd[:2] == ["docker", "version"]:
                return subprocess.CompletedProcess(cmd, 0, "", "")
            if cmd[:2] == ["docker", "run"]:
                return subprocess.CompletedProcess(
                    cmd,
                    0,
                    "1234567890abcdef\n",
                    "",
                )
            return subprocess.CompletedProcess(cmd, 0, "", "")

        monkeypatch.setattr(docker_backend.subprocess, "run", _fake_run)

        backend = docker_backend.DockerSandboxBackend(image="base:latest")
        handle = backend.create(image="override:latest")
        assert handle.sandbox_id == "1234567890ab"

        run_cmds = [cmd for cmd in calls if cmd[:2] == ["docker", "run"]]
        assert len(run_cmds) == 1
        run_cmd = run_cmds[0]
        assert "--add-host" in run_cmd
        assert "host.docker.internal:host-gateway" in run_cmd
        assert "override:latest" in run_cmd

    @pytest.mark.skipif(_DOCKER_AVAILABLE, reason="Only test error when Docker missing")
    def test_backend_raises_without_docker(self):
        from openenv.core.harness.sandbox.docker_backend import DockerSandboxBackend

        with pytest.raises(RuntimeError, match="Docker daemon"):
            DockerSandboxBackend()


@docker
class TestDockerSandboxBackendIntegration:
    """Integration tests against a real Docker daemon."""

    def test_create_and_kill(self):
        from openenv.core.harness.sandbox.docker_backend import DockerSandboxBackend

        backend = DockerSandboxBackend(image="ubuntu:22.04")
        sandbox = backend.create(timeout_s=60)
        try:
            assert sandbox.sandbox_id
            assert len(sandbox.sandbox_id) == 12
        finally:
            sandbox.kill()

    def test_exec_echo(self):
        from openenv.core.harness.sandbox.docker_backend import DockerSandboxBackend

        backend = DockerSandboxBackend(image="ubuntu:22.04")
        sandbox = backend.create(timeout_s=60)
        try:
            result = sandbox.exec("echo hello world")
            assert result.exit_code == 0
            assert "hello world" in result.stdout
        finally:
            sandbox.kill()

    def test_exec_nonzero_exit(self):
        from openenv.core.harness.sandbox.docker_backend import DockerSandboxBackend

        backend = DockerSandboxBackend(image="ubuntu:22.04")
        sandbox = backend.create(timeout_s=60)
        try:
            result = sandbox.exec("exit 42")
            assert result.exit_code == 42
        finally:
            sandbox.kill()

    def test_exec_with_env(self):
        from openenv.core.harness.sandbox.docker_backend import DockerSandboxBackend

        backend = DockerSandboxBackend(image="ubuntu:22.04")
        sandbox = backend.create(timeout_s=60)
        try:
            result = sandbox.exec("echo $MY_VAR", envs={"MY_VAR": "test123"})
            assert result.exit_code == 0
            assert "test123" in result.stdout
        finally:
            sandbox.kill()

    def test_exec_with_cwd(self):
        from openenv.core.harness.sandbox.docker_backend import DockerSandboxBackend

        backend = DockerSandboxBackend(image="ubuntu:22.04")
        sandbox = backend.create(timeout_s=60)
        try:
            result = sandbox.exec("pwd", cwd="/tmp")
            assert result.exit_code == 0
            assert "/tmp" in result.stdout
        finally:
            sandbox.kill()

    def test_write_and_read_text(self):
        from openenv.core.harness.sandbox.docker_backend import DockerSandboxBackend

        backend = DockerSandboxBackend(image="ubuntu:22.04")
        sandbox = backend.create(timeout_s=60)
        try:
            sandbox.write_text("/tmp/test.txt", "hello from test")
            content = sandbox.read_text("/tmp/test.txt")
            assert content == "hello from test"
        finally:
            sandbox.kill()

    def test_write_creates_parent_dirs(self):
        from openenv.core.harness.sandbox.docker_backend import DockerSandboxBackend

        backend = DockerSandboxBackend(image="ubuntu:22.04")
        sandbox = backend.create(timeout_s=60)
        try:
            sandbox.write_text("/home/user/deep/nested/file.txt", "nested content")
            content = sandbox.read_text("/home/user/deep/nested/file.txt")
            assert content == "nested content"
        finally:
            sandbox.kill()

    def test_write_special_chars(self):
        from openenv.core.harness.sandbox.docker_backend import DockerSandboxBackend

        backend = DockerSandboxBackend(image="ubuntu:22.04")
        sandbox = backend.create(timeout_s=60)
        try:
            text = "line1\nline2\n'quotes' and \"doubles\" and $vars"
            sandbox.write_text("/tmp/special.txt", text)
            content = sandbox.read_text("/tmp/special.txt")
            assert content == text
        finally:
            sandbox.kill()

    def test_read_missing_file_raises(self):
        from openenv.core.harness.sandbox.docker_backend import DockerSandboxBackend

        backend = DockerSandboxBackend(image="ubuntu:22.04")
        sandbox = backend.create(timeout_s=60)
        try:
            with pytest.raises(FileNotFoundError):
                sandbox.read_text("/nonexistent/path.txt")
        finally:
            sandbox.kill()

    def test_exists(self):
        from openenv.core.harness.sandbox.docker_backend import DockerSandboxBackend

        backend = DockerSandboxBackend(image="ubuntu:22.04")
        sandbox = backend.create(timeout_s=60)
        try:
            assert not sandbox.exists("/tmp/check_me.txt")
            sandbox.write_text("/tmp/check_me.txt", "exists")
            assert sandbox.exists("/tmp/check_me.txt")
        finally:
            sandbox.kill()

    def test_start_bg_and_wait(self):
        from openenv.core.harness.sandbox.docker_backend import DockerSandboxBackend

        backend = DockerSandboxBackend(image="ubuntu:22.04")
        sandbox = backend.create(timeout_s=60)
        try:
            job = sandbox.start_bg("sleep 1 && echo done > /tmp/bg_out.txt")
            exit_code = job.wait(timeout=10)
            assert exit_code == 0
            content = sandbox.read_text("/tmp/bg_out.txt")
            assert "done" in content
        finally:
            sandbox.kill()

    def test_start_bg_kill(self):
        from openenv.core.harness.sandbox.docker_backend import DockerSandboxBackend

        backend = DockerSandboxBackend(image="ubuntu:22.04")
        sandbox = backend.create(timeout_s=60)
        try:
            job = sandbox.start_bg("sleep 300")
            time.sleep(0.5)
            job.kill()
            # Should be able to wait without hanging
            exit_code = job.wait(timeout=5)
            # Exit code after kill is implementation-defined
            assert isinstance(exit_code, int)
        finally:
            sandbox.kill()

    def test_start_bg_timeout(self):
        from openenv.core.harness.sandbox.docker_backend import DockerSandboxBackend

        backend = DockerSandboxBackend(image="ubuntu:22.04")
        sandbox = backend.create(timeout_s=60)
        try:
            job = sandbox.start_bg("sleep 300")
            with pytest.raises(TimeoutError):
                job.wait(timeout=1)
            job.kill()
        finally:
            sandbox.kill()

    def test_create_with_envs(self):
        from openenv.core.harness.sandbox.docker_backend import DockerSandboxBackend

        backend = DockerSandboxBackend(image="ubuntu:22.04")
        sandbox = backend.create(timeout_s=60, envs={"INIT_VAR": "from_create"})
        try:
            result = sandbox.exec("echo $INIT_VAR")
            assert "from_create" in result.stdout
        finally:
            sandbox.kill()

    def test_create_with_metadata(self):
        from openenv.core.harness.sandbox.docker_backend import DockerSandboxBackend

        backend = DockerSandboxBackend(image="ubuntu:22.04")
        sandbox = backend.create(
            timeout_s=60,
            metadata={"episode_id": "ep-123"},
        )
        try:
            result = subprocess.run(
                [
                    "docker",
                    "inspect",
                    "--format",
                    '{{index .Config.Labels "openenv.episode_id"}}',
                    sandbox._container_id,
                ],
                capture_output=True,
                text=True,
            )
            assert "ep-123" in result.stdout
        finally:
            sandbox.kill()

    def test_factory_creates_docker_backend(self):
        from openenv.core.harness.sandbox import create_sandbox_backend

        backend = create_sandbox_backend("docker", image="ubuntu:22.04")
        sandbox = backend.create(timeout_s=60)
        try:
            result = sandbox.exec("echo ok")
            assert result.exit_code == 0
        finally:
            sandbox.kill()

    def test_satisfies_sandbox_handle_protocol(self):
        from openenv.core.harness.sandbox import SandboxHandle
        from openenv.core.harness.sandbox.docker_backend import DockerSandboxBackend

        backend = DockerSandboxBackend(image="ubuntu:22.04")
        sandbox = backend.create(timeout_s=60)
        try:
            assert isinstance(sandbox, SandboxHandle)
            assert hasattr(sandbox, "sandbox_id")
            assert hasattr(sandbox, "exec")
            assert hasattr(sandbox, "start_bg")
            assert hasattr(sandbox, "write_text")
            assert hasattr(sandbox, "read_text")
            assert hasattr(sandbox, "exists")
            assert hasattr(sandbox, "kill")
        finally:
            sandbox.kill()

    def test_satisfies_sandbox_backend_protocol(self):
        from openenv.core.harness.sandbox import SandboxBackend
        from openenv.core.harness.sandbox.docker_backend import DockerSandboxBackend

        assert issubclass(DockerSandboxBackend, SandboxBackend)

    def test_satisfies_bg_job_protocol(self):
        from openenv.core.harness.sandbox.docker_backend import DockerSandboxBackend

        backend = DockerSandboxBackend(image="ubuntu:22.04")
        sandbox = backend.create(timeout_s=60)
        try:
            job = sandbox.start_bg("sleep 1")
            assert hasattr(job, "pid")
            assert hasattr(job, "wait")
            assert hasattr(job, "kill")
            job.kill()
        finally:
            sandbox.kill()

    def test_kill_is_idempotent(self):
        from openenv.core.harness.sandbox.docker_backend import DockerSandboxBackend

        backend = DockerSandboxBackend(image="ubuntu:22.04")
        sandbox = backend.create(timeout_s=60)
        sandbox.kill()
        sandbox.kill()  # should not raise
