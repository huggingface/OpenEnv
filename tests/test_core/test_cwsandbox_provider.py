"""Unit tests for CWSandboxProvider."""

from __future__ import annotations

import builtins
from types import SimpleNamespace
from unittest.mock import patch

import pytest  # type: ignore[import-not-found]
from openenv.core.containers.runtime.cwsandbox_provider import (  # type: ignore[import-not-found]
    CWSandboxProvider,
)


class _AuthError(Exception):
    pass


class _SandboxNotRunningError(Exception):
    pass


class _FakeRef:
    def __init__(self, value=None):
        self.value = value

    def result(self):
        if isinstance(self.value, BaseException):
            raise self.value
        return self.value


class _FakeProcess:
    def __init__(self, stdout: str = ""):
        self._result = SimpleNamespace(stdout=stdout, stderr="", returncode=0)

    def result(self):
        return self._result


class _FakeNetworkOptions:
    def __init__(self, **kwargs):
        self.ingress_mode = kwargs.get("ingress_mode")
        self.exposed_ports = kwargs.get("exposed_ports")
        self.egress_mode = kwargs.get("egress_mode")


class _FakeSandbox:
    def __init__(self):
        self.sandbox_id = "sandbox-123"
        self.service_address = "sandbox.example.test:8000"
        self.exposed_ports = ((8000, "http"),)
        self.exec_calls: list[list[str]] = []
        self.wait_calls: list[float | None] = []
        self.stopped = False
        self.dead_process = False
        self.log = ""
        self.stop_error: BaseException | None = None

    def wait(self, timeout=None):
        self.wait_calls.append(timeout)
        return self

    def exec(self, command, timeout_seconds=None):
        self.exec_calls.append(command)
        script = command[-1]
        if "test -f /app/env/openenv.yaml" in script:
            return _FakeProcess("")
        if "find /app" in script:
            return _FakeProcess("/app/envs/coding_env/openenv.yaml\n")
        if script.startswith("cat /app/envs/coding_env/openenv.yaml"):
            return _FakeProcess("spec_version: 1\napp: coding_env.server.app:app\n")
        if "kill -0" in script:
            return _FakeProcess("DEAD\n" if self.dead_process else "RUNNING\n")
        if script.startswith("cat /tmp/openenv-server.log"):
            return _FakeProcess(self.log)
        return _FakeProcess("")

    def stop(self, missing_ok=False):
        self.stopped = True
        self.stop_missing_ok = missing_ok
        if self.stop_error is not None:
            raise self.stop_error
        return _FakeRef()


class _FakeSandboxClass:
    def __init__(self):
        self.last_run_kwargs = None
        self.last_list_kwargs = None
        self.deleted: list[dict] = []
        self.instance = _FakeSandbox()
        self.list_result = _FakeRef([])
        self.run_count = 0

    def run(self, **kwargs):
        self.run_count += 1
        self.last_run_kwargs = kwargs
        return self.instance

    def delete(self, sandbox_id, **kwargs):
        self.deleted.append({"sandbox_id": sandbox_id, **kwargs})
        return _FakeRef()

    def list(self, **kwargs):
        self.last_list_kwargs = kwargs
        return self.list_result


@pytest.fixture()
def fake_sdk():
    sandbox_cls = _FakeSandboxClass()
    return SimpleNamespace(
        CWSandboxAuthenticationError=_AuthError,
        SandboxNotRunningError=_SandboxNotRunningError,
        NetworkOptions=_FakeNetworkOptions,
        Sandbox=sandbox_cls,
        _sandbox_cls=sandbox_cls,
    )


def test_start_container_configures_sandbox(fake_sdk):
    provider = CWSandboxProvider(
        sdk=fake_sdk,
        ingress_mode="public",
        egress_mode="internet",
        request_timeout_seconds=123,
        max_lifetime_seconds=456,
        max_timeout_seconds=789,
        resources={"requests": {"cpu": "1"}},
        tags=["openenv"],
    )

    url = provider.start_container(
        "coding-env:latest",
        env_vars={"DEBUG": "1"},
    )

    kwargs = fake_sdk._sandbox_cls.last_run_kwargs
    assert url == "http://sandbox.example.test:8000"
    assert kwargs["container_image"] == "coding-env:latest"
    assert kwargs["environment_variables"] == {"DEBUG": "1"}
    assert kwargs["request_timeout_seconds"] == 123
    assert kwargs["max_lifetime_seconds"] == 456
    assert kwargs["max_timeout_seconds"] == 789
    assert kwargs["resources"] == {"requests": {"cpu": "1"}}
    assert kwargs["tags"] == ["openenv"]
    assert kwargs["network"].ingress_mode == "public"
    assert kwargs["network"].exposed_ports == (8000,)
    assert kwargs["network"].egress_mode == "internet"


def test_base_url_requires_active_sandbox(fake_sdk):
    provider = CWSandboxProvider(sdk=fake_sdk)

    with pytest.raises(RuntimeError, match="no active base_url"):
        _ = provider.base_url


def test_start_container_finds_openenv_yaml_with_generic_search(fake_sdk):
    provider = CWSandboxProvider(sdk=fake_sdk)

    provider.start_container("coding-env:latest")

    scripts = [" ".join(call) for call in fake_sdk._sandbox_cls.instance.exec_calls]
    assert any(
        "find /app -maxdepth 5 -name openenv.yaml" in script for script in scripts
    )
    assert any("cat /app/envs/coding_env/openenv.yaml" in script for script in scripts)
    assert any("nohup bash -c" in script for script in scripts)
    assert any("coding_env.server.app:app" in script for script in scripts)


def test_port_other_than_8000_raises(fake_sdk):
    provider = CWSandboxProvider(sdk=fake_sdk)

    with pytest.raises(ValueError, match="only supports port 8000"):
        provider.start_container("coding-env:latest", port=3000)


def test_service_address_uses_exposed_port_scheme(fake_sdk):
    fake_sdk._sandbox_cls.instance.service_address = (
        "e67b6f1841b3.c2d60ed8.builds.cwsandbox.com:443"
    )
    fake_sdk._sandbox_cls.instance.exposed_ports = ((8000, "http"),)
    provider = CWSandboxProvider(sdk=fake_sdk)

    url = provider.start_container("coding-env:latest")

    assert url == "http://e67b6f1841b3.c2d60ed8.builds.cwsandbox.com:443"


def test_service_address_uses_https_exposed_port_scheme(fake_sdk):
    fake_sdk._sandbox_cls.instance.service_address = (
        "e67b6f1841b3.c2d60ed8.builds.cwsandbox.com:443"
    )
    fake_sdk._sandbox_cls.instance.exposed_ports = ((8000, "https"),)
    provider = CWSandboxProvider(sdk=fake_sdk)

    url = provider.start_container("coding-env:latest")

    assert url == "https://e67b6f1841b3.c2d60ed8.builds.cwsandbox.com:443"


def test_explicit_cmd_skips_yaml_discovery(fake_sdk):
    provider = CWSandboxProvider(sdk=fake_sdk, cmd="python -m server")

    provider.start_container("coding-env:latest")

    scripts = [" ".join(call) for call in fake_sdk._sandbox_cls.instance.exec_calls]
    assert not any(
        "cat /app/envs/coding_env/openenv.yaml" in script for script in scripts
    )
    assert any("python -m server" in script for script in scripts)


def test_start_container_rejects_second_active_sandbox(fake_sdk):
    provider = CWSandboxProvider(sdk=fake_sdk)
    provider.start_container("coding-env:latest")

    with pytest.raises(RuntimeError, match="already has an active sandbox"):
        provider.start_container("other-env:latest")

    assert fake_sdk._sandbox_cls.run_count == 1


def test_preflight_lists_sandboxes(fake_sdk):
    CWSandboxProvider.preflight(sdk=fake_sdk)

    assert fake_sdk._sandbox_cls.last_list_kwargs == {}


def test_preflight_rejects_invalid_credentials(fake_sdk):
    fake_sdk._sandbox_cls.list_result = _FakeRef(_AuthError("bad token"))

    with pytest.raises(SystemExit, match="auth check failed.*CWSANDBOX_API_KEY"):
        CWSandboxProvider.preflight(sdk=fake_sdk)


def test_wait_for_ready_polls_health(fake_sdk):
    provider = CWSandboxProvider(sdk=fake_sdk)
    provider.start_container("coding-env:latest")
    response = SimpleNamespace(status_code=200)

    with (
        patch("openenv.core.containers.runtime.cwsandbox_provider.time.sleep"),
        patch("requests.get", return_value=response) as get,
    ):
        provider.wait_for_ready("http://sandbox.example.test:8000")

    get.assert_called_with("http://sandbox.example.test:8000/health", timeout=5.0)


def test_wait_for_ready_suppresses_server_log_by_default(fake_sdk):
    import requests

    provider = CWSandboxProvider(sdk=fake_sdk)
    provider.start_container("coding-env:latest", env_vars={"TOKEN": "secret-123"})
    fake_sdk._sandbox_cls.instance.dead_process = True
    fake_sdk._sandbox_cls.instance.log = "Traceback TOKEN=secret-123 failed"

    with (
        patch("openenv.core.containers.runtime.cwsandbox_provider.time.sleep"),
        patch("requests.get", side_effect=requests.ConnectionError("not ready")),
    ):
        with pytest.raises(RuntimeError, match="Server output is not surfaced") as exc:
            provider.wait_for_ready("http://sandbox.example.test:8000")

    message = str(exc.value)
    assert "secret-123" not in message
    assert "Traceback" not in message


def test_wait_for_ready_redacts_server_log_when_opted_in(fake_sdk):
    import requests

    provider = CWSandboxProvider(sdk=fake_sdk, surface_server_logs=True)
    provider.start_container("coding-env:latest", env_vars={"TOKEN": "secret-123"})
    fake_sdk._sandbox_cls.instance.dead_process = True
    fake_sdk._sandbox_cls.instance.log = "Traceback TOKEN=secret-123 failed"

    with (
        patch("openenv.core.containers.runtime.cwsandbox_provider.time.sleep"),
        patch("requests.get", side_effect=requests.ConnectionError("not ready")),
    ):
        with pytest.raises(RuntimeError, match="Log \\(redacted\\)") as exc:
            provider.wait_for_ready("http://sandbox.example.test:8000")

    message = str(exc.value)
    assert "secret-123" not in message
    assert "TOKEN=***" in message
    assert "Traceback" in message


def test_stop_container_is_idempotent_and_deletes(fake_sdk):
    provider = CWSandboxProvider(sdk=fake_sdk)
    provider.start_container("coding-env:latest")

    provider.stop_container()
    provider.stop_container()

    sandbox = fake_sdk._sandbox_cls.instance
    assert sandbox.stopped is True
    assert sandbox.stop_missing_ok is True
    assert fake_sdk._sandbox_cls.deleted == [
        {
            "sandbox_id": "sandbox-123",
            "base_url": None,
            "timeout_seconds": 300.0,
            "missing_ok": True,
        }
    ]


def test_close_stops_and_deletes_active_sandbox(fake_sdk):
    provider = CWSandboxProvider(sdk=fake_sdk)
    provider.start_container("coding-env:latest")

    provider.close()

    sandbox = fake_sdk._sandbox_cls.instance
    assert sandbox.stopped is True
    assert fake_sdk._sandbox_cls.deleted == [
        {
            "sandbox_id": "sandbox-123",
            "base_url": None,
            "timeout_seconds": 300.0,
            "missing_ok": True,
        }
    ]


def test_stop_container_deletes_when_sandbox_is_already_stopped(fake_sdk):
    provider = CWSandboxProvider(sdk=fake_sdk)
    provider.start_container("coding-env:latest")
    fake_sdk._sandbox_cls.instance.stop_error = _SandboxNotRunningError("stopped")

    provider.stop_container()

    assert fake_sdk._sandbox_cls.deleted == [
        {
            "sandbox_id": "sandbox-123",
            "base_url": None,
            "timeout_seconds": 300.0,
            "missing_ok": True,
        }
    ]


def test_stop_container_clears_redact_values(fake_sdk):
    provider = CWSandboxProvider(sdk=fake_sdk)
    provider.start_container("coding-env:latest", env_vars={"TOKEN": "secret-123"})
    assert provider._redact_values == {"secret-123"}

    provider.stop_container()

    assert provider._redact_values == set()


def test_missing_cwsandbox_dependency_raises_runtime_error():
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "cwsandbox":
            raise ImportError("missing")
        return real_import(name, *args, **kwargs)

    with patch("builtins.__import__", side_effect=fake_import):
        with pytest.raises(RuntimeError, match="requires cwsandbox"):
            CWSandboxProvider()
