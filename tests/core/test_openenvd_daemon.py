# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the manifest-driven openenvd entrypoint."""

from unittest.mock import Mock

import pytest
from openenv.core.openenvd import daemon


def test_manifest_is_required_before_server_start(monkeypatch, capsys):
    # A legacy operator token must not enable a separate task-management service.
    monkeypatch.setenv("OPENENVD_TOKEN", "legacy-admin-token")
    run = Mock()
    monkeypatch.setattr(daemon.uvicorn, "run", run)
    with pytest.raises(SystemExit) as error:
        daemon.main([])
    assert error.value.code == 2
    assert "--manifest" in capsys.readouterr().err
    run.assert_not_called()


def test_manifest_disabled_runs_original_app(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from unittest.mock import Mock

    from openenv.core.openenvd import daemon

    manifest = tmp_path / "openenv.yaml"
    manifest.write_text("app: example.app:app\nopenenvd:\n  enabled: false\n")
    original_app = object()
    monkeypatch.setattr(
        daemon.importlib,
        "import_module",
        lambda name: SimpleNamespace(app=original_app),
    )
    run = Mock()
    monkeypatch.setattr(daemon.uvicorn, "run", run)
    daemon.main(["--manifest", str(manifest)])
    run.assert_called_once_with(original_app, host="127.0.0.1", port=8100)


def test_manifest_discovers_factory_without_environment_code_changes(
    tmp_path, monkeypatch
):
    from types import SimpleNamespace
    from unittest.mock import Mock

    from openenv.core.openenvd import daemon, runtime, surfaces

    manifest = tmp_path / "openenv.yaml"
    manifest.write_text("app: example.app:app\nopenenvd:\n  enabled: true\n")
    env_factory = SimpleNamespace(__module__="example.env", __qualname__="Environment")
    action_type = SimpleNamespace(__module__="example.models", __qualname__="Action")
    original_app = SimpleNamespace(
        state=SimpleNamespace(openenv_spec=(env_factory, action_type))
    )
    monkeypatch.setattr(
        daemon.importlib,
        "import_module",
        lambda name: SimpleNamespace(app=original_app),
    )
    runtime_factory = Mock()
    monkeypatch.setattr(runtime, "Runtime", runtime_factory)
    monkeypatch.setattr(surfaces, "create_surface_app", Mock())
    monkeypatch.setattr(daemon.uvicorn, "run", Mock())
    daemon.main(
        [
            "--manifest",
            str(manifest),
            "--workspace",
            str(tmp_path),
            "--asset-root",
            str(tmp_path),
            "--uid",
            "12345",
            "--gid",
            "12345",
        ]
    )
    assert runtime_factory.call_args.args[1:3] == (
        "example.env:Environment",
        "example.models:Action",
    )
