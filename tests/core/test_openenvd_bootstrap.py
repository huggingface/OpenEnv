# SPDX-License-Identifier: BSD-3-Clause
"""The worker bootstrap seals the worker process before any openenv import."""

import ctypes
import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest


def load_bootstrap():
    path = (
        Path(__file__).resolve().parents[2]
        / "src/openenv/core/openenvd/_worker_bootstrap.py"
    )
    spec = importlib.util.spec_from_file_location("_worker_bootstrap", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_bootstrap_seal_process_is_linux_only_noop_elsewhere():
    module = load_bootstrap()
    module.seal_process()
    if sys.platform == "linux":
        assert ctypes.CDLL(None).prctl(3, 0, 0, 0, 0) == 0


def test_bootstrap_main_seals_then_delegates(monkeypatch):
    module = load_bootstrap()
    calls = []
    monkeypatch.setattr(module, "seal_process", lambda: calls.append("seal"))
    monkeypatch.setattr(module.site, "main", lambda: calls.append("site"))
    monkeypatch.setattr(
        module.runpy,
        "run_module",
        lambda name, run_name: calls.append((name, run_name)),
    )
    monkeypatch.setattr(sys, "argv", ["_worker_bootstrap.py", "7", "factory", "action"])
    module.main()
    assert calls == ["seal", "site", ("openenv.core.openenvd.worker", "__main__")]
    assert sys.argv[0] == "openenv.core.openenvd.worker"


@pytest.mark.skipif(sys.platform != "linux", reason="Linux dumpable attribute")
def test_bootstrap_script_seals_before_running_worker(tmp_path):
    # A stub worker module reports the dumpable attribute it observes.
    stub_dir = tmp_path / "stubs"
    package = stub_dir / "openenv/core/openenvd"
    package.mkdir(parents=True)
    (stub_dir / "openenv/__init__.py").write_text("")
    (stub_dir / "openenv/core/__init__.py").write_text("")
    (stub_dir / "openenv/core/openenvd/__init__.py").write_text("")
    (package / "worker.py").write_text(
        "import ctypes; print(ctypes.CDLL(None).prctl(3, 0, 0, 0, 0))"
    )
    bootstrap = (
        Path(__file__).resolve().parents[2]
        / "src/openenv/core/openenvd/_worker_bootstrap.py"
    )
    result = subprocess.run(
        [sys.executable, "-S", str(bootstrap), "7", "factory", "action"],
        capture_output=True,
        env={"PYTHONPATH": str(stub_dir), "PATH": "/usr/bin:/bin"},
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "0"
