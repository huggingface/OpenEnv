# SPDX-License-Identifier: BSD-3-Clause
"""The bootstrap protects the process and descriptors before site or app imports."""

import ctypes
import importlib.util
import json
import subprocess
import sys
import venv
from pathlib import Path
from types import SimpleNamespace

BOOTSTRAP = (
    Path(__file__).resolve().parents[3]
    / "src/openenv/core/openenvd/_worker_bootstrap.py"
)


def load_bootstrap():
    spec = importlib.util.spec_from_file_location("_worker_bootstrap", BOOTSTRAP)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_bootstrap_seal_process_is_linux_only_noop_elsewhere():
    module = load_bootstrap()
    module.seal_process()
    if sys.platform == "linux":
        assert ctypes.CDLL(None).prctl(3, 0, 0, 0, 0) == 0


def test_bootstrap_main_seals_and_protects_stdio_before_site(monkeypatch):
    module = load_bootstrap()
    calls = []
    monkeypatch.setattr(module, "seal_process", lambda: calls.append("seal"))

    def protect():
        calls.append("protect stdio")
        return 7, 8

    monkeypatch.setattr(module, "protect_stdio", protect)
    monkeypatch.setattr(module.site, "main", lambda: calls.append("site"))

    def import_worker(name):
        calls.append(("import", name))
        return SimpleNamespace(main=lambda *fds: calls.append(("worker", fds)))

    monkeypatch.setattr(module.importlib, "import_module", import_worker)
    monkeypatch.setattr(sys, "argv", ["-c"])
    module.main()
    assert calls == [
        "seal",
        "protect stdio",
        "site",
        ("import", "openenv.core.openenvd.worker"),
        ("worker", (7, 8)),
    ]
    assert sys.argv == ["openenv.core.openenvd.worker"]


def test_bootstrap_source_protects_site_hooks_and_worker_imports(tmp_path):
    # Exercise the exact isolated-interpreter command used over SSH. A venv
    # supplies a site hook and stub worker without using ignored PYTHONPATH.
    python_root = tmp_path / "python"
    venv.EnvBuilder(symlinks=True).create(python_root)
    site_packages = (
        python_root
        / "lib"
        / f"python{sys.version_info.major}.{sys.version_info.minor}"
        / "site-packages"
    )
    stub_dir = tmp_path / "stubs"
    package = stub_dir / "openenv/core/openenvd"
    package.mkdir(parents=True)
    for directory in (stub_dir / "openenv", package.parent, package):
        (directory / "__init__.py").write_text("")
    (stub_dir / "bootstrap_probe.py").write_text(
        "import ctypes, os, sys\n"
        "os.environ['SITE_DUMPABLE'] = str(ctypes.CDLL(None).prctl(3, 0, 0, 0, 0)) "
        "if sys.platform == 'linux' else 'unavailable'\n"
        "print('site hook diagnostics', flush=True)\n"
    )
    (site_packages / "bootstrap-test.pth").write_text(
        f"{stub_dir}\nimport bootstrap_probe\n"
    )
    (package / "worker.py").write_text(
        """
import ctypes
import json
import os
import sys

print('worker import diagnostics', flush=True)

def main(control_read, control_write):
    result = {
        'site_dumpable': os.environ['SITE_DUMPABLE'],
        'worker_dumpable': ctypes.CDLL(None).prctl(3, 0, 0, 0, 0)
            if sys.platform == 'linux' else None,
        'inheritable': [os.get_inheritable(control_read), os.get_inheritable(control_write)],
        'ordinary_stdin': os.read(0, 1).decode(),
        'control_stdin': os.read(control_read, 32).decode(),
    }
    os.write(control_write, json.dumps(result).encode() + b'\\n')
"""
    )
    result = subprocess.run(
        [str(python_root / "bin/python"), "-I", "-S", "-c", BOOTSTRAP.read_text()],
        input="private control input",
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    response = json.loads(result.stdout)
    assert response["inheritable"] == [False, False]
    assert response["ordinary_stdin"] == ""
    assert response["control_stdin"] == "private control input"
    assert "site hook diagnostics" in result.stderr
    assert "worker import diagnostics" in result.stderr
    if sys.platform == "linux":
        assert response["site_dumpable"] == "0"
        assert response["worker_dumpable"] == 0
