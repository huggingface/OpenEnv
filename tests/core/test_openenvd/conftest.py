# SPDX-License-Identifier: BSD-3-Clause
"""Shared fixtures for openenvd tests."""

import sys
import sysconfig
import venv
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture(scope="session")
def worker_python(tmp_path_factory):
    # -I ignores PYTHONPATH. A disposable venv adds the source packages through
    # the same site initialization used by an installed sandbox image.
    root = tmp_path_factory.mktemp("worker-python")
    venv.EnvBuilder(system_site_packages=True, symlinks=True).create(root)
    site_packages = (
        root
        / "lib"
        / f"python{sys.version_info.major}.{sys.version_info.minor}"
        / "site-packages"
    )
    modules = root / "test_modules"
    modules.mkdir()
    (site_packages / "openenv-worker-test.pth").write_text(
        f"{ROOT / 'src'}\n{ROOT / 'envs'}\n{modules}\n{sysconfig.get_path('purelib')}\n"
    )
    return root / "bin/python", modules
