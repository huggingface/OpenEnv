# SPDX-License-Identifier: BSD-3-Clause

import ctypes
import json
import os
import sys

import pytest
from openenv.core.openenvd import landlock
from openenv.core.openenvd.landlock import (
    abi_version,
    handled_fs_access,
    LandlockNetPortAttr,
    LandlockPathBeneathAttr,
    LandlockRulesetAttr,
    LandlockSpec,
    LandlockUnavailable,
    restrict_self,
    ruleset_attr_size,
)

IS_LINUX = sys.platform.startswith("linux")


def test_struct_layouts_match_kernel_uapi():
    assert ctypes.sizeof(LandlockRulesetAttr) == 24
    assert ctypes.sizeof(LandlockPathBeneathAttr) == 12
    assert ctypes.sizeof(LandlockNetPortAttr) == 16
    assert [ruleset_attr_size(a) for a in (1, 3, 4, 5, 6, 7)] == [8, 8, 16, 16, 24, 24]


def test_handled_access_grows_with_abi():
    assert handled_fs_access(0) == 0
    assert handled_fs_access(1) == (1 << 13) - 1
    assert handled_fs_access(2) & landlock.LANDLOCK_ACCESS_FS_REFER
    assert not handled_fs_access(2) & landlock.LANDLOCK_ACCESS_FS_TRUNCATE
    assert handled_fs_access(3) & landlock.LANDLOCK_ACCESS_FS_TRUNCATE
    assert not handled_fs_access(4) & landlock.LANDLOCK_ACCESS_FS_IOCTL_DEV
    assert handled_fs_access(5) & landlock.LANDLOCK_ACCESS_FS_IOCTL_DEV


def test_spec_json_round_trip():
    spec = LandlockSpec(
        read_only=["/usr"],
        read_write=["/tmp"],
        connect_tcp=[80],
        bind_tcp=None,
        scope=False,
        hard_requirement=True,
    )
    assert LandlockSpec.from_json(spec.to_json()) == spec
    assert LandlockSpec.from_json(json.dumps(spec.to_json())) == spec


@pytest.mark.skipif(IS_LINUX, reason="non-Linux behaviour")
def test_unavailable_off_linux():
    assert abi_version() == 0
    assert restrict_self(LandlockSpec(read_write=["/tmp"])) == 0
    with pytest.raises(LandlockUnavailable):
        restrict_self(LandlockSpec(hard_requirement=True))


@pytest.mark.skipif(
    not IS_LINUX or abi_version() == 0, reason="needs Linux with Landlock"
)
def test_restricted_child_cannot_write_or_read_outside(tmp_path):
    allowed = tmp_path / "ro"
    allowed.mkdir()
    (allowed / "f").write_text("ok")
    outside = tmp_path / "outside"
    outside.write_text("secret")

    pid = os.fork()
    if pid == 0:
        code = 0
        try:
            restrict_self(LandlockSpec(read_only=[str(allowed)]))
            if (allowed / "f").read_text() != "ok":
                code |= 1
            try:
                (allowed / "g").write_text("x")
                code |= 2
            except PermissionError:
                pass
            try:
                outside.read_text()
                code |= 4
            except PermissionError:
                pass
        except BaseException:
            code = 99
        os._exit(code)
    _, status = os.waitpid(pid, 0)
    assert os.waitstatus_to_exitcode(status) == 0
