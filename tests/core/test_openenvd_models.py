# SPDX-License-Identifier: BSD-3-Clause

"""Tests for openenvd task models."""

import pytest
from openenv.core.openenvd.models import TaskSpec
from pydantic import ValidationError


class TestTaskSpec:
    def test_minimal_spec(self):
        spec = TaskSpec(name="observer", argv=["sleep", "30"])
        assert spec.name == "observer"
        assert spec.argv == ["sleep", "30"]
        assert spec.cwd is None
        assert spec.uid is None
        assert spec.gid is None
        assert spec.network_isolated is False

    def test_empty_argv_rejected(self):
        with pytest.raises(ValidationError):
            TaskSpec(name="x", argv=[])

    def test_invalid_name_rejected(self):
        for bad in ["", "has space", "-leading", "UPPER", "a/b"]:
            with pytest.raises(ValidationError):
                TaskSpec(name=bad, argv=["true"])

    @pytest.mark.parametrize(
        "settings",
        [
            {"uid": 1000},
            {"gid": 1000},
            {"uid": 0, "gid": 1000},
            {"uid": 1000, "gid": 0},
            {"uid": True, "gid": 1000},
            {"uid": 2**32 - 1, "gid": 1000},
            {"network_isolation": True},
            {"restart_policy": "always"},
            {"auto_uid": True},
            {"cwd": "relative/path"},
            {"cwd": "/invalid\0path"},
        ],
    )
    def test_unsafe_or_mistyped_settings_rejected(self, settings):
        with pytest.raises(ValidationError):
            TaskSpec(name="task", argv=["true"], **settings)

    @pytest.mark.parametrize("argv", [[""], ["true", "bad\0argument"]])
    def test_invalid_process_arguments_rejected(self, argv):
        with pytest.raises(ValidationError):
            TaskSpec(name="task", argv=argv)
