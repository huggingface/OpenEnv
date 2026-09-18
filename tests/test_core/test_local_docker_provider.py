# SPDX-License-Identifier: BSD-3-Clause

import subprocess
from unittest.mock import patch

import pytest
from openenv.core.containers.runtime.providers import LocalDockerProvider


@pytest.mark.parametrize("stop_times_out", [False, True])
def test_stop_removes_container_after_grace_period(stop_times_out: bool) -> None:
    with patch("subprocess.run") as run:
        provider = LocalDockerProvider()
        provider._container_id = "test-container"
        run.reset_mock()
        stop_result = (
            subprocess.TimeoutExpired("docker stop", 10) if stop_times_out else None
        )
        run.side_effect = [stop_result, None]

        provider.stop_container()

        commands = [call.args[0] for call in run.call_args_list]
        remove = ["docker", "rm", "test-container"]
        if stop_times_out:
            remove.insert(2, "--force")
        assert commands == [["docker", "stop", "test-container"], remove]
