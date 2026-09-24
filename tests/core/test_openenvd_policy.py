# SPDX-License-Identifier: BSD-3-Clause
from pathlib import Path

import pytest
from openenv.core.openenvd.policy import OpenEnvDConfig, Principal, SurfacePolicy
from pydantic import ValidationError


@pytest.mark.parametrize(
    "pattern", ["reset", "*", "gr*", "grader.*", "grader.read_file", "r*", "[r]eset"]
)
def test_agent_rejects_privileged_patterns(pattern):
    with pytest.raises(ValidationError):
        SurfacePolicy(principal="agent", tools=[pattern])


def test_declarative_policies_bind_principals():
    config = OpenEnvDConfig.model_validate(
        {
            "enabled": True,
            "surfaces": {
                "agent": {"tools": ["env.*"]},
                "grader": {
                    "tools": ["env.*", "grader.*"],
                    "allow_privileged_exec": True,
                },
            },
        }
    )
    assert config.surfaces[Principal.AGENT].permits_tool("env.echo")
    assert not config.surfaces[Principal.AGENT].permits_tool("grader.read_file")
    assert not OpenEnvDConfig().enabled


@pytest.mark.parametrize(
    "policy",
    [
        {"principal": "grader"},
        {"allow_lifecycle": True},
        {"allow_privileged_exec": True},
        {"fs_read": ["/workspace/**"]},
        {"fs_read": ["/**"]},
        {"fs_read": ["/openenvd/assets/**"]},
        {"fs_read": ["/workspace/../openenvd/**"]},
    ],
)
def test_invalid_agent_declarations(policy):
    with pytest.raises(ValidationError):
        OpenEnvDConfig.model_validate({"surfaces": {"agent": policy}})


def test_observer_is_read_only():
    with pytest.raises(ValidationError):
        SurfacePolicy(principal="observer", tools=["env.*"])
    assert SurfacePolicy(principal="observer", stream=["process"]).stream == (
        "process",
    )


def test_paths_are_allowlisted():
    policy = SurfacePolicy(principal="grader", fs_read=["/workspace/**"])
    assert policy.permits_read(Path("/workspace/file"))
    assert not policy.permits_read(Path("/etc/passwd"))
