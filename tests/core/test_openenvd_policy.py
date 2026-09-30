# SPDX-License-Identifier: BSD-3-Clause
from pathlib import Path

import pytest
from openenv.core.openenvd.policy import (
    OpenEnvDConfig,
    OpenShellConfig,
    Principal,
    SurfacePolicy,
)
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


def test_openshell_default_policy_is_private_and_denies_network():
    config = OpenShellConfig(image="example/openenv:latest", gateway="local")
    assert config.workspace == "default"
    assert config.python == "/usr/local/bin/python3"
    assert config.policy["landlock"] == {"compatibility": "hard_requirement"}
    assert config.policy["process"] == {"run_as_user": "1000", "run_as_group": "1000"}
    assert config.policy["filesystem_policy"]["include_workdir"] is False
    assert config.policy["network_policies"] == {}
    config.policy["network_policies"]["test"] = {}
    assert (
        OpenShellConfig(image="example:latest", gateway="local").policy[
            "network_policies"
        ]
        == {}
    )


def test_openshell_preserves_native_network_policy():
    policy = OpenShellConfig(image="example:latest", gateway="local").policy
    policy["network_policies"] = {
        "api": {
            "name": "api",
            "endpoints": [{"host": "api.example.com", "port": 443}],
            "binaries": [{"path": "/usr/local/bin/python3"}],
        }
    }
    config = OpenShellConfig(image="example:latest", gateway="local", policy=policy)
    assert config.policy["network_policies"] == policy["network_policies"]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("image", ""),
        ("image", "--privileged"),
        ("image", "example\nimage"),
        ("gateway", ""),
        ("gateway", "--local"),
        ("gateway", "local\0"),
        ("workspace", "--default"),
        ("workspace", "space name"),
        ("python", "python3"),
        ("python", "/sandbox/../usr/bin/python3"),
        ("python", "/sandbox/python3"),
        ("python", "/tmp/python3"),
    ],
)
def test_openshell_rejects_unsafe_launch_parameters(field, value):
    with pytest.raises(ValidationError):
        OpenShellConfig.model_validate(
            {"image": "example:latest", "gateway": "local", field: value}
        )


@pytest.mark.parametrize(
    ("section", "field", "value"),
    [
        ("landlock", "compatibility", "best_effort"),
        ("process", "run_as_user", "root"),
        ("process", "run_as_user", "0"),
        ("process", "run_as_user", "0000"),
        ("process", "run_as_user", 1000),
        ("process", "run_as_group", "0"),
        ("process", "run_as_group", "sandbox"),
        ("filesystem_policy", "include_workdir", True),
        ("filesystem_policy", "include_workdir", 0),
        ("filesystem_policy", "read_write", ["/"]),
        ("filesystem_policy", "read_write", ["/usr"]),
        ("filesystem_policy", "read_write", ["/sandbox/../usr"]),
        ("filesystem_policy", "read_write", ["/dev"]),
        ("filesystem_policy", "read_write", ["/sandbox-other"]),
    ],
)
def test_openshell_rejects_weakened_process_and_filesystem_policy(
    section, field, value
):
    policy = OpenShellConfig(image="example:latest", gateway="local").policy
    policy[section][field] = value
    with pytest.raises(ValidationError):
        OpenShellConfig(image="example:latest", gateway="local", policy=policy)


def test_openshell_accepts_confined_writable_subdirectories():
    policy = OpenShellConfig(image="example:latest", gateway="local").policy
    policy["filesystem_policy"]["read_write"] = [
        "/sandbox/episode",
        "/tmp/cache",
        "/dev/null",
    ]
    OpenShellConfig(image="example:latest", gateway="local", policy=policy)


def test_legacy_network_policy_requires_migration():
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        OpenEnvDConfig.model_validate({"network": {"allow": []}})
