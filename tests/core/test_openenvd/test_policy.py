# SPDX-License-Identifier: BSD-3-Clause
"""The openenvd contract: surfaces, assets, workload paths, egress, guarantees."""

from pathlib import Path

import pytest
from openenv.core.openenvd.backends.openshell import render_policy
from openenv.core.openenvd.policy import (
    Guarantee,
    load_config,
    OpenEnvDConfig,
    OpenShellConfig,
    Principal,
    SurfacePolicy,
)
from pydantic import ValidationError

OPENSHELL = {"image": "example:latest", "gateway": "local"}


def _config(**block) -> OpenEnvDConfig:
    return OpenEnvDConfig.model_validate({"enabled": True, **block})


def _native_policy() -> dict:
    return render_policy(_config(openshell=OPENSHELL))


@pytest.mark.parametrize(
    "pattern", ["reset", "*", "gr*", "grader.*", "grader.read_file", "r*", "[r]eset"]
)
def test_agent_rejects_privileged_patterns(pattern):
    with pytest.raises(ValidationError):
        SurfacePolicy(principal="agent", tools=[pattern])


def test_declarative_policies_bind_principals():
    config = _config(
        openshell=OPENSHELL,
        surfaces={
            "agent": {"tools": ["env.*"]},
            "grader": {"tools": ["env.*", "grader.*"], "allow_privileged_exec": True},
        },
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


def test_defaults_select_openshell_with_no_requirements():
    config = OpenEnvDConfig()
    assert config.enforcement.backend == "openshell"
    assert config.enforcement.require == ()
    assert config.egress.mode == "none"
    assert "/sandbox" in config.workload.read_write


def test_enabled_openshell_requires_its_section():
    with pytest.raises(ValidationError, match="requires an openshell section"):
        _config()
    _config(enforcement={"backend": "local"})


def test_required_guarantees_parse():
    config = _config(
        openshell=OPENSHELL,
        enforcement={"require": ["asset_isolation", "control_plane_isolation"]},
    )
    assert config.enforcement.require == (
        Guarantee.ASSET_ISOLATION,
        Guarantee.CONTROL_PLANE_ISOLATION,
    )


@pytest.mark.parametrize("path", ["/abs/solution", "../solution", ""])
def test_assets_are_relative_to_the_asset_root(path):
    with pytest.raises(ValidationError, match="asset sources"):
        _config(openshell=OPENSHELL, privileged_assets={"solution": path})


@pytest.mark.parametrize(
    "read_write", [["/"], ["/usr"], ["/sandbox/../usr"], ["/dev"], ["/sandbox-other"]]
)
def test_workload_writes_are_confined(read_write):
    with pytest.raises(ValidationError, match="paths"):
        _config(openshell=OPENSHELL, workload={"read_write": read_write})


@pytest.mark.parametrize("python", ["/sandbox/python3", "/tmp/python3"])
def test_worker_interpreter_must_not_be_writable(python):
    with pytest.raises(ValidationError, match="must not be writable"):
        _config(openshell={**OPENSHELL, "python": python})


def test_egress_allowlist_requires_rules_and_none_forbids_them():
    with pytest.raises(ValidationError, match="requires at least one rule"):
        _config(openshell=OPENSHELL, egress={"mode": "allowlist"})
    with pytest.raises(ValidationError, match="only valid when"):
        _config(openshell=OPENSHELL, egress={"allow": [{"host": "pypi.org"}]})


def test_native_policy_excludes_neutral_declarations():
    with pytest.raises(ValidationError, match="not both"):
        _config(
            openshell={**OPENSHELL, "policy": _native_policy()},
            egress={"mode": "allowlist", "allow": [{"host": "pypi.org"}]},
        )


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
        ("run_as_user", "0"),
        ("run_as_group", "sandbox"),
    ],
)
def test_openshell_rejects_unsafe_launch_parameters(field, value):
    with pytest.raises(ValidationError):
        OpenShellConfig.model_validate({**OPENSHELL, field: value})


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
def test_native_policy_rejects_weakened_isolation(section, field, value):
    policy = _native_policy()
    policy[section][field] = value
    with pytest.raises(ValidationError):
        OpenShellConfig(**OPENSHELL, policy=policy)


def test_native_policy_accepts_confined_writable_subdirectories():
    policy = _native_policy()
    policy["filesystem_policy"]["read_write"] = [
        "/sandbox/episode",
        "/tmp/cache",
        "/dev/null",
    ]
    OpenShellConfig(**OPENSHELL, policy=policy)


def test_load_config_reads_openenv_yaml(tmp_path):
    manifest = tmp_path / "openenv.yaml"
    manifest.write_text(
        "name: demo\n"
        "openenvd:\n"
        "  enabled: true\n"
        "  enforcement: {backend: local, require: []}\n"
    )
    assert load_config(manifest).enforcement.backend == "local"
    (tmp_path / "plain.yaml").write_text("name: demo\n")
    assert not load_config(tmp_path / "plain.yaml").enabled
