# SPDX-License-Identifier: BSD-3-Clause
"""Backend selection, refusal before launch, OpenShell probing and rendering, local runs."""

import json
from unittest.mock import AsyncMock

import pytest
from openenv.core.openenvd import (
    EnforcementBackend,
    EnforcementUnavailable,
    get_backend,
    Guarantee,
    IsolationError,
    OpenEnvDConfig,
    register_backend,
)
from openenv.core.openenvd.backends import (
    local as local_mod,
    openshell as openshell_mod,
)
from openenv.core.openenvd.backends.local import LocalBackend
from openenv.core.openenvd.backends.openshell import (
    BASELINE_READ_ONLY,
    OpenShellBackend,
    render_policy,
)

OPENSHELL = {"image": "example:latest", "gateway": "local"}


def _config(**block) -> OpenEnvDConfig:
    block.setdefault("openshell", OPENSHELL)
    return OpenEnvDConfig.model_validate({"enabled": True, **block})


# --- selection and refusal -------------------------------------------------


def test_registry_builds_the_selected_backend(monkeypatch):
    from openenv.core.openenvd import backends

    assert isinstance(get_backend(_config()), OpenShellBackend)
    assert isinstance(
        get_backend(_config(enforcement={"backend": "local"})), LocalBackend
    )
    monkeypatch.setattr(backends, "_REGISTRY", dict(backends._REGISTRY))
    register_backend("custom", LocalBackend)
    assert isinstance(
        get_backend(_config(enforcement={"backend": "custom"})), LocalBackend
    )
    with pytest.raises(EnforcementUnavailable, match="no such backend"):
        get_backend(_config(enforcement={"backend": "missing"}))


async def test_local_backend_refuses_every_guarantee():
    backend = LocalBackend(_config(enforcement={"backend": "local"}))
    for guarantee in Guarantee:
        with pytest.raises(EnforcementUnavailable) as exc:
            await backend.ensure([guarantee])
        assert exc.value.missing == {guarantee}
    await backend.ensure([])


async def test_missing_guarantee_refuses_before_probing():
    class Partial(EnforcementBackend):
        name = "partial"
        guarantees = frozenset({Guarantee.PRIVILEGE_DROP})
        probe = AsyncMock()
        sandbox = python = None

    backend = Partial(_config())
    with pytest.raises(EnforcementUnavailable) as exc:
        await backend.ensure([Guarantee.PRIVILEGE_DROP, Guarantee.EGRESS_CONTROL])
    assert exc.value.missing == {Guarantee.EGRESS_CONTROL}
    backend.probe.assert_not_awaited()


def test_openshell_provides_every_guarantee():
    assert OpenShellBackend.guarantees == frozenset(Guarantee)


# --- OpenShell probe -------------------------------------------------------


@pytest.fixture
def cli(monkeypatch):
    """Fake `openshell` on PATH; records commands, answers `--version` and inventory."""
    state = {
        "version": "openshell 0.1.2",
        "inventory": '{"sandboxes": [], "next_page_token": ""}',
        "calls": [],
    }
    monkeypatch.setattr(openshell_mod.shutil, "which", lambda b: "/usr/bin/" + b)

    async def run(self, argv, **kwargs):
        state["calls"].append(argv)
        if argv[-1] == "--version":
            return state["version"]
        if "list" in argv:
            if state["inventory"] is None:
                raise IsolationError("OpenShell command failed")
            return state["inventory"]
        return ""

    monkeypatch.setattr(openshell_mod.OpenShellSandbox, "_run", run)
    return state


async def test_probe_accepts_supported_cli_and_reachable_gateway(cli):
    await OpenShellBackend(_config()).probe()
    inventory = cli["calls"][-1]
    assert inventory[:3] == ["/usr/bin/openshell", "--gateway", "local"]
    assert inventory[inventory.index("sandbox") + 1] == "list"


@pytest.mark.parametrize(
    "version", ["0.0.116", "0.1.1", "0.2.0", "0.1.2.dev1", "0.1.2rc1", "garbage"]
)
async def test_probe_rejects_unsupported_versions(cli, version):
    cli["version"] = "openshell " + version
    with pytest.raises(EnforcementUnavailable, match="OpenShell"):
        await OpenShellBackend(_config()).probe()
    assert all("list" not in call for call in cli["calls"])


@pytest.mark.parametrize("inventory", [None, "not json", '{"unexpected": true}'])
async def test_probe_rejects_unregistered_or_unreachable_gateway(cli, inventory):
    # `openshell status` exits 0 for an unregistered gateway, so the probe must
    # not rely on it: an inventory query fails or returns no sandbox list.
    cli["inventory"] = inventory
    with pytest.raises(EnforcementUnavailable, match="not registered or not reachable"):
        await OpenShellBackend(_config()).probe()


async def test_probe_rejects_missing_binaries(monkeypatch):
    monkeypatch.setattr(openshell_mod.shutil, "which", lambda b: None)
    with pytest.raises(EnforcementUnavailable, match="must be installed"):
        await OpenShellBackend(_config()).probe()


# --- OpenShell policy rendering and admission ------------------------------


def test_render_defaults_deny_egress_and_drop_privileges():
    policy = render_policy(_config())
    assert policy["landlock"] == {"compatibility": "hard_requirement"}
    assert policy["process"] == {"run_as_user": "1000", "run_as_group": "1000"}
    assert policy["filesystem_policy"]["include_workdir"] is False
    assert "/sandbox" in policy["filesystem_policy"]["read_write"]
    assert policy["network_policies"] == {}


def test_render_scopes_egress_rules_to_declared_binaries():
    config = _config(
        egress={
            "mode": "allowlist",
            "allow": [
                {"host": "api.anthropic.com", "binaries": ["/usr/local/bin/node"]},
                {"host": "pypi.org"},
            ],
        }
    )
    rules = render_policy(config)["network_policies"]
    assert rules["egress_0"]["endpoints"] == [
        {"host": "api.anthropic.com", "port": 443, "protocol": "tcp"}
    ]
    assert rules["egress_0"]["binaries"] == [{"path": "/usr/local/bin/node"}]
    assert rules["egress_1"]["binaries"] == [{"path": "/**"}]


def test_native_policy_is_used_verbatim():
    native = render_policy(_config())
    native["filesystem_policy"]["read_only"] = ["/usr"]
    config = _config(openshell={**OPENSHELL, "policy": native})
    assert render_policy(config) == native


def _sandbox_for(config):
    return OpenShellBackend(config).sandbox(timeout_s=30)


def test_admission_requires_the_exact_policy_without_network_rules():
    sandbox = _sandbox_for(_config())
    effective = json.loads(json.dumps(sandbox.policy))
    assert sandbox._admitted(effective)
    effective["filesystem_policy"]["read_only"].append("/app")
    assert not sandbox._admitted(effective)


def test_admission_tolerates_only_the_documented_baseline_with_network_rules():
    sandbox = _sandbox_for(
        _config(egress={"mode": "allowlist", "allow": [{"host": "pypi.org"}]})
    )
    effective = json.loads(json.dumps(sandbox.policy))
    effective["filesystem_policy"]["read_only"] += list(BASELINE_READ_ONLY)
    assert sandbox._admitted(effective)
    effective["filesystem_policy"]["read_write"].append("/var")
    assert not sandbox._admitted(effective)


# --- local backend ---------------------------------------------------------


async def test_local_sandbox_runs_processes_on_a_private_seed_copy(tmp_path):
    backend = LocalBackend(_config(enforcement={"backend": "local"}))
    seed = tmp_path / "workspace"
    seed.mkdir()
    (seed / "input.txt").write_text("seed")
    private = tmp_path / "private"
    private.mkdir(mode=0o700)

    sandbox = backend.sandbox(timeout_s=30)
    await sandbox.start(seed, private)
    process = await sandbox.spawn(
        [
            backend.python,
            "-c",
            "import os, pathlib; pathlib.Path('out.txt').write_text("
            "pathlib.Path('input.txt').read_text() + os.environ['MARK'])",
        ],
        {**backend.workload_env(), "MARK": "!"},
    )
    assert await process.wait() == 0
    staging = tmp_path / "staging"
    staging.mkdir()
    await sandbox.download(staging)
    await sandbox.close()

    assert (staging / "out.txt").read_text() == "seed!"
    assert not (seed / "out.txt").exists()
    assert sandbox.id is None
    assert list(private.iterdir()) == []


async def test_local_sandbox_close_kills_running_processes(tmp_path, monkeypatch):
    backend = LocalBackend(_config(enforcement={"backend": "local"}))
    seed = tmp_path / "workspace"
    seed.mkdir()
    sandbox = backend.sandbox(timeout_s=30)
    await sandbox.start(seed, tmp_path)
    process = await sandbox.spawn(
        [backend.python, "-c", "import time; time.sleep(60)"], backend.workload_env()
    )
    await sandbox.close()
    assert process.returncode is not None
    assert local_mod.LocalSandbox  # module imported for monkeypatch symmetry
