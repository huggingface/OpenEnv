# SPDX-License-Identifier: BSD-3-Clause

"""Unit assembly without a kernel: plans, translation, ordering and the trace audit.

The live behaviour (real containers, both tiers) is covered by
`integration/test_live_unit.py`, which runs in Docker.
"""

import json

import pytest
from openenv.core.openenvd.check import evaluate, strengths_from, ZoneCheck
from openenv.core.openenvd.contract import (
    Guarantee,
    parse_manifest,
    Phase,
    Strength,
    Tier,
    ZoneKind,
)
from openenv.core.openenvd.probes import EnforcementReport
from openenv.core.openenvd.unit import (
    HARNESS_PORTS,
    PLACEHOLDER_KEY,
    SOCKET_DIR,
    Unit,
    UnitPaths,
)

MANIFEST = {
    "app": "echo_env.server.app:app",
    "privileged_assets": {"oracle": "assets/oracle.json"},
    "zones": {
        "agent": {"containers": {"env": {}, "harness": {"argv": ["harness"]}}},
        "services": {
            "containers": {
                "sim": {
                    "argv": ["sim", "--state", "/var/lib/sim"],
                    "expose": {"host": "sim.example", "port": 80},
                    "port": 8080,
                    "state": "/var/lib/sim",
                }
            }
        },
        "observers": {
            "containers": {
                "grader": {
                    "argv": ["grade"],
                    "reads": ["assets.oracle", "runner.output", "trace"],
                    "phases": ["grading"],
                    "after": "runner",
                },
                "runner": {
                    "argv": ["run"],
                    "reads": ["workspace"],
                    "phases": ["grading"],
                    "output": "/out/report.json",
                },
            }
        },
    },
}


class FakeLauncher:
    def __init__(self, namespaced):
        self.namespaced = namespaced

    async def start(self, plan, bundle_dir):
        pass

    async def stop(self, name):
        pass

    async def wait(self, name, timeout):
        return True


def _unit(tmp_path, namespaced, monkeypatch):
    report = EnforcementReport(Tier.CONTAINERS if namespaced else Tier.LANDLOCK, {}, [])
    paths = UnitPaths(
        state=tmp_path / "s" / "state", run=tmp_path / "s" / "run", assets=tmp_path
    )
    (tmp_path / "assets").mkdir()
    (tmp_path / "assets" / "oracle.json").write_text("{}")
    unit = Unit(
        parse_manifest(MANIFEST),
        report,
        paths,
        FakeLauncher(namespaced),
        cgroups=None,
        python="/usr/bin/python3",
    )
    monkeypatch.setattr(unit, "_rootfs", lambda rootfs: rootfs)
    unit.episode_id = "ep1"
    unit._prepare_dirs()
    return unit


def _shim(unit, key):
    _, spec = unit._plan(unit._containers[key])
    return spec


def test_harness_sees_only_loopback_relays_and_a_placeholder_key(tmp_path, monkeypatch):
    unit = _unit(tmp_path, True, monkeypatch)
    spec = _shim(unit, "agent-harness")
    assert spec["env"]["ANTHROPIC_API_KEY"] == PLACEHOLDER_KEY
    assert spec["env"]["MCP_URL"].startswith("http://127.0.0.1:")
    ports = sorted(HARNESS_PORTS.values())
    assert sorted(f["port"] for f in spec["forwards"]) == ports
    # Strict means "only the relay ports", not an AF_INET ban, so the harness
    # can still reach its forwarders.
    assert spec["landlock"]["connect_tcp"] == ports
    assert spec["landlock"]["bind_tcp"] == []
    assert spec["principal_filter"] == {}


def test_env_reaches_services_through_its_own_socket_dir(tmp_path, monkeypatch):
    unit = _unit(tmp_path, True, monkeypatch)
    spec = _shim(unit, "agent-env")
    assert spec["forwards"] == [
        {
            "kind": "tcp_to_unix",
            "host": "127.0.0.10",
            "port": 80,
            "path": f"{SOCKET_DIR}/svc-sim.sock",
        }
    ]
    assert spec["env"]["OPENENVD_SERVICE_SIM"] == "http://sim.example:80"
    hosts = (unit._sock_dir(unit._containers["agent-env"]) / "hosts").read_text()
    assert "127.0.0.10 sim.example" in hosts
    assert "--uds" in spec["argv"] and f"{SOCKET_DIR}/env.sock" in spec["argv"]


def test_services_and_observers_cannot_connect_anywhere(tmp_path, monkeypatch):
    unit = _unit(tmp_path, True, monkeypatch)
    sim = _shim(unit, "services-sim")
    assert sim["landlock"]["connect_tcp"] == []
    assert sim["landlock"]["bind_tcp"] == [8080]
    grader = _shim(unit, "observers-grader")
    assert grader["landlock"]["connect_tcp"] == []


def test_observer_mounts_are_read_only_and_nosymfollow(tmp_path, monkeypatch):
    unit = _unit(tmp_path, True, monkeypatch)
    plan, _ = unit._plan(unit._containers["observers-grader"])
    by_dest = {m.destination: m.options for m in plan.mounts}
    for dest in ("/assets", "/inputs"):
        assert {"ro", "nosymfollow", "nodev", "nosuid", "noexec"} <= set(by_dest[dest])
    assert "ro" not in by_dest["/out"]
    assert "/workspace" not in by_dest


def test_landlock_tier_translates_container_paths(tmp_path, monkeypatch):
    unit = _unit(tmp_path, False, monkeypatch)
    sock = unit._sock_dir(unit._containers["services-sim"])
    spec = _shim(unit, "services-sim")
    state = str(unit._episode_dir() / "services" / "sim" / "merged")
    assert spec["argv"] == ["sim", "--state", state]
    assert spec["forwards"][0]["path"] == str(sock / "svc.sock")
    assert state in spec["landlock"]["read_write"]
    env_spec = _shim(unit, "agent-env")
    # Without a network namespace, a privileged port becomes a high one.
    assert env_spec["forwards"][0]["port"] >= 20_000
    assert env_spec["env"]["OPENENVD_SERVICE_SIM"].startswith("http://127.0.0.10:")


def test_runner_runs_before_the_verdict(tmp_path, monkeypatch):
    unit = _unit(tmp_path, True, monkeypatch)
    observers = unit._observers_for(Phase.GRADING)
    assert [c.name for c in unit._order(observers)] == ["runner", "grader"]


def test_audit_flags_dropped_thinking(tmp_path, monkeypatch):
    from openenv.core.openenvd.trace import TraceRecorder

    unit = _unit(tmp_path, True, monkeypatch)
    unit._trace = TraceRecorder(tmp_path / "trace.jsonl", b"k", append_only=False)
    unit._trace.append(
        "model.response",
        "model_proxy",
        {"request_id": "r1", "text": "hi", "thinking": "the plan"},
    )
    unit._trace.append(
        "harness.self_report", "harness", {"request_id": "r1", "text": "hi"}
    )
    assert unit._audit_trace() == 1
    lines = [json.loads(line) for line in unit._trace.path.read_text().splitlines()]
    assert lines[-1]["kind"] == "trace.mismatch"
    assert lines[-1]["data"]["mismatch"] == "thinking_dropped"


@pytest.mark.parametrize(
    "results, strict, failure",
    [
        ({"read_hidden_0": "allowed"}, False, "read_hidden_0"),
        ({"connect_1.1.1.1_53": "allowed"}, False, "connect_1.1.1.1_53"),
        ({"inet_socket": "allowed"}, True, "inet_socket"),
        ({"fork_past_pids_max": "allowed"}, False, "fork_past_pids_max"),
        ({"visible_pids": 40}, False, "visible_pids"),
        ({"uid": 0}, False, "uid"),
    ],
)
def test_check_evaluation(results, strict, failure):
    failures = evaluate(ZoneKind.AGENT, results, strict=strict)
    assert [f.split(":")[0] for f in failures] == [failure]


def test_check_strengths_downgrade_only_what_failed():
    checks = [ZoneCheck(ZoneKind.AGENT, {}, ["visible_pids: x"])]
    strengths = strengths_from(checks)
    assert strengths[Guarantee.PRINCIPAL_ISOLATION] is Strength.NOT_SUPPORTED
    assert strengths[Guarantee.ASSET_ISOLATION] is Strength.PREVENTED
    assert evaluate(ZoneKind.AGENT, {"inet_socket": "allowed"}, strict=False) == []
