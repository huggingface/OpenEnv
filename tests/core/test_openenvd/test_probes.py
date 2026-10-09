# SPDX-License-Identifier: BSD-3-Clause

"""Tier assessment, enforcement checks and probe behavior."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest
from openenv.core.openenvd import probes as P
from openenv.core.openenvd.contract import EnforcementSpec, Guarantee, Strength, Tier
from openenv.core.openenvd.probes import (
    assess,
    EnforcementUnavailable,
    ensure,
    landlock_abi,
    ProbeResult,
)

G, S = Guarantee, Strength
LINUX = sys.platform.startswith("linux")


def results(
    userns: bool = True,
    cgroup: bool = True,
    oci: bool = True,
    abi: int = 5,
    seccomp: bool = True,
) -> list[ProbeResult]:
    return [
        ProbeResult("userns", userns, "" if userns else "unshare: EPERM"),
        ProbeResult("cgroup_writable", cgroup, "" if cgroup else "mkdir: EROFS"),
        ProbeResult("cgroup_kill", True),
        ProbeResult("landlock", abi >= 1, f"abi={abi}" if abi else "ENOSYS"),
        ProbeResult("seccomp", seccomp, "" if seccomp else "filter mode: EINVAL"),
        ProbeResult("oci_crun", oci, "/usr/bin/crun" if oci else ""),
        ProbeResult("oci_runc", False),
        ProbeResult("oci_runsc", False),
    ]


def test_containers_tier_prevents_everything():
    report = assess(results())
    assert report.tier is Tier.CONTAINERS
    assert report.strengths == {g: S.PREVENTED for g in Guarantee}


@pytest.mark.parametrize("kw", [{"userns": False}, {"cgroup": False}, {"oci": False}])
def test_missing_container_prerequisite_falls_to_landlock(kw):
    assert assess(results(**kw)).tier is Tier.LANDLOCK


def test_landlock_tier_abi4_with_seccomp():
    report = assess(results(userns=False, abi=4))
    assert report.tier is Tier.LANDLOCK
    assert report.strengths == {
        G.ASSET_ISOLATION: S.PREVENTED,
        G.CONTROL_PLANE_ISOLATION: S.PREVENTED,
        G.EGRESS_CONTROL: S.PREVENTED,
        G.PRIVILEGE_DROP: S.DETECTED_AND_REAPED,
        G.PRINCIPAL_ISOLATION: S.NOT_SUPPORTED,
        G.SERVICE_ISOLATION: S.PREVENTED,
        G.OBSERVER_ISOLATION: S.PREVENTED,
        G.RESOURCE_ISOLATION: S.DETECTED_AND_REAPED,
        G.TRACE_INTEGRITY: S.DETECTED_AND_REAPED,
    }


def test_landlock_tier_old_abi_without_seccomp():
    report = assess(results(userns=False, abi=3, seccomp=False))
    assert report.tier is Tier.LANDLOCK
    assert report.strengths[G.CONTROL_PLANE_ISOLATION] is S.NOT_SUPPORTED
    assert report.strengths[G.SERVICE_ISOLATION] is S.NOT_SUPPORTED
    assert report.strengths[G.EGRESS_CONTROL] is S.NOT_SUPPORTED
    assert report.strengths[G.ASSET_ISOLATION] is S.PREVENTED


def test_none_tier():
    report = assess(results(userns=False, abi=0))
    assert report.tier is Tier.NONE
    assert set(report.strengths.values()) == {S.NOT_SUPPORTED}
    assert len(report.strengths) == len(Guarantee)


def test_assess_with_no_probes_is_none():
    assert assess([]).tier is Tier.NONE


def test_landlock_abi():
    assert landlock_abi(results(abi=6)) == 6
    assert landlock_abi(results(abi=0)) == 0
    assert landlock_abi([ProbeResult("landlock", True, "weird")]) == 0
    assert landlock_abi([]) == 0


def test_to_info_has_only_tier_and_strengths():
    report = assess(results(userns=False, abi=3))
    info = report.to_info()
    assert set(info) == {"tier", "guarantees"}
    assert info["tier"] == "landlock"
    assert info["guarantees"]["principal_isolation"] == "not_supported"
    blob = json.dumps(info)
    for leak in ("/usr/bin/crun", "EPERM", "abi=", "crun", "userns"):
        assert leak not in blob


def test_ensure_passes_when_met():
    spec = EnforcementSpec(require=["asset_isolation"], tiers=[Tier.CONTAINERS])
    ensure(spec, assess(results()))
    ensure(EnforcementSpec(), assess(results(userns=False, abi=0)))


def test_ensure_list_form_means_prevented():
    spec = EnforcementSpec.model_validate({"require": ["principal_isolation"]})
    with pytest.raises(EnforcementUnavailable) as exc:
        ensure(spec, assess(results(userns=False)))
    msg = str(exc.value)
    assert "principal_isolation: prevented required, not_supported obtained" in msg
    assert "user namespaces: unshare: EPERM" in msg


def test_ensure_dict_form_accepts_weaker_minimum():
    spec = EnforcementSpec.model_validate(
        {"require": {"resource_isolation": "detected_and_reaped"}}
    )
    ensure(spec, assess(results(userns=False)))
    with pytest.raises(EnforcementUnavailable, match="resource_isolation"):
        ensure(spec, assess(results(userns=False, abi=0)))


def test_ensure_lists_every_gap_with_its_reason():
    spec = EnforcementSpec.model_validate(
        {"require": ["control_plane_isolation", "egress_control", "asset_isolation"]}
    )
    report = assess(results(userns=False, abi=3, seccomp=False))
    with pytest.raises(EnforcementUnavailable) as exc:
        ensure(spec, report)
    msg = str(exc.value)
    assert "control_plane_isolation: prevented required, not_supported" in msg
    assert "landlock abi=3 < 4" in msg
    assert "egress_control: prevented required, not_supported" in msg
    assert "seccomp: filter mode: EINVAL" in msg
    assert "asset_isolation" not in msg  # met in the landlock tier


def test_ensure_tier_restriction():
    spec = EnforcementSpec(tiers=[Tier.CONTAINERS])
    with pytest.raises(EnforcementUnavailable) as exc:
        ensure(spec, assess(results(oci=False)))
    msg = str(exc.value)
    assert "tier: one of [containers] required, landlock obtained" in msg
    assert "no OCI runtime" in msg
    ensure(EnforcementSpec(tiers=[Tier.LANDLOCK, Tier.CONTAINERS]), assess(results()))


def test_ensure_none_tier_explains_landlock():
    spec = EnforcementSpec(require={G.ASSET_ISOLATION: S.DETECTED_AND_REAPED})
    with pytest.raises(EnforcementUnavailable, match="landlock: ENOSYS"):
        ensure(spec, assess(results(userns=False, cgroup=False, abi=0)))


@pytest.mark.skipif(LINUX, reason="non-Linux behavior")
@pytest.mark.parametrize(
    "probe",
    [
        P.probe_userns,
        P.probe_landlock,
        P.probe_seccomp,
        lambda: P.probe_overlay(Path("/tmp")),
        lambda: P.probe_nosymfollow(Path("/tmp")),
        lambda: P.probe_append_only(Path("/tmp")),
    ],
)
def test_syscall_probes_report_not_linux(probe):
    result = probe()
    assert result.ok is False
    assert result.detail == "not linux"


def test_probe_cgroup_writable_on_fake(tmp_path: Path):
    (tmp_path / "cgroup.subtree_control").write_text("cpu memory")
    result = P.probe_cgroup_writable(tmp_path)
    assert result == ProbeResult("cgroup_writable", True)
    assert not (tmp_path / ".openenvd-probe").exists()


def test_probe_cgroup_writable_empty_subtree_control(tmp_path: Path):
    (tmp_path / "cgroup.subtree_control").write_text("")
    assert P.probe_cgroup_writable(tmp_path).ok is True


def test_probe_cgroup_writable_without_cgroupfs(tmp_path: Path):
    result = P.probe_cgroup_writable(tmp_path)
    assert result.ok is False and result.detail


@pytest.mark.skipif(os.geteuid() == 0, reason="root ignores permissions")
def test_probe_cgroup_writable_read_only(tmp_path: Path):
    (tmp_path / "cgroup.subtree_control").write_text("")
    tmp_path.chmod(0o500)
    try:
        result = P.probe_cgroup_writable(tmp_path)
    finally:
        tmp_path.chmod(0o700)
    assert result.ok is False
    assert result.detail.startswith("mkdir")


def test_probe_cgroup_kill(tmp_path: Path):
    assert P.probe_cgroup_kill(tmp_path).ok is False
    (tmp_path / "cgroup.kill").write_text("")
    assert P.probe_cgroup_kill(tmp_path).ok is True
    assert P.probe_cgroup_kill(tmp_path / "missing").ok is False


def test_probe_oci_runtimes(monkeypatch):
    found = {"runc": "/usr/sbin/runc"}
    monkeypatch.setattr(P.shutil, "which", lambda name: found.get(name))
    out = P.probe_oci_runtimes()
    assert set(out) == {"crun", "runc", "runsc"}
    assert out["runc"] == ProbeResult("oci_runc", True, "/usr/sbin/runc")
    assert out["crun"].ok is False


def test_probe_all_never_raises(tmp_path: Path):
    out = P.probe_all(tmp_path / "cg", tmp_path / "scratch")
    names = [r.name for r in out]
    assert names[:8] == [
        "userns",
        "cgroup_writable",
        "cgroup_kill",
        "landlock",
        "seccomp",
        "overlay",
        "nosymfollow",
        "append_only",
    ]
    assert {"oci_crun", "oci_runc", "oci_runsc"} <= set(names)
    assess(out)  # whatever the host, assess accepts it


@pytest.mark.skipif(not LINUX, reason="Linux only")
def test_real_probes_are_well_formed(tmp_path: Path):
    for probe in (
        P.probe_userns(),
        P.probe_landlock(),
        P.probe_seccomp(),
        P.probe_overlay(tmp_path),
        P.probe_nosymfollow(tmp_path),
        P.probe_append_only(tmp_path),
    ):
        assert isinstance(probe.ok, bool)
        assert probe.ok or probe.detail
    assert list(tmp_path.iterdir()) == []  # probes clean up their scratch


@pytest.mark.skipif(not LINUX, reason="Linux only")
def test_real_landlock_detail():
    result = P.probe_landlock()
    if result.ok:
        assert landlock_abi([result]) >= 1


@pytest.mark.skipif(not LINUX, reason="Linux only")
def test_real_seccomp_matches_proc_status():
    status = Path("/proc/self/status").read_text()
    if "Seccomp_filters:" in status:
        assert P.probe_seccomp().ok is True
