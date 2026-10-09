# SPDX-License-Identifier: BSD-3-Clause

"""openenvd as PID 1 of an environment unit.

Startup: load the manifest, probe what the unit's runtime allows, refuse if the
manifest needs more, adopt the cgroup root, then serve the one external listener.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import secrets
import signal
import sys
from pathlib import Path

from .cgroups import CgroupTree
from .contract import load_manifest, Tier
from .probes import assess, EnforcementUnavailable, ensure, probe_all
from .proctree import ProcessLauncher, ProcessTree
from .relays import stop_server
from .runtime import find_runtime, OciRuntime
from .surfaces import serve_surfaces, surfaces_app
from .unit import ModelRoute, OciLauncher, Unit, UnitPaths

logger = logging.getLogger("openenvd")


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="openenvd", description=__doc__)
    p.add_argument("--manifest", type=Path, default=Path("openenv.yaml"))
    p.add_argument("--assets", type=Path, help="asset root (default: manifest dir)")
    p.add_argument("--seed", type=Path, help="read-only seed for /workspace")
    p.add_argument("--state", type=Path, default=Path("/var/lib/openenvd"))
    p.add_argument("--cgroup-root", type=Path, default=Path("/sys/fs/cgroup"))
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8100)
    p.add_argument("--python", default=sys.executable)
    p.add_argument(
        "--tokens",
        type=Path,
        help="JSON file with orchestrator and observer tokens; read once, then deleted",
    )
    p.add_argument("--model-url", help="upstream for harness model calls")
    p.add_argument(
        "--model-key-file", type=Path, help="file holding the real model key; deleted"
    )
    p.add_argument("--model-provider", default="anthropic")
    p.add_argument(
        "--trace-key-file",
        type=Path,
        help="HMAC key for trace seals; read once, deleted",
    )
    p.add_argument("--check", action="store_true", help="run the enforcement check")
    p.add_argument("--probe", action="store_true", help="print the tier and exit")
    return p


def _read_once(path: Path) -> str:
    text = path.read_text()
    try:
        path.unlink()
    except OSError:
        pass
    return text


def _tokens(path: Path | None) -> tuple[str, str]:
    if path is None:
        orchestrator = os.environ.pop("OPENENVD_ORCHESTRATOR_TOKEN", "")
        observer = os.environ.pop("OPENENVD_OBSERVER_TOKEN", "")
    else:
        data = json.loads(_read_once(path))
        orchestrator, observer = data.get("orchestrator", ""), data.get("observer", "")
    if not orchestrator:
        orchestrator = secrets.token_urlsafe(32)
        logger.warning("no orchestrator token given; generated one for this run")
        print(f"OPENENVD_ORCHESTRATOR_TOKEN={orchestrator}", flush=True)
    return orchestrator, observer or secrets.token_urlsafe(32)


def build_unit(args: argparse.Namespace) -> Unit:
    """Probe, enforce the manifest's requirements, and assemble the unit."""
    manifest = load_manifest(args.manifest)
    state = args.state.resolve()
    paths = UnitPaths(
        state=state / "state",
        run=state / "run",
        assets=(args.assets or args.manifest.parent).resolve(),
        seed=args.seed
        or (Path(manifest.workspace_seed) if manifest.workspace_seed else None),
    )
    for d in (paths.state, paths.run):
        d.mkdir(parents=True, exist_ok=True)
        d.chmod(0o711)
    scratch = paths.run / "probe"
    scratch.mkdir(exist_ok=True)
    report = assess(probe_all(args.cgroup_root, scratch))
    logger.info("tier %s", report.tier.value)
    ensure(manifest.enforcement, report)
    asset_sources = {
        str(
            (paths.assets / rel).parent
            if (paths.assets / rel).is_file()
            else paths.assets / rel
        )
        for rel in manifest.privileged_assets.values()
    }
    hidden = sorted({str(state), "/run/secrets", *asset_sources})
    if report.tier is Tier.CONTAINERS:
        cgroups = CgroupTree(args.cgroup_root)
        cgroups.adopt_self("control")
        preferred = manifest.zones.agent.runtime.oci
        binary = find_runtime(preferred)
        if binary is None or (
            manifest.zones.agent.runtime.require and Path(binary).name != preferred
        ):
            raise EnforcementUnavailable(f"OCI runtime {preferred} is not installed")
        runtime = OciRuntime(binary, paths.run / "oci", paths.state / "logs")
        launcher = OciLauncher(runtime, secrets.token_hex(4), hidden=hidden)
    else:
        writable = any(p.name == "cgroup_writable" and p.ok for p in report.probes)
        if writable:
            cgroups = CgroupTree(args.cgroup_root)
            cgroups.adopt_self("control")
        else:
            cgroups = ProcessTree()
        launcher = ProcessLauncher(cgroups, args.python, paths.state / "logs")
        launcher.hidden = hidden
        logger.warning(
            "tier %s: zones share the unit's namespaces and are separated by "
            "Landlock and seccomp",
            report.tier.value,
        )
    model = None
    if args.model_url:
        key = _read_once(args.model_key_file).strip() if args.model_key_file else ""
        model = ModelRoute(args.model_url, key, args.model_provider)
    key = (
        _read_once(args.trace_key_file).strip().encode()
        if args.trace_key_file
        else None
    )
    unit = Unit(
        manifest,
        report,
        paths,
        launcher,
        cgroups,
        model=model,
        python=args.python,
        trace_key=key,
    )
    return unit


async def serve(args: argparse.Namespace) -> int:
    try:
        unit = build_unit(args)
    except EnforcementUnavailable as exc:
        print(f"openenvd: refusing to start: {exc}", file=sys.stderr)
        return 3
    if args.check:
        from .check import run_check

        result = await run_check(unit)
        print(json.dumps(result.to_json(), indent=2))
        required = unit.manifest.enforcement.require
        short = [
            g.value
            for g, minimum in required.items()
            if g in result.strengths and not result.strengths[g].satisfies(minimum)
        ]
        await unit.shutdown()
        for failure in result.failures:
            print(f"openenvd check: {failure}", file=sys.stderr)
        return 1 if short else 0
    orchestrator, observer = _tokens(args.tokens)
    env_sock = str(unit._sock_dir(unit._containers["agent-env"]) / "env.sock")
    app = surfaces_app(
        unit, env_sock, orchestrator_token=orchestrator, observer_token=observer
    )
    server = await serve_surfaces(app, args.host, args.port)
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, stop.set)
    logger.info("openenvd listening on %s:%s", args.host, args.port)
    print("openenvd ready", flush=True)
    await stop.wait()
    await stop_server(server)
    await unit.shutdown()
    return 0


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=os.environ.get("OPENENVD_LOG", "INFO"),
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    args = _parser().parse_args(argv)
    if args.probe:
        state = args.state / "run" / "probe"
        state.mkdir(parents=True, exist_ok=True)
        report = assess(probe_all(args.cgroup_root, state))
        print(
            json.dumps(
                {**report.to_info(), "probes": [p.__dict__ for p in report.probes]},
                indent=2,
            )
        )
        return 0
    return asyncio.run(serve(args))
