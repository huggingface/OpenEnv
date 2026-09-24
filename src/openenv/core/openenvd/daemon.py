# SPDX-License-Identifier: BSD-3-Clause

"""Launch the environment declared in openenv.yaml, with optional isolation."""

from __future__ import annotations

import argparse
import importlib
import os
import sys
from pathlib import Path
from typing import Optional

import uvicorn
import yaml


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        prog="openenvd",
        description="OpenEnv privileged environment sidecar daemon",
        epilog="Set OPENENVD_<PRINCIPAL>_TOKEN for each configured privileged surface.",
    )
    parser.add_argument("--host", default="127.0.0.1", help="bind address")
    parser.add_argument("--port", type=int, default=8100, help="bind port")
    parser.add_argument(
        "--manifest",
        type=Path,
        required=True,
        help="openenv.yaml for the isolated environment runtime",
    )
    parser.add_argument("--factory", help="Environment factory as module:attribute")
    parser.add_argument(
        "--action-class", default="openenv.core.env_server.mcp_types:CallToolAction"
    )
    parser.add_argument("--workspace", type=Path, help="Dedicated episode workspace")
    parser.add_argument(
        "--asset-root", type=Path, help="Daemon-owned 0700 asset source directory"
    )
    parser.add_argument("--uid", type=int, help="Reserved unprivileged workload UID")
    parser.add_argument("--gid", type=int, help="Reserved unprivileged workload GID")
    parser.add_argument("--timeout", type=float, default=300)
    parser.add_argument("--cgroup-root", type=Path, default=Path("/sys/fs/cgroup"))
    args = parser.parse_args(argv)
    try:
        from .policy import load_config, Principal
        from .runtime import Runtime
        from .surfaces import create_surface_app

        config = load_config(args.manifest)
        factory = args.factory
        action_class = args.action_class
        manifest_root = args.manifest.resolve().parent
        if not factory or not config.enabled:
            manifest = yaml.safe_load(args.manifest.read_text())
            module, attribute = manifest["app"].split(":", 1)
            sys.path.insert(0, str(manifest_root))
            env_app = getattr(importlib.import_module(module), attribute)
            if not config.enabled:
                uvicorn.run(env_app, host=args.host, port=args.port)
                return
            try:
                env_factory, action_type = env_app.state.openenv_spec
            except AttributeError as exc:
                raise ValueError("custom apps require an explicit --factory") from exc
            if "<locals>" in env_factory.__qualname__:
                raise ValueError(
                    "environment factory must be importable at module scope"
                )
            factory = f"{env_factory.__module__}:{env_factory.__qualname__}"
            action_class = f"{action_type.__module__}:{action_type.__qualname__}"
        if not all((args.workspace, args.asset_root, args.uid, args.gid)):
            parser.error(
                "--manifest requires --workspace, --asset-root, --uid, and --gid"
            )
        runtime = Runtime(
            config,
            factory,
            action_class,
            args.workspace,
            uid=args.uid,
            gid=args.gid,
            asset_root=args.asset_root,
            timeout_s=args.timeout,
            python_path=manifest_root,
            cgroup_root=args.cgroup_root,
        )
        tokens = {
            p: os.environ.get(f"OPENENVD_{p.value.upper()}_TOKEN", "")
            for p in Principal
            if p != Principal.AGENT
        }
        app = create_surface_app(runtime, tokens)

    except ValueError as e:
        parser.error(str(e))
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
