# SPDX-License-Identifier: BSD-3-Clause

"""`openenv check`: test what a unit actually enforces."""

from pathlib import Path
from typing import Annotated

import typer

app = typer.Typer(help="Test what an environment unit actually enforces")


@app.command("enforcement")
def enforcement(
    manifest: Annotated[
        Path, typer.Option(help="The environment's openenv.yaml")
    ] = Path("openenv.yaml"),
    assets: Annotated[
        Path | None, typer.Option(help="Asset root (default: the manifest's directory)")
    ] = None,
    state: Annotated[
        Path, typer.Option(help="Scratch directory for openenvd's state")
    ] = Path("/var/lib/openenvd/check"),
    cgroup_root: Annotated[Path, typer.Option(help="The unit's cgroup root")] = Path(
        "/sys/fs/cgroup"
    ),
) -> None:
    """
    Start a throwaway container in every zone and try what the zone must not do.

    Run it inside the unit, as the user openenvd runs as. It prints the strength
    each guarantee achieved and exits non-zero if any falls below what the
    manifest's `enforcement.require` asks for.
    """
    from openenv.core.openenvd.daemon import main

    argv = [
        "--check",
        "--manifest",
        str(manifest),
        "--state",
        str(state),
        "--cgroup-root",
        str(cgroup_root),
    ]
    if assets is not None:
        argv += ["--assets", str(assets)]
    raise typer.Exit(main(argv))


@app.command("tier")
def tier(
    state: Annotated[
        Path, typer.Option(help="Scratch directory for the probes")
    ] = Path("/tmp/openenvd-probe"),
    cgroup_root: Annotated[Path, typer.Option(help="The unit's cgroup root")] = Path(
        "/sys/fs/cgroup"
    ),
) -> None:
    """Probe this machine and print the tier and guarantees openenvd would get."""
    from openenv.core.openenvd.daemon import main

    raise typer.Exit(
        main(["--probe", "--state", str(state), "--cgroup-root", str(cgroup_root)])
    )
