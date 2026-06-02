# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Build a pre-baked E2B template with opencode + proxy deps already installed.

Run-time per rollout drops from ~3 min (cold install) to ~30s once the
template is built, because we skip:

  - ``curl https://opencode.ai/install | bash`` (~30-90s)
  - ``pip install fastapi uvicorn httpx`` (~30-60s)
  - directory layout setup
  - copying the proxy source

The template ships:

  - opencode CLI at ``/home/user/.opencode/bin/opencode``
  - Python deps for the in-sandbox proxy
  - The proxy source at ``/home/user/proxy/interception.py``
  - Pre-created dirs: ``~/.config/opencode``, ``~/logs/{agent,verifier}``,
    ``~/task``, ``~/workdir``, ``~/proxy``
  - Default workdir: ``/home/user/workdir``

Usage::

    .venv/bin/python envs/opencode_env/tests/build_e2b_template.py
    # → builds (or rebuilds) ``opencode-rl`` template, prints template id

Then ``test_five_sorts_e2e.py`` will use it via ``--template opencode-rl``.

Requires ``E2B_API_KEY`` in the environment. First build is ~3-8 min;
subsequent builds reuse the cache and can finish in <60s.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from e2b import Template, default_build_logger


_ENV_DIR = Path(__file__).resolve().parent
_PROXY_SOURCE = _ENV_DIR / "interception.py"


def _load_env(path: Path) -> None:
    if not path.exists():
        return
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        k = k.strip()
        v = v.strip().strip('"').strip("'")
        if k and k not in os.environ:
            os.environ[k] = v


def build_template(name: str, *, skip_cache: bool = False) -> str:
    if not _PROXY_SOURCE.exists():
        raise RuntimeError(f"proxy source missing at {_PROXY_SOURCE}")

    # Template.copy() resolves relative paths against the caller's source
    # file directory. This script lives next to ``interception.py`` so the
    # bare filename works.

    # Stage 1 (root): system-wide pip deps for the proxy.
    # Stage 2 (user): opencode install + dir layout + proxy copy.
    template = (
        Template()
        .from_python_image("3.12")
        .pip_install(
            [
                "fastapi>=0.104",
                "uvicorn[standard]>=0.24",
                "httpx>=0.27",
            ]
        )
        .set_user("user")
        .run_cmd("curl -fsSL https://opencode.ai/install | bash")
        .run_cmd("/home/user/.opencode/bin/opencode --version")
        .make_dir("/home/user/.config/opencode")
        .make_dir("/home/user/logs/agent")
        .make_dir("/home/user/logs/verifier")
        .make_dir("/home/user/task")
        .make_dir("/home/user/workdir")
        .make_dir("/home/user/proxy")
        .copy("interception.py", "/home/user/proxy/interception.py")
        .set_workdir("/home/user/workdir")
    )
    if skip_cache:
        template = template.skip_cache()

    info = Template.build(
        template,
        name,
        cpu_count=2,
        memory_mb=2048,
        on_build_logs=default_build_logger(),
    )
    return info.template_id if hasattr(info, "template_id") else str(info)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="build_e2b_template")
    p.add_argument(
        "--name",
        default="opencode-rl",
        help="Template name (default: opencode-rl).",
    )
    p.add_argument(
        "--skip-cache",
        action="store_true",
        help="Force a clean rebuild, ignoring cache.",
    )
    args = p.parse_args(argv)

    _load_env(_ENV_DIR / ".env")
    if not os.environ.get("E2B_API_KEY"):
        print("ERROR: E2B_API_KEY required.", file=sys.stderr)
        return 2

    print(f"Building template '{args.name}' "
          f"(proxy source: {_PROXY_SOURCE})")
    print(f"Skip cache: {args.skip_cache}")
    print()

    template_id = build_template(args.name, skip_cache=args.skip_cache)
    print()
    print(f"Built. Template id/name: {template_id}")
    print(f"Use in code: Sandbox.create(template='{args.name}')")
    return 0


if __name__ == "__main__":
    sys.exit(main())
