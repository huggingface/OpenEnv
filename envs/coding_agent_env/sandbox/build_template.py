# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Build a pre-baked E2B template with opencode already installed."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from e2b import default_build_logger, Template

_REPO_ROOT = Path(__file__).resolve().parents[3]


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
    template = (
        Template()
        .from_python_image("3.12")
        .set_user("user")
        .run_cmd("curl -fsSL https://opencode.ai/install | bash")
        .run_cmd("/home/user/.opencode/bin/opencode --version")
        .make_dir("/home/user/.config/opencode")
        .make_dir("/home/user/logs/agent")
        .make_dir("/home/user/logs/verifier")
        .make_dir("/home/user/task")
        .make_dir("/home/user/workdir")
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
    p.add_argument("--name", default="coding-agent-rl")
    p.add_argument("--skip-cache", action="store_true")
    args = p.parse_args(argv)
    _load_env(_REPO_ROOT / "envs" / "coding_agent_env" / "sandbox" / ".env")
    if not os.environ.get("E2B_API_KEY"):
        print("ERROR: E2B_API_KEY required.", file=sys.stderr)
        return 2
    template_id = build_template(args.name, skip_cache=args.skip_cache)
    print(f"Built. Template id/name: {template_id}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
