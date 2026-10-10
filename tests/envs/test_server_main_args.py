# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The `server` console script calls `main()` with no arguments.

So `main()` cannot take host/port parameters (they would be unreachable), and
an env that advertises `server --port` must parse argv inside `main()`.
Importing every env would pull in its optional dependencies, so the `main`
function is extracted with `ast` and run on its own with `uvicorn.run` patched.
"""

from __future__ import annotations

import ast
import os
import pathlib
import sys

import pytest
import uvicorn

ENVS_DIR = pathlib.Path(__file__).resolve().parents[2] / "envs"
APPS = sorted(ENVS_DIR.glob("*/server/app.py"))


def _main_def(path: pathlib.Path) -> ast.FunctionDef:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "main"
    )


@pytest.mark.parametrize("path", APPS, ids=lambda p: p.parent.parent.name)
def test_server_main_parses_port(path: pathlib.Path, monkeypatch) -> None:
    main_def = _main_def(path)
    assert not main_def.args.args, "the `server` console script calls main() bare"

    if "--port" not in ast.unparse(main_def):
        pytest.skip("main() does not advertise --port")

    calls = []
    monkeypatch.setattr(uvicorn, "run", lambda app, **kwargs: calls.append(kwargs))
    monkeypatch.setattr(
        sys, "argv", ["server", "--host", "127.0.0.1", "--port", "8001"]
    )
    namespace = {"app": object(), "os": os}
    exec(
        compile(ast.Module(body=[main_def], type_ignores=[]), str(path), "exec"),
        namespace,
    )
    namespace["main"]()

    assert calls == [{"host": "127.0.0.1", "port": 8001}]
