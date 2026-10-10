# SPDX-License-Identifier: BSD-3-Clause

"""Envs whose instances share no state serve several WebSocket sessions at once.

The check reads the source, so it runs without any of the envs' dependencies.
"""

import ast
from pathlib import Path

import pytest

ENVS = Path(__file__).resolve().parents[2] / "envs"


@pytest.mark.parametrize(
    "env_dir, module, class_name",
    [
        ("atari_env", "atari_environment", "AtariEnvironment"),
        ("chess_env", "chess_environment", "ChessEnvironment"),
        ("connect4_env", "connect4_environment", "Connect4Environment"),
        ("grid_world_env", "grid_world_environment", "GridWorldEnvironment"),
        ("maze_env", "maze_env_environment", "MazeEnvironment"),
        ("pelican_svg_env", "pelican_svg_environment", "PelicanSvgEnvironment"),
        ("snake_env", "snake_environment", "SnakeEnvironment"),
    ],
)
def test_env_serves_concurrent_sessions(env_dir, module, class_name):
    server = ENVS / env_dir / "server"

    environment = ast.parse((server / f"{module}.py").read_text())
    env_class = next(
        node
        for node in ast.walk(environment)
        if isinstance(node, ast.ClassDef) and node.name == class_name
    )
    declarations = {ast.unparse(statement) for statement in env_class.body}
    assert declarations & {
        "SUPPORTS_CONCURRENT_SESSIONS = True",
        "SUPPORTS_CONCURRENT_SESSIONS: bool = True",
    }

    app = ast.parse((server / "app.py").read_text())
    create_app = next(
        node
        for node in ast.walk(app)
        if isinstance(node, ast.Call) and ast.unparse(node.func) == "create_app"
    )
    max_concurrent_envs = {
        keyword.arg: ast.unparse(keyword.value) for keyword in create_app.keywords
    }.get("max_concurrent_envs")
    assert max_concurrent_envs == "int(os.getenv('MAX_CONCURRENT_ENVS', '8'))"
