# SPDX-License-Identifier: BSD-3-Clause

"""An env that declares a `State` subclass must serve it on `/schema` and `/state`.

`HTTPEnvServer` defaults `state_cls` to the base `State`, which publishes only
`episode_id` and `step_count` and strips every field a subclass declares from the
`/state` body. An env therefore has to pass its own class to the app factory; declaring
`class FooState(State)` is not enough on its own.

The WebSocket `state` frame calls `model_dump()` on the live environment and keeps those
fields either way, so an env that forgets this does not fail anywhere. It just serves two
different answers for the same object depending on the transport, which is what makes the
omission worth a test rather than a review note.

Checked with `ast` rather than by importing: importing every env's `server/app.py` would
pull in playwright, carla, dm_control and the rest of the optional-dependency tail, so this
would skip on exactly the machines that should be guarding it.
"""

from __future__ import annotations

import ast
import pathlib
import warnings

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
ENVS_DIR = REPO_ROOT / "envs"

# Every helper that ends up constructing an HTTPEnvServer.
APP_FACTORIES = {"create_app", "create_fastapi_app", "HTTPEnvServer"}


def _parse(path: pathlib.Path) -> ast.Module | None:
    try:
        with warnings.catch_warnings():
            # Parsing every env module surfaces pre-existing SyntaxWarnings (an env with
            # an unescaped backslash in a docstring, say). They belong to that env, not to
            # what this test checks, and they drown the run's own output.
            warnings.simplefilter("ignore", SyntaxWarning)
            return ast.parse(path.read_text(encoding="utf-8"))
    except (OSError, SyntaxError):
        return None


def _state_subclasses(env_dir: pathlib.Path) -> set[str]:
    """Names in this env that subclass `State` directly."""
    found: set[str] = set()
    for module in env_dir.rglob("*.py"):
        tree = _parse(module)
        if tree is None:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and any(
                getattr(base, "id", getattr(base, "attr", None)) == "State"
                for base in node.bases
            ):
                found.add(node.name)
    return found


def _factory_call(tree: ast.Module) -> ast.Call | None:
    """The last app-factory call in the module, which is the one that builds `app`."""
    call = None
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and getattr(node.func, "id", "") in APP_FACTORIES:
            call = node
    return call


def _imported_names(tree: ast.Module) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                names.add(alias.asname or alias.name)
    return names


def _envs_declaring_state() -> list[str]:
    envs = []
    for env_dir in sorted(p for p in ENVS_DIR.iterdir() if p.is_dir()):
        if not (env_dir / "server" / "app.py").exists():
            continue
        if _state_subclasses(env_dir):
            envs.append(env_dir.name)
    return envs


ENVS_WITH_STATE = _envs_declaring_state()


def test_fixture_finds_envs():
    """Guard against the discovery above silently matching nothing."""
    assert len(ENVS_WITH_STATE) > 20, ENVS_WITH_STATE


@pytest.mark.parametrize("env_name", ENVS_WITH_STATE)
def test_env_serves_its_own_state_class(env_name: str):
    env_dir = ENVS_DIR / env_name
    tree = _parse(env_dir / "server" / "app.py")
    assert tree is not None, f"{env_name}: server/app.py does not parse"

    call = _factory_call(tree)
    if call is None:
        pytest.skip(f"{env_name} builds its app outside a recognised factory call")

    keyword = next((kw for kw in call.keywords if kw.arg == "state_cls"), None)
    declared = sorted(_state_subclasses(env_dir))
    assert keyword is not None, (
        f"{env_name} declares {declared} but its app factory does not pass state_cls, "
        "so /schema and /state fall back to the base State model and drop every field "
        "the subclass adds"
    )

    passed = ast.unparse(keyword.value)
    assert passed in declared, (
        f"{env_name} passes state_cls={passed}, which is not a State subclass declared "
        f"in this env ({declared})"
    )
    assert passed in _imported_names(tree), (
        f"{env_name} passes state_cls={passed} but never imports it in server/app.py"
    )
