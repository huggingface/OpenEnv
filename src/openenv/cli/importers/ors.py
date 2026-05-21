"""Open Reward Standard source importer."""

from __future__ import annotations

import ast
import shutil
import textwrap
from pathlib import Path

from openenv.cli.commands.init import (
    _copy_template_directory,
    _create_template_replacements,
)

from .base import DetectedEnvironment


_ORS_MODULES = {
    "ors",
    "ors.environment",
    "openreward",
    "openreward.environment",
    "openrewardstandard",
    "openrewardstandard.environment",
}

_EXCLUDED_DIRS = {
    ".git",
    ".hg",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".tox",
    ".venv",
    "__pycache__",
    "build",
    "dist",
    "node_modules",
    "venv",
}

_EXCLUDED_FILE_SUFFIXES = {".pyc", ".pyo"}


def _is_excluded(path: Path) -> bool:
    return any(part in _EXCLUDED_DIRS for part in path.parts)


def _iter_python_files(source: Path) -> list[Path]:
    return [
        path
        for path in sorted(source.rglob("*.py"))
        if not _is_excluded(path.relative_to(source))
    ]


def _module_path(source: Path, file_path: Path) -> str:
    rel = file_path.relative_to(source).with_suffix("")
    parts = list(rel.parts)
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def _dotted_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = _dotted_name(node.value)
        if base:
            return f"{base}.{node.attr}"
    if isinstance(node, ast.Subscript):
        return _dotted_name(node.value)
    return None


def _collect_environment_aliases(
    tree: ast.AST,
) -> tuple[set[str], dict[str, str]]:
    environment_aliases: set[str] = set()
    module_aliases: dict[str, str] = {}

    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if module in _ORS_MODULES:
                for alias in node.names:
                    if alias.name == "Environment":
                        environment_aliases.add(alias.asname or alias.name)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name in _ORS_MODULES:
                    module_aliases[alias.asname or alias.name.split(".", 1)[0]] = (
                        alias.name
                    )

    return environment_aliases, module_aliases


def _inherits_ors_environment(
    base: ast.AST,
    environment_aliases: set[str],
    module_aliases: dict[str, str],
) -> bool:
    dotted = _dotted_name(base)
    if dotted is None:
        return False
    if dotted in environment_aliases:
        return True

    for alias, module in module_aliases.items():
        if dotted == f"{alias}.Environment":
            return module in {"ors", "openreward", "openrewardstandard"}
        if dotted == f"{alias}.environment.Environment":
            return module in _ORS_MODULES
        if dotted == f"{alias}.Environment" and module.endswith(".environment"):
            return True
    return False


def detect_ors_environments(source: Path) -> list[DetectedEnvironment]:
    """Detect ORS/OpenReward environment classes without importing source files."""
    source = source.resolve()
    matches: list[DetectedEnvironment] = []

    for file_path in _iter_python_files(source):
        try:
            tree = ast.parse(file_path.read_text(encoding="utf-8"))
        except (SyntaxError, UnicodeDecodeError):
            continue

        environment_aliases, module_aliases = _collect_environment_aliases(tree)
        if not environment_aliases and not module_aliases:
            continue

        for node in tree.body:
            if not isinstance(node, ast.ClassDef):
                continue
            if any(
                _inherits_ors_environment(base, environment_aliases, module_aliases)
                for base in node.bases
            ):
                matches.append(
                    DetectedEnvironment(
                        source_type="ors",
                        class_name=node.name,
                        module_path=_module_path(source, file_path),
                        file_path=file_path,
                    )
                )

    return matches


def _safe_vendor_dir_name(source: Path) -> str:
    name = source.name.strip().replace("-", "_")
    return name if name.isidentifier() else "source"


def _copy_source_tree(source: Path, destination: Path) -> None:
    def ignore(_dir: str, names: list[str]) -> set[str]:
        ignored = set()
        for name in names:
            path = Path(name)
            if name in _EXCLUDED_DIRS or path.suffix in _EXCLUDED_FILE_SUFFIXES:
                ignored.add(name)
        return ignored

    shutil.copytree(source, destination, ignore=ignore)


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(content).lstrip(), encoding="utf-8", newline="\n")


def _append_dependency_files(env_dir: Path, env_name: str) -> None:
    requirements = env_dir / "server" / "requirements.txt"
    if requirements.exists():
        content = requirements.read_text(encoding="utf-8")
        if "ors-sdk" not in content:
            requirements.write_text(
                content.rstrip() + "\nors-sdk>=0.1.0\n",
                encoding="utf-8",
                newline="\n",
            )

    pyproject = env_dir / "pyproject.toml"
    if pyproject.exists():
        content = pyproject.read_text(encoding="utf-8")
        if '"ors-sdk' not in content:
            content = content.replace(
                '    "openenv-core[core]>=0.2.2",',
                '    "openenv-core[core]>=0.2.2",\n    "ors-sdk>=0.1.0",',
            )
        if "[tool.setuptools.package-data]" not in content:
            content += (
                '\n[tool.setuptools.package-data]\n'
                f'"{env_name}" = ["vendor/**/*"]\n'
            )
        pyproject.write_text(content, encoding="utf-8", newline="\n")


def _wrapper_source(
    *,
    env_name: str,
    class_name_prefix: str,
    source_module: str,
    source_class: str,
    vendor_dir: str,
) -> str:
    return f'''
    from __future__ import annotations

    import asyncio
    import inspect
    import sys
    import threading
    from importlib import import_module
    from pathlib import Path
    from typing import Any
    from uuid import uuid4

    from openenv.core.env_server.interfaces import Environment
    from openenv.core.env_server.mcp_types import (
        CallToolAction,
        CallToolObservation,
        ListToolsAction,
        ListToolsObservation,
        Tool,
        ToolError,
        ToolErrorType,
    )
    from openenv.core.env_server.types import Observation, State


    _VENDORED_SOURCE_ROOT = Path(__file__).resolve().parents[1] / "vendor" / "{vendor_dir}"
    if str(_VENDORED_SOURCE_ROOT) not in sys.path:
        sys.path.insert(0, str(_VENDORED_SOURCE_ROOT))

    _ORIGINAL_ENV_CLASS = getattr(import_module("{source_module}"), "{source_class}")


    def _run_sync(value: Any) -> Any:
        if not inspect.isawaitable(value):
            return value
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(value)

        result: dict[str, Any] = {{}}

        def runner() -> None:
            try:
                result["value"] = asyncio.run(value)
            except BaseException as exc:
                result["error"] = exc

        thread = threading.Thread(target=runner)
        thread.start()
        thread.join()
        if "error" in result:
            raise result["error"]
        return result.get("value")


    def _dump(value: Any) -> Any:
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        if isinstance(value, list):
            return [_dump(item) for item in value]
        if isinstance(value, tuple):
            return [_dump(item) for item in value]
        if isinstance(value, dict):
            return {{str(key): _dump(item) for key, item in value.items()}}
        if hasattr(value, "model_dump"):
            return _dump(value.model_dump())
        if hasattr(value, "__dict__"):
            return _dump(value.__dict__)
        return str(value)


    def _normalize_split(split: Any) -> dict[str, Any]:
        value = _dump(split)
        if isinstance(value, dict):
            return value
        split_name = str(value)
        split_type = split_name if split_name in {{"train", "validation", "test"}} else "validation"
        return {{"name": split_name, "type": split_type}}


    def _tool_from_ors(tool: Any) -> Tool:
        value = _dump(tool)
        input_schema = value.get("input_schema") or value.get("inputSchema")
        if input_schema is None:
            input_schema = {{"type": "object", "properties": {{}}}}
        return Tool(
            name=value["name"],
            description=value.get("description") or "",
            input_schema=input_schema,
        )


    class {class_name_prefix}Environment(Environment):
        """OpenEnv wrapper around a vendored ORS/OpenReward environment."""

        SUPPORTS_CONCURRENT_SESSIONS = False

        def __init__(self):
            self._ors_cls = _ORIGINAL_ENV_CLASS
            self._ors_env: Any | None = None
            self._state = State(episode_id=str(uuid4()), step_count=0)
            self._task_spec: Any | None = None
            self._last_reward: float | None = None
            self._done = False

        def list_splits(self) -> list[dict[str, Any]]:
            splits = _run_sync(self._ors_cls.list_splits())
            return [_normalize_split(split) for split in splits]

        def list_tasks(self, split: str) -> list[Any]:
            return _dump(_run_sync(self._ors_cls.list_tasks(split)))

        def num_tasks(self, split: str) -> int:
            return int(_run_sync(self._ors_cls.num_tasks(split)))

        def get_task(self, split: str, index: int) -> Any:
            return _dump(_run_sync(self._ors_cls.get_task(split, index)))

        def get_task_range(
            self,
            split: str,
            start: int | None = None,
            stop: int | None = None,
        ) -> list[Any]:
            return _dump(_run_sync(self._ors_cls.get_task_range(split, start, stop)))

        def _first_task(self) -> tuple[str, int, Any]:
            splits = self.list_splits()
            if not splits:
                raise RuntimeError("ORS environment has no splits")
            split = splits[0]["name"]
            return split, 0, self.get_task(split, 0)

        def reset(
            self,
            seed: int | None = None,
            episode_id: str | None = None,
            task_spec: dict[str, Any] | None = None,
            split: str | None = None,
            index: int | None = None,
            secrets: dict[str, str] | None = None,
            **kwargs: Any,
        ) -> Observation:
            self.close()
            if task_spec is None:
                if split is None and index is None:
                    split, index, task_spec = self._first_task()
                elif split is None or index is None:
                    raise ValueError("split and index must be provided together")
                else:
                    task_spec = self.get_task(split, index)

            self._task_spec = _dump(task_spec)
            self._ors_env = self._ors_cls(
                task_spec=task_spec,
                secrets=secrets or {{}},
            )
            _run_sync(self._ors_env.setup())
            prompt = _dump(_run_sync(self._ors_env.get_prompt()))
            self._last_reward = None
            self._done = False
            self._state = State(
                episode_id=episode_id or str(uuid4()),
                step_count=0,
                source_type="ors",
                original_env_class="{source_class}",
                task_spec=self._task_spec,
                split=split,
                index=index,
            )
            return Observation(
                done=False,
                reward=None,
                metadata={{
                    "source_type": "ors",
                    "original_env_class": "{source_class}",
                    "task_spec": self._task_spec,
                    "prompt": prompt,
                }},
            )

        def _ensure_session(self) -> None:
            if self._ors_env is None:
                self.reset()

        def _all_tools(self) -> list[Tool]:
            shared = _run_sync(self._ors_cls.list_tools())
            tools = [_tool_from_ors(tool) for tool in getattr(shared, "tools", [])]
            if self._ors_env is not None:
                task_tools = _run_sync(self._ors_env.list_task_tools())
                tools.extend(_tool_from_ors(tool) for tool in getattr(task_tools, "tools", []))
            return tools

        def step(
            self,
            action: Any,
            timeout_s: float | None = None,
            **kwargs: Any,
        ) -> Observation:
            if isinstance(action, ListToolsAction):
                return ListToolsObservation(tools=self._all_tools())
            if not isinstance(action, CallToolAction):
                raise TypeError(f"Unsupported action type: {{type(action).__name__}}")

            self._ensure_session()
            assert self._ors_env is not None
            result = _run_sync(self._ors_env._call_tool(action.tool_name, action.arguments))
            root = getattr(result, "root", result)
            ok = getattr(root, "ok", False)
            self._state.step_count += 1

            if ok:
                output = root.output
                blocks = _dump(getattr(output, "blocks", []))
                metadata = _dump(getattr(output, "metadata", None)) or {{}}
                reward = getattr(output, "reward", None)
                done = bool(getattr(output, "finished", False))
                self._last_reward = reward
                self._done = done
                return CallToolObservation(
                    tool_name=action.tool_name,
                    result={{"blocks": blocks, "metadata": metadata}},
                    reward=reward,
                    done=done,
                    metadata=metadata,
                )

            message = str(getattr(root, "error", "ORS tool call failed"))
            return CallToolObservation(
                tool_name=action.tool_name,
                result=None,
                error=ToolError(
                    error_type=ToolErrorType.EXECUTION_ERROR,
                    message=message,
                ),
                reward=None,
                done=False,
            )

        @property
        def state(self) -> State:
            return self._state

        def close(self) -> None:
            if self._ors_env is None:
                return
            try:
                _run_sync(self._ors_env.teardown())
            finally:
                self._ors_env = None
    '''


def _app_source(*, env_name: str, class_name_prefix: str) -> str:
    return f'''
    from __future__ import annotations

    from openenv.core.env_server.http_server import create_app
    from openenv.core.env_server.mcp_types import CallToolAction, CallToolObservation

    from .{env_name}_environment import {class_name_prefix}Environment


    app = create_app(
        {class_name_prefix}Environment,
        CallToolAction,
        CallToolObservation,
        env_name="{env_name}",
        max_concurrent_envs=1,
    )


    def main(host: str = "0.0.0.0", port: int = 8000):
        import uvicorn

        uvicorn.run(app, host=host, port=port)


    if __name__ == "__main__":
        import argparse

        parser = argparse.ArgumentParser()
        parser.add_argument("--port", type=int, default=8000)
        args = parser.parse_args()
        main(port=args.port)
    '''


class ORSImporter:
    """Importer for source repos that define ORS/OpenReward environments."""

    source_type = "ors"

    def detect(self, source: Path) -> list[DetectedEnvironment]:
        return detect_ors_environments(source)

    def generate(
        self,
        *,
        source: Path,
        destination: Path,
        env_name: str,
        detected: DetectedEnvironment,
    ) -> None:
        replacements = _create_template_replacements(env_name)
        _copy_template_directory(
            "openenv.cli.templates.openenv_env",
            "",
            destination,
            replacements,
            env_name,
        )

        vendor_dir = _safe_vendor_dir_name(source)
        _copy_source_tree(source, destination / "vendor" / vendor_dir)

        prefix = replacements["__ENV_CLASS_NAME__"]
        _write_text(
            destination / "server" / f"{env_name}_environment.py",
            _wrapper_source(
                env_name=env_name,
                class_name_prefix=prefix,
                source_module=detected.module_path,
                source_class=detected.class_name,
                vendor_dir=vendor_dir,
            ),
        )
        _write_text(
            destination / "server" / "app.py",
            _app_source(env_name=env_name, class_name_prefix=prefix),
        )
        _append_dependency_files(destination, env_name)
