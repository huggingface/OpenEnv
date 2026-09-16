"""Grader for the `fix-sum-bug` example task.

Imports `stats.py` out of the working directory, runs a handful of checks, and
writes the score into the verifier log directory using Harbor's reward contract:

- `reward.json` — flat map of numeric metrics, including the primary `reward`
- `reward.txt`  — the same scalar, for runtimes that only read the plain file

Depends on nothing outside the standard library so it runs on a bare
`python:3-slim` image.
"""

import importlib.util
import json
import os
import sys
from pathlib import Path


LOGS_DIR = Path(os.environ.get("HARBOR_LOGS_DIR", "/logs/verifier"))
WORKDIR = Path(os.environ.get("HARBOR_WORKDIR", os.getcwd()))

CHECKS = [
    ("total_empty", lambda m: m.total([]) == 0),
    ("total_values", lambda m: m.total([1, 2, 3]) == 6),
    ("total_single", lambda m: m.total([7]) == 7),
    ("mean_empty", lambda m: m.mean([]) == 0.0),
    ("mean_values", lambda m: m.mean([2, 4]) == 3.0),
]


def load_stats():
    """Import `stats.py` from the working directory."""
    module_path = WORKDIR / "stats.py"
    spec = importlib.util.spec_from_file_location("stats_under_test", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not import {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    results: dict[str, float] = {}
    try:
        module = load_stats()
    except Exception as exc:
        print(f"could not load stats.py: {exc}")
        write_reward(0.0, {name: 0.0 for name, _ in CHECKS})
        return 1

    for name, check in CHECKS:
        try:
            passed = bool(check(module))
        except Exception as exc:
            passed = False
            print(f"{name}: raised {type(exc).__name__}: {exc}")
        results[name] = 1.0 if passed else 0.0
        print(f"{name}: {'PASS' if passed else 'FAIL'}")

    reward = sum(results.values()) / len(results)
    print(f"score: {reward:.4f} ({int(sum(results.values()))}/{len(results)} checks)")
    write_reward(reward, results)
    return 0 if reward == 1.0 else 1


def write_reward(reward: float, checks: dict[str, float]) -> None:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    payload = {"reward": round(reward, 6), **checks}
    (LOGS_DIR / "reward.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    (LOGS_DIR / "reward.txt").write_text(f"{reward:.6f}\n", encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())
