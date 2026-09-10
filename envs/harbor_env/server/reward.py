# SPDX-License-Identifier: BSD-3-Clause

"""Harbor's reward-file contract.

A Harbor verifier communicates its verdict by writing files into
`/logs/verifier/`, never through its exit code:

| File          | Format                                     |
| ------------- | ------------------------------------------ |
| `reward.json` | flat JSON map of metric name -> int/float  |
| `reward.txt`  | a single int/float                         |

Harbor reads `reward.json` first and falls back to `reward.txt`; this module
implements exactly that precedence.

The reward is *produced inside the environment* and only forwarded from here —
consistent with OpenEnv's "rewards in environment" invariant. When a verifier
writes no reward file, [`RewardReport.value`] is `None`. We deliberately do not
invent a reward from the exit code: a silently fabricated `0.0` is
indistinguishable from a genuine failure and would poison training data.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable


#: Reads a file name inside `/logs/verifier/`, returning `None` when it is absent.
#: Taking a reader rather than a path keeps this logic identical for a local
#: directory and for files inside a container.
LogReader = Callable[[str], "str | None"]

REWARD_JSON = "reward.json"
REWARD_TXT = "reward.txt"

#: Non-standard sidecar written by Repo2RLEnv verifiers. Harbor requires
#: `reward.json` to be flat and numeric, so the nested component breakdown
#: (f2p/p2p counts, diff-similarity components, judge status) lives here. It is
#: surfaced for analysis but is never the source of the scalar reward.
REWARD_DETAILS_JSON = "reward-details.json"

#: Key treated as the scalar reward when `reward.json` carries several metrics.
PRIMARY_METRIC = "reward"


@dataclass(frozen=True)
class RewardReport:
    """The verdict recovered from a verifier's `/logs/verifier/` output.

    Args:
        value (`float`, *optional*):
            The scalar reward, or `None` when the verifier wrote no reward file.
        source (`str`):
            Which artifact the scalar came from: `"reward.json"`, `"reward.txt"`,
            or `"missing"`.
        metrics (`dict[str, float]`):
            Every numeric metric from `reward.json`.
        details (`dict[str, Any]`):
            Parsed `reward-details.json` sidecar, when present.
        errors (`list[str]`):
            Problems encountered while reading the artifacts.
    """

    value: float | None = None
    source: str = "missing"
    metrics: dict[str, float] = field(default_factory=dict)
    details: dict[str, Any] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)

    @property
    def graded(self) -> bool:
        """Whether the verifier produced a usable reward."""
        return self.value is not None

    def as_info(self) -> dict[str, Any]:
        """A JSON-serializable digest for `Observation.info`."""
        info: dict[str, Any] = {"reward_source": self.source}
        if self.metrics:
            info["reward_metrics"] = self.metrics
        if self.details:
            info["reward_details"] = self.details
        if self.errors:
            info["reward_errors"] = self.errors
        return info


def read_reward(read_text: LogReader) -> RewardReport:
    """Recover the verdict from a verifier log directory.

    Args:
        read_text ([`LogReader`]):
            Reads a file name inside `/logs/verifier/`, returning `None` when it
            does not exist.

    Returns:
        [`RewardReport`]

    Examples:

    ```python
    report = read_reward(lambda name: (logs_dir / name).read_text() if (logs_dir / name).exists() else None)
    report.value    # 0.833333
    report.source   # 'reward.json'
    ```
    """
    errors: list[str] = []
    details = _load_details(read_text, errors)
    metrics, scalar = _load_metrics(read_text, errors)

    if scalar is not None:
        return RewardReport(scalar, REWARD_JSON, metrics, details, errors)

    scalar = _load_scalar_txt(read_text, errors)
    if scalar is not None:
        return RewardReport(scalar, REWARD_TXT, metrics, details, errors)

    return RewardReport(None, "missing", metrics, details, errors)


def _load_metrics(
    read_text: LogReader, errors: list[str]
) -> tuple[dict[str, float], float | None]:
    """Parse `reward.json` into metrics plus, if unambiguous, a scalar."""
    raw = _read(read_text, REWARD_JSON, errors)
    if raw is None:
        return {}, None

    try:
        payload = json.loads(raw)
    except ValueError as exc:
        errors.append(f"{REWARD_JSON} is not valid JSON: {exc}")
        return {}, None
    if not isinstance(payload, dict):
        errors.append(
            f"{REWARD_JSON} must be a JSON object, got {type(payload).__name__}"
        )
        return {}, None

    metrics = {
        str(key): float(val)
        for key, val in payload.items()
        # bool is an int subclass; Harbor's schema is numeric-only.
        if isinstance(val, (int, float)) and not isinstance(val, bool)
    }
    if PRIMARY_METRIC in metrics:
        return metrics, metrics[PRIMARY_METRIC]
    if len(metrics) == 1:
        return metrics, next(iter(metrics.values()))
    if metrics:
        # Several metrics and no agreed primary one: fall through to reward.txt
        # rather than picking arbitrarily.
        errors.append(
            f"{REWARD_JSON} has no {PRIMARY_METRIC!r} key among {sorted(metrics)}; "
            f"falling back to {REWARD_TXT}"
        )
    return metrics, None


def _load_scalar_txt(read_text: LogReader, errors: list[str]) -> float | None:
    raw = _read(read_text, REWARD_TXT, errors)
    if raw is None:
        return None
    try:
        return float(raw.strip())
    except ValueError:
        errors.append(f"{REWARD_TXT} is not a number: {raw.strip()[:80]!r}")
        return None


def _load_details(read_text: LogReader, errors: list[str]) -> dict[str, Any]:
    raw = _read(read_text, REWARD_DETAILS_JSON, errors)
    if raw is None:
        return {}
    try:
        payload = json.loads(raw)
    except ValueError as exc:
        errors.append(f"{REWARD_DETAILS_JSON} is not valid JSON: {exc}")
        return {}
    return payload if isinstance(payload, dict) else {}


def _read(read_text: LogReader, name: str, errors: list[str]) -> str | None:
    try:
        return read_text(name)
    except Exception as exc:  # pragma: no cover - backend-specific failures
        errors.append(f"could not read {name}: {exc}")
        return None
