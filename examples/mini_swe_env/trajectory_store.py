"""Trajectory storage and filtering for teacher rollout collection.

Stores full multi-turn trajectories as JSONL with rich metadata for
downstream SFT/RFT training.

Storage format (one JSON object per line):

    {
        "trajectory_id": "...",
        "instance_id": "...",
        "task_id": "...",
        "repo": "...",
        "teacher_model": "Qwen/Qwen3.6-27B",
        "rollout_index": 0,         # which attempt (0..N-1) for this task
        "resolved": true,
        "reward": 1.0,
        "reward_source": "host_answer_tool",
        "turns": 12,
        "wall_s": 245.3,
        "answer_called": true,
        "tool_calls_count": 15,
        "messages": [...],          # full conversation (system + user + assistant + tool)
        "tool_calls": [...],        # extracted tool call log
        "test_outcomes": {...},     # FAIL_TO_PASS / PASS_TO_PASS results
        "timestamp_utc": "2026-05-24T...",
        "metadata": {...},          # additional context
    }

Usage::

    store = TrajectoryStore("trajectories/teacher_27b")
    store.append(trajectory_record)
    store.flush()

    # Filter resolved trajectories for SFT
    resolved = store.filter(resolved=True, max_turns=15)

    # Compute pass@k
    stats = store.pass_at_k_stats(k_values=[1, 4, 8])
"""

from __future__ import annotations

import json
import logging
import math
import os
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Sequence

_log = logging.getLogger(__name__)


@dataclass
class TrajectoryRecord:
    """One complete rollout trajectory with metadata."""

    # Identifiers
    trajectory_id: str
    instance_id: str
    task_id: str
    repo: str

    # Teacher info
    teacher_model: str

    # Rollout metadata
    rollout_index: int  # which attempt for this task (0..N-1)
    resolved: bool
    reward: float
    reward_source: str

    # Conversation
    turns: int
    wall_s: float
    answer_called: bool
    tool_calls_count: int
    messages: list[dict[str, Any]]  # full conversation history
    tool_calls: list[dict[str, Any]]  # extracted tool calls log

    # Grading details
    test_outcomes: dict[str, Any] = field(default_factory=dict)

    # Timestamps
    timestamp_utc: str = ""

    # Catch-all metadata
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.timestamp_utc:
            self.timestamp_utc = datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "TrajectoryRecord":
        # Handle any extra keys gracefully
        known_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in d.items() if k in known_fields}
        return cls(**filtered)


class TrajectoryStore:
    """JSONL-backed trajectory store with in-memory index.

    Supports:
    - Append-only writes (crash-safe with periodic flush)
    - Filtering by resolved, turn count, etc.
    - pass@k statistics computation
    - Loading from existing JSONL files
    - Periodic upload to HF Hub dataset repo (crash-safe persistence)
    """

    def __init__(
        self,
        output_dir: str | Path,
        *,
        hub_repo_id: str | None = None,
        hub_upload_every: int = 5,
    ) -> None:
        self._dir = Path(output_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._filepath = self._dir / "trajectories.jsonl"
        self._records: list[TrajectoryRecord] = []
        self._pending_writes: list[TrajectoryRecord] = []

        # HF Hub persistence
        self._hub_repo_id = hub_repo_id
        self._hub_upload_every = max(1, hub_upload_every)
        self._since_last_upload = 0
        self._hub_api: Any = None
        if hub_repo_id:
            self._init_hub()

        # Load existing records if file exists
        if self._filepath.exists():
            self._load_existing()

    @property
    def filepath(self) -> Path:
        return self._filepath

    @property
    def records(self) -> list[TrajectoryRecord]:
        return list(self._records)

    def __len__(self) -> int:
        return len(self._records)

    def append(self, record: TrajectoryRecord) -> None:
        """Add a trajectory record (buffered, call flush() to persist)."""
        self._records.append(record)
        self._pending_writes.append(record)

    def flush(self) -> None:
        """Write pending records to disk and upload to Hub if configured."""
        if not self._pending_writes:
            return
        with open(self._filepath, "a") as f:
            for rec in self._pending_writes:
                f.write(json.dumps(rec.to_dict(), ensure_ascii=False) + "\n")
        count = len(self._pending_writes)
        self._pending_writes = []
        _log.debug("flushed %d trajectories to %s", count, self._filepath)

        # Periodic upload to HF Hub
        if self._hub_repo_id:
            self._since_last_upload += count
            if self._since_last_upload >= self._hub_upload_every:
                self._upload_to_hub()
                self._since_last_upload = 0

    def upload_now(self) -> None:
        """Force an immediate upload to HF Hub (call at end of run)."""
        if self._hub_repo_id:
            self._upload_to_hub()

    def _init_hub(self) -> None:
        """Initialize HF Hub repo for trajectory persistence."""
        try:
            from huggingface_hub import HfApi
            self._hub_api = HfApi()
            self._hub_api.create_repo(
                repo_id=self._hub_repo_id,
                repo_type="dataset",
                private=True,
                exist_ok=True,
            )
            _log.info("hub persistence enabled: %s", self._hub_repo_id)

            # Try to download existing trajectories on startup (resume support)
            try:
                from huggingface_hub import hf_hub_download
                local_path = hf_hub_download(
                    repo_id=self._hub_repo_id,
                    filename="trajectories.jsonl",
                    repo_type="dataset",
                    local_dir=str(self._dir),
                )
                _log.info("downloaded existing trajectories from hub")
            except Exception:
                _log.debug("no existing trajectories on hub (fresh start)")

        except ImportError:
            _log.warning("huggingface_hub not installed, hub persistence disabled")
            self._hub_repo_id = None
        except Exception as exc:
            _log.warning("hub init failed: %s (persistence disabled)", exc)
            self._hub_repo_id = None

    def _upload_to_hub(self) -> None:
        """Upload trajectories.jsonl + stats.json to HF Hub."""
        if not self._hub_api or not self._filepath.exists():
            return
        try:
            self._hub_api.upload_file(
                path_or_fileobj=str(self._filepath),
                path_in_repo="trajectories.jsonl",
                repo_id=self._hub_repo_id,
                repo_type="dataset",
                commit_message=f"Update trajectories ({len(self._records)} total)",
            )
            # Also upload stats
            stats_path = self._dir / "stats.json"
            stats = self.pass_at_k_stats()
            stats_summary = {k: v for k, v in stats.items() if k != "per_task"}
            stats_path.write_text(json.dumps(stats_summary, indent=2))
            self._hub_api.upload_file(
                path_or_fileobj=str(stats_path),
                path_in_repo="stats.json",
                repo_id=self._hub_repo_id,
                repo_type="dataset",
                commit_message=f"Update stats ({len(self._records)} trajectories)",
            )
            _log.info(
                "hub_upload complete: %d trajectories to %s",
                len(self._records),
                self._hub_repo_id,
            )
        except Exception as exc:
            _log.warning("hub upload failed (will retry next flush): %s", exc)

    def filter(
        self,
        *,
        resolved: bool | None = None,
        max_turns: int | None = None,
        min_turns: int | None = None,
        teacher_model: str | None = None,
        instance_ids: set[str] | None = None,
    ) -> list[TrajectoryRecord]:
        """Filter trajectories by criteria."""
        results = []
        for rec in self._records:
            if resolved is not None and rec.resolved != resolved:
                continue
            if max_turns is not None and rec.turns > max_turns:
                continue
            if min_turns is not None and rec.turns < min_turns:
                continue
            if teacher_model is not None and rec.teacher_model != teacher_model:
                continue
            if instance_ids is not None and rec.instance_id not in instance_ids:
                continue
            results.append(rec)
        return results

    def resolved_trajectories(self, max_turns: int | None = None) -> list[TrajectoryRecord]:
        """Convenience: get all resolved trajectories, optionally capped by turn count."""
        return self.filter(resolved=True, max_turns=max_turns)

    def pass_at_k_stats(
        self,
        k_values: Sequence[int] = (1, 4, 8),
    ) -> dict[str, Any]:
        """Compute pass@k statistics across all tasks.

        Uses the unbiased estimator:
            pass@k = 1 - C(n-c, k) / C(n, k)
        where n = total rollouts for the task, c = number resolved.

        Returns dict with per-k rates and per-task details.
        """
        # Group by instance_id
        by_task: dict[str, list[TrajectoryRecord]] = defaultdict(list)
        for rec in self._records:
            by_task[rec.instance_id].append(rec)

        stats: dict[str, Any] = {
            "total_tasks": len(by_task),
            "total_trajectories": len(self._records),
            "total_resolved": sum(1 for r in self._records if r.resolved),
            "resolve_rate": (
                sum(1 for r in self._records if r.resolved) / len(self._records)
                if self._records
                else 0.0
            ),
        }

        for k in k_values:
            pass_rates: list[float] = []
            for instance_id, recs in by_task.items():
                n = len(recs)
                c = sum(1 for r in recs if r.resolved)
                if n < k:
                    # Not enough rollouts — use empirical rate
                    pass_rates.append(1.0 if c > 0 else 0.0)
                else:
                    # Unbiased estimator: pass@k = 1 - C(n-c, k) / C(n, k)
                    pass_rates.append(_pass_at_k(n, c, k))

            rate = sum(pass_rates) / len(pass_rates) if pass_rates else 0.0
            stats[f"pass@{k}"] = round(rate, 4)

        # Per-task breakdown
        per_task: dict[str, dict[str, Any]] = {}
        for instance_id, recs in by_task.items():
            n = len(recs)
            c = sum(1 for r in recs if r.resolved)
            per_task[instance_id] = {
                "n_rollouts": n,
                "n_resolved": c,
                "resolve_rate": round(c / n, 4) if n > 0 else 0.0,
                "avg_turns": round(sum(r.turns for r in recs) / n, 1) if n > 0 else 0,
                "avg_wall_s": round(sum(r.wall_s for r in recs) / n, 1) if n > 0 else 0,
            }
        stats["per_task"] = per_task

        return stats

    def summary(self) -> str:
        """Human-readable summary of the store."""
        if not self._records:
            return "TrajectoryStore: empty"

        resolved = sum(1 for r in self._records if r.resolved)
        n_tasks = len(set(r.instance_id for r in self._records))
        avg_turns = sum(r.turns for r in self._records) / len(self._records)
        avg_wall = sum(r.wall_s for r in self._records) / len(self._records)

        lines = [
            f"TrajectoryStore: {self._filepath}",
            f"  Total trajectories: {len(self._records)}",
            f"  Unique tasks: {n_tasks}",
            f"  Resolved: {resolved}/{len(self._records)} ({100*resolved/len(self._records):.1f}%)",
            f"  Avg turns: {avg_turns:.1f}",
            f"  Avg wall time: {avg_wall:.1f}s",
        ]

        stats = self.pass_at_k_stats()
        for k in (1, 4, 8):
            key = f"pass@{k}"
            if key in stats:
                lines.append(f"  {key}: {stats[key]*100:.1f}%")

        return "\n".join(lines)

    def export_for_sft(
        self,
        output_path: str | Path,
        *,
        resolved_only: bool = True,
        max_turns: int | None = 15,
        deduplicate: bool = True,
    ) -> int:
        """Export filtered trajectories in chat-format JSONL for SFT training.

        Each line is: {"messages": [...], "instance_id": "...", "teacher_model": "..."}

        Returns the number of exported trajectories.
        """
        candidates = self.filter(resolved=resolved_only, max_turns=max_turns)

        if deduplicate:
            # Keep shortest successful trajectory per task
            best_per_task: dict[str, TrajectoryRecord] = {}
            for rec in candidates:
                existing = best_per_task.get(rec.instance_id)
                if existing is None or rec.turns < existing.turns:
                    best_per_task[rec.instance_id] = rec
            candidates = list(best_per_task.values())

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        count = 0
        with open(output_path, "w") as f:
            for rec in candidates:
                entry = {
                    "messages": rec.messages,
                    "instance_id": rec.instance_id,
                    "teacher_model": rec.teacher_model,
                    "turns": rec.turns,
                    "trajectory_id": rec.trajectory_id,
                }
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
                count += 1

        _log.info("exported %d trajectories for SFT to %s", count, output_path)
        return count

    def _load_existing(self) -> None:
        """Load records from existing JSONL file."""
        count = 0
        with open(self._filepath) as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                    self._records.append(TrajectoryRecord.from_dict(d))
                    count += 1
                except (json.JSONDecodeError, TypeError) as exc:
                    _log.warning("skipping invalid line %d: %s", line_num, exc)
        if count:
            _log.info("loaded %d existing trajectories from %s", count, self._filepath)


def _pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased pass@k estimator.

    pass@k = 1 - C(n-c, k) / C(n, k)

    Uses log-space computation for numerical stability.
    """
    if n - c < k:
        return 1.0
    if c == 0:
        return 0.0
    # log(C(n-c, k)) - log(C(n, k))
    log_num = sum(math.log(n - c - i) for i in range(k))
    log_den = sum(math.log(n - i) for i in range(k))
    return 1.0 - math.exp(log_num - log_den)


__all__ = [
    "TrajectoryRecord",
    "TrajectoryStore",
]
