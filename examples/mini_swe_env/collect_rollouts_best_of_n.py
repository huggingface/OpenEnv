#!/usr/bin/env python3
"""Teacher trajectory collection via Best-of-N rollouts.

Runs a teacher model (e.g. Qwen3.6-27B) through the Pi scaffold on
SWE-Gym tasks, collecting N rollouts per task with full trajectory
capture for downstream SFT distillation.

Architecture:
    HF Sandbox (SWE-Gym image) → Pi agent
        → InterceptionServer (this machine, public URL)
            → forward to teacher vLLM endpoint
            ← response back to Pi
    On answer() → host-side grading (FAIL_TO_PASS / PASS_TO_PASS)

Required env vars:
    SWE_LLM_BASE_URL       Teacher vLLM endpoint (e.g. https://your-api.com/v1)
    SWE_LLM_API_KEY        Bearer token for the endpoint
    SWE_LLM_MODEL          Model name (e.g. Qwen/Qwen3.6-27B)
    INTERCEPTION_BASE_URL  Public URL where HF sandboxes can reach this machine
    INTERCEPTION_AUTH_TOKEN Auth token for interception server

Optional:
    HF_TOKEN               HuggingFace token (for sandbox creation)

Example:
    SWE_LLM_BASE_URL=https://your-vllm.com/v1 \
    SWE_LLM_API_KEY=sk-... \
    SWE_LLM_MODEL=Qwen/Qwen3.6-27B \
    INTERCEPTION_BASE_URL=https://your-public-url.example.com \
    INTERCEPTION_AUTH_TOKEN=secret123 \
    PYTHONPATH=src:envs python examples/mini_swe_env/collect_rollouts_best_of_n.py \
        --n-rollouts 4 --max-concurrent 3 --output-dir trajectories/teacher_27b
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import time
import uuid
from collections import defaultdict
from pathlib import Path
from typing import Any

import httpx

_root = Path(__file__).resolve().parent.parent.parent
for _p in (_root / "src", _root / "envs"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from mini_swe_env.harness import SWEAgentConfig, SWESessionFactory
from mini_swe_env.task_loader_swegym import load_swegym_tasks
from openenv.core.harness.agents.interception_server import InterceptionServer
from openenv.core.harness.sandbox import create_sandbox_backend

sys.path.insert(0, str(Path(__file__).resolve().parent))
from trajectory_store import TrajectoryRecord, TrajectoryStore

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
_log = logging.getLogger("collect-teacher")


# ── Rate Limiter ───────────────────────────────────────────────────────────


class TokenBucketRateLimiter:
    """Async token-bucket rate limiter with exponential backoff on 429s.

    Allows `rate` requests per `per` seconds. On rate-limit hits (429),
    applies exponential backoff before retrying.
    """

    def __init__(self, rate: float, per: float = 60.0) -> None:
        self._rate = rate
        self._per = per
        self._tokens = rate
        self._last_refill = time.monotonic()
        self._lock = asyncio.Lock()
        self._backoff_until = 0.0  # monotonic time until which we're backing off
        self._consecutive_429s = 0

    async def acquire(self) -> None:
        """Wait until a token is available."""
        while True:
            async with self._lock:
                now = time.monotonic()

                # Respect backoff
                if now < self._backoff_until:
                    wait = self._backoff_until - now
                    _log.debug("rate limiter backing off %.1fs", wait)
                else:
                    # Refill tokens
                    elapsed = now - self._last_refill
                    self._tokens = min(
                        self._rate, self._tokens + elapsed * (self._rate / self._per)
                    )
                    self._last_refill = now

                    if self._tokens >= 1.0:
                        self._tokens -= 1.0
                        self._consecutive_429s = 0
                        return
                    wait = (1.0 - self._tokens) * (self._per / self._rate)

            await asyncio.sleep(wait)

    def report_429(self) -> None:
        """Report a 429 response to trigger exponential backoff."""
        self._consecutive_429s += 1
        backoff = min(2 ** self._consecutive_429s, 120.0)  # cap at 2 min
        self._backoff_until = time.monotonic() + backoff
        _log.warning(
            "429 rate limit hit (consecutive=%d), backing off %.1fs",
            self._consecutive_429s,
            backoff,
        )

    def report_success(self) -> None:
        """Report a successful request to reset backoff state."""
        if self._consecutive_429s > 0:
            self._consecutive_429s = 0
            _log.debug("rate limiter backoff reset after success")


# ── CLI Args ───────────────────────────────────────────────────────────────


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Collect Best-of-N teacher trajectories for SFT distillation"
    )
    p.add_argument(
        "--task-variant",
        default="lite",
        choices=["lite", "full"],
        help="SWE-Gym variant (default: lite, 230 tasks)",
    )
    p.add_argument(
        "--n-rollouts",
        type=int,
        default=4,
        help="Number of rollouts per task (default: 4)",
    )
    p.add_argument(
        "--max-concurrent",
        type=int,
        default=3,
        help="Max concurrent rollouts (default: 3)",
    )
    p.add_argument(
        "--max-turns",
        type=int,
        default=50,
        help="Max agent turns per rollout (default: 50)",
    )
    p.add_argument(
        "--agent-timeout-s",
        type=float,
        default=1800.0,
        help="Agent timeout in seconds (default: 1800)",
    )
    p.add_argument(
        "--request-timeout-s",
        type=float,
        default=180.0,
        help="Per-LLM-request timeout (default: 180s)",
    )
    p.add_argument(
        "--rate-limit",
        type=float,
        default=30.0,
        help="Max LLM requests per minute (default: 30)",
    )
    p.add_argument(
        "--output-dir",
        default="trajectories/teacher_27b",
        help="Output directory for trajectories (default: trajectories/teacher_27b)",
    )
    p.add_argument(
        "--interception-port",
        type=int,
        default=9090,
        help="InterceptionServer port (default: 9090)",
    )
    p.add_argument(
        "--interception-host",
        default="0.0.0.0",
        help="InterceptionServer bind host (default: 0.0.0.0)",
    )
    p.add_argument(
        "--hf-flavor",
        default="cpu-basic",
        help="HF Sandbox flavor (default: cpu-basic)",
    )
    p.add_argument(
        "--max-tasks",
        type=int,
        default=None,
        help="Limit to first N tasks (for testing)",
    )
    p.add_argument(
        "--start-task",
        type=int,
        default=0,
        help="Start from task index (for resuming partial runs)",
    )
    p.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Max retries per rollout on infra failure (default: 3)",
    )
    p.add_argument(
        "--export-sft",
        action="store_true",
        help="Export SFT-ready JSONL after collection",
    )
    p.add_argument(
        "--hub-repo-id",
        default=os.environ.get("TRAJECTORY_HUB_REPO", ""),
        help="HF Dataset repo for trajectory persistence (default: $TRAJECTORY_HUB_REPO)",
    )
    p.add_argument(
        "--hub-upload-every",
        type=int,
        default=5,
        help="Upload to Hub every N trajectories (default: 5)",
    )
    return p


# ── Env Var Helpers ────────────────────────────────────────────────────────


def _must_env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(f"Missing required env var: {name}")
    return value


# ── Single Rollout ─────────────────────────────────────────────────────────


async def _run_one_rollout(
    *,
    gym_task: Any,
    rollout_index: int,
    factory: SWESessionFactory,
    server: InterceptionServer,
    client: httpx.AsyncClient,
    llm_base_url: str,
    llm_api_key: str,
    llm_model: str,
    interception_base_url: str,
    rate_limiter: TokenBucketRateLimiter,
    max_turns: int,
    request_timeout_s: float,
    agent_timeout_s: float,
) -> TrajectoryRecord | None:
    """Run one complete rollout, returning a TrajectoryRecord or None on failure."""

    swe_task = gym_task.to_swe_task()
    episode_id = f"collect-{gym_task.instance_id}-r{rollout_index}-{uuid.uuid4().hex[:6]}"

    _log.info(
        "rollout_start instance_id=%s rollout=%d episode=%s",
        gym_task.instance_id,
        rollout_index,
        episode_id,
    )

    t0 = time.time()
    session = factory.create(task=swe_task, episode_id=episode_id)

    turns = 0
    answer_called = False
    answer_bridged = False
    messages_captured: list[dict[str, Any]] = []
    tool_calls_log: list[dict[str, Any]] = []

    try:
        while turns < max_turns:
            intercept = await session.next_request(timeout_s=agent_timeout_s)
            if intercept is None:
                _log.info(
                    "agent_exited instance_id=%s turns=%d",
                    gym_task.instance_id,
                    turns,
                )
                break

            turns += 1

            # Capture the request messages
            body = intercept.get("body") or {}
            req_messages = body.get("messages") or intercept.get("messages") or []

            # Only capture on first turn (full history) or append new msgs
            if turns == 1:
                messages_captured = list(req_messages)

            # ── Rate-limited LLM call with exponential backoff ──
            await rate_limiter.acquire()

            llm_resp = await _forward_to_llm(
                client,
                intercept=intercept,
                base_url=llm_base_url,
                api_key=llm_api_key,
                model=llm_model,
                timeout_s=request_timeout_s,
                rate_limiter=rate_limiter,
            )

            choice0 = (llm_resp.get("choices") or [{}])[0]
            msg = choice0.get("message") or {}
            tool_calls = msg.get("tool_calls") or []

            # Capture assistant message
            assistant_msg: dict[str, Any] = {"role": "assistant"}
            if msg.get("content"):
                assistant_msg["content"] = msg["content"]
            if tool_calls:
                assistant_msg["tool_calls"] = tool_calls
            messages_captured.append(assistant_msg)

            # Log tool calls
            for tc in tool_calls:
                fn = tc.get("function") or {}
                tool_calls_log.append(
                    {
                        "turn": turns,
                        "name": fn.get("name", ""),
                        "arguments": fn.get("arguments", ""),
                        "id": tc.get("id", ""),
                    }
                )

            # Check for answer tool call
            if any(
                (tc.get("function") or {}).get("name") == "answer"
                for tc in tool_calls
            ):
                answer_called = True

            await session.deliver(intercept, llm_resp)

            # Check if host-side answer was bridged
            if bool(getattr(session, "answer_called", False)):
                answer_called = True
            if bool(getattr(session, "answer_bridged", False)):
                answer_bridged = True

            if answer_called:
                break

            _log.debug(
                "turn=%d instance=%s tools=%s",
                turns,
                gym_task.instance_id,
                [fn.get("name", "?") for tc in tool_calls for fn in [tc.get("function") or {}]],
            )

        # ── Grade ─────────────────────────────────────────────────
        vr = session.verify(transcript=[])
        reward = float(getattr(vr, "env_reward", 0.0) or 0.0)
        resolved = reward >= 1.0
        reward_source = (vr.metrics or {}).get("reward_source", "unknown")
        test_outcomes = (vr.artifacts or {}).get("verify_details", {})
        wall_s = round(time.time() - t0, 2)

        _log.info(
            "rollout_done instance_id=%s rollout=%d resolved=%s reward=%.2f "
            "turns=%d wall_s=%.1f source=%s",
            gym_task.instance_id,
            rollout_index,
            resolved,
            reward,
            turns,
            wall_s,
            reward_source,
        )

        # Capture agent log if zero turns (agent never called LLM)
        if turns == 0:
            try:
                artifacts = session.collect_artifacts()
                agent_log = artifacts.get("agent_log", "") if isinstance(artifacts, dict) else ""
                if isinstance(agent_log, str) and agent_log.strip():
                    _log.warning("zero_turns agent_log tail:\n%s", agent_log[-3000:])
            except Exception as log_exc:
                _log.warning("failed to collect agent log: %s", log_exc)

        record = TrajectoryRecord(
            trajectory_id=episode_id,
            instance_id=gym_task.instance_id,
            task_id=f"swegym::{gym_task.instance_id}",
            repo=gym_task.repo,
            teacher_model=llm_model,
            rollout_index=rollout_index,
            resolved=resolved,
            reward=reward,
            reward_source=reward_source,
            turns=turns,
            wall_s=wall_s,
            answer_called=answer_called,
            tool_calls_count=len(tool_calls_log),
            messages=messages_captured,
            tool_calls=tool_calls_log,
            test_outcomes=test_outcomes if isinstance(test_outcomes, dict) else {},
            metadata={
                "answer_bridged": answer_bridged,
                "episode_id": episode_id,
            },
        )
        return record

    except Exception as exc:
        wall_s = round(time.time() - t0, 2)
        _log.error(
            "rollout_error instance_id=%s rollout=%d error=%s wall_s=%.1f",
            gym_task.instance_id,
            rollout_index,
            str(exc)[:200],
            wall_s,
        )
        # Attempt to capture agent log for debugging
        try:
            artifacts = session.collect_artifacts()
            agent_log = artifacts.get("agent_log", "") if isinstance(artifacts, dict) else ""
            if isinstance(agent_log, str) and agent_log.strip():
                _log.error("agent_log tail:\n%s", agent_log[-3000:])
            else:
                _log.warning("no agent_log captured")
        except Exception as log_exc:
            _log.warning("failed to collect agent log: %s", log_exc)
        return None
    finally:
        session.close()


async def _forward_to_llm(
    client: httpx.AsyncClient,
    *,
    intercept: dict[str, Any],
    base_url: str,
    api_key: str,
    model: str,
    timeout_s: float,
    rate_limiter: TokenBucketRateLimiter,
    max_retries: int = 5,
) -> dict[str, Any]:
    """Forward intercepted request to the teacher LLM with retry + backoff."""
    body = dict(intercept.get("body") or {})
    body["model"] = model
    # Don't request logprobs for teacher collection (not needed for SFT)
    body.pop("logprobs", None)
    body.pop("top_logprobs", None)
    body.pop("stream", None)
    body.pop("stream_options", None)
    # Disable thinking mode — get direct content, not reasoning tokens
    body.setdefault("chat_template_kwargs", {"enable_thinking": False})

    for attempt in range(max_retries):
        try:
            r = await client.post(
                f"{base_url.rstrip('/')}/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json=body,
                timeout=timeout_s,
            )

            if r.status_code == 429:
                rate_limiter.report_429()
                # Wait for backoff then retry
                backoff = min(2 ** (attempt + 1), 120.0)
                _log.warning(
                    "LLM 429 (attempt %d/%d), backing off %.1fs",
                    attempt + 1,
                    max_retries,
                    backoff,
                )
                await asyncio.sleep(backoff)
                # Re-acquire token after backoff
                await rate_limiter.acquire()
                continue

            if r.status_code != 200:
                raise RuntimeError(f"LLM error {r.status_code}: {r.text[:500]}")

            rate_limiter.report_success()
            return r.json()

        except httpx.TimeoutException:
            if attempt + 1 < max_retries:
                backoff = min(2 ** (attempt + 1), 60.0)
                _log.warning(
                    "LLM timeout (attempt %d/%d), retrying in %.1fs",
                    attempt + 1,
                    max_retries,
                    backoff,
                )
                await asyncio.sleep(backoff)
                continue
            raise

    raise RuntimeError(f"LLM request failed after {max_retries} retries (429s)")


# ── Task Runner (manages N rollouts for one task) ──────────────────────────


async def _collect_task_rollouts(
    *,
    gym_task: Any,
    n_rollouts: int,
    existing_count: int,
    factory: SWESessionFactory,
    server: InterceptionServer,
    client: httpx.AsyncClient,
    llm_base_url: str,
    llm_api_key: str,
    llm_model: str,
    interception_base_url: str,
    rate_limiter: TokenBucketRateLimiter,
    max_turns: int,
    request_timeout_s: float,
    agent_timeout_s: float,
    max_retries: int,
    semaphore: asyncio.Semaphore,
    store: TrajectoryStore,
) -> list[TrajectoryRecord]:
    """Collect all N rollouts for a single task, respecting concurrency."""
    records: list[TrajectoryRecord] = []
    start_idx = existing_count

    for rollout_idx in range(start_idx, n_rollouts):
        async with semaphore:
            record = None
            for retry in range(max_retries):
                try:
                    record = await _run_one_rollout(
                        gym_task=gym_task,
                        rollout_index=rollout_idx,
                        factory=factory,
                        server=server,
                        client=client,
                        llm_base_url=llm_base_url,
                        llm_api_key=llm_api_key,
                        llm_model=llm_model,
                        interception_base_url=interception_base_url,
                        rate_limiter=rate_limiter,
                        max_turns=max_turns,
                        request_timeout_s=request_timeout_s,
                        agent_timeout_s=agent_timeout_s,
                    )
                    if record is not None:
                        break
                except Exception as exc:
                    backoff = min(2 ** (retry + 1), 60.0)
                    _log.warning(
                        "infra_retry instance_id=%s rollout=%d retry=%d/%d "
                        "error=%s backoff=%.1fs",
                        gym_task.instance_id,
                        rollout_idx,
                        retry + 1,
                        max_retries,
                        str(exc)[:150],
                        backoff,
                    )
                    await asyncio.sleep(backoff)

            if record is not None:
                store.append(record)
                store.flush()
                records.append(record)
            else:
                _log.error(
                    "rollout_abandoned instance_id=%s rollout=%d after %d retries",
                    gym_task.instance_id,
                    rollout_idx,
                    max_retries,
                )

    return records


# ── Main ───────────────────────────────────────────────────────────────────


async def _run(args: argparse.Namespace) -> int:
    llm_base_url = _must_env("SWE_LLM_BASE_URL")
    llm_api_key = _must_env("SWE_LLM_API_KEY")
    llm_model = _must_env("SWE_LLM_MODEL")
    interception_base_url = _must_env("INTERCEPTION_BASE_URL").rstrip("/")
    interception_token = _must_env("INTERCEPTION_AUTH_TOKEN")

    # ── Load tasks ────────────────────────────────────────────────
    _log.info("Loading SWE-Gym %s tasks...", args.task_variant)
    tasks = load_swegym_tasks(args.task_variant)
    _log.info("Loaded %d tasks", len(tasks))

    # Apply slicing
    tasks = tasks[args.start_task:]
    if args.max_tasks is not None:
        tasks = tasks[: args.max_tasks]
    _log.info(
        "Will process %d tasks (start=%d, max=%s)",
        len(tasks),
        args.start_task,
        args.max_tasks,
    )

    # ── Load/create trajectory store ──────────────────────────────
    hub_repo = args.hub_repo_id.strip() if args.hub_repo_id else None
    store = TrajectoryStore(
        args.output_dir,
        hub_repo_id=hub_repo or None,
        hub_upload_every=args.hub_upload_every,
    )
    _log.info("Trajectory store: %s (%d existing)", store.filepath, len(store))
    if hub_repo:
        _log.info("Hub persistence: %s (every %d)", hub_repo, args.hub_upload_every)

    # Count existing rollouts per task for resume
    existing_counts: dict[str, int] = defaultdict(int)
    for rec in store.records:
        existing_counts[rec.instance_id] += 1

    tasks_needing_work = [
        t for t in tasks if existing_counts[t.instance_id] < args.n_rollouts
    ]
    _log.info(
        "Tasks needing work: %d / %d (others already have %d rollouts)",
        len(tasks_needing_work),
        len(tasks),
        args.n_rollouts,
    )

    if not tasks_needing_work:
        _log.info("All tasks complete! Nothing to do.")
        _print_summary(store, args)
        return 0

    # ── Start InterceptionServer ──────────────────────────────────
    server = InterceptionServer(
        port=args.interception_port,
        host=args.interception_host,
        secret=interception_token,
        tool_name_allowlist={"answer"},
    )
    await server.start()
    _log.info(
        "InterceptionServer started on %s:%d (public: %s)",
        args.interception_host,
        server.port,
        interception_base_url,
    )

    # ── Create sandbox backend + session factory ──────────────────
    backend = create_sandbox_backend("hf", flavor=args.hf_flavor)
    cfg = SWEAgentConfig(
        base_url=interception_base_url,
        api_key=interception_token,
        model=llm_model,
        agent_timeout_s=args.agent_timeout_s,
    )
    factory = SWESessionFactory(
        agent="pi",
        config=cfg,
        sandbox_backend=backend,
        mode="interception_gate",
        interception_server=server,
        interception_base_url=interception_base_url,
    )

    # ── Rate limiter + concurrency ────────────────────────────────
    rate_limiter = TokenBucketRateLimiter(rate=args.rate_limit, per=60.0)
    semaphore = asyncio.Semaphore(args.max_concurrent)

    # ── Run collection ────────────────────────────────────────────
    total_start = time.time()
    completed_tasks = 0
    total_resolved = 0
    total_rollouts = 0

    try:
        async with httpx.AsyncClient() as client:
            # Process tasks with bounded concurrency.
            # We launch all tasks as coroutines but the semaphore inside
            # _collect_task_rollouts gates actual sandbox creation.
            coros = []
            for gym_task in tasks_needing_work:
                existing = existing_counts[gym_task.instance_id]
                coros.append(
                    _collect_task_rollouts(
                        gym_task=gym_task,
                        n_rollouts=args.n_rollouts,
                        existing_count=existing,
                        factory=factory,
                        server=server,
                        client=client,
                        llm_base_url=llm_base_url,
                        llm_api_key=llm_api_key,
                        llm_model=llm_model,
                        interception_base_url=interception_base_url,
                        rate_limiter=rate_limiter,
                        max_turns=args.max_turns,
                        request_timeout_s=args.request_timeout_s,
                        agent_timeout_s=args.agent_timeout_s,
                        max_retries=args.max_retries,
                        semaphore=semaphore,
                        store=store,
                    )
                )

            # Use gather to run tasks — semaphore controls actual concurrency
            results = await asyncio.gather(*coros, return_exceptions=True)

            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    _log.error(
                        "task_failed instance_id=%s error=%s",
                        tasks_needing_work[i].instance_id,
                        str(result)[:200],
                    )
                elif isinstance(result, list):
                    completed_tasks += 1
                    total_rollouts += len(result)
                    total_resolved += sum(1 for r in result if r.resolved)

                    # Periodic progress
                    if completed_tasks % 10 == 0:
                        elapsed = time.time() - total_start
                        _log.info(
                            "progress: %d/%d tasks done, %d rollouts "
                            "(%d resolved), %.0fs elapsed",
                            completed_tasks,
                            len(tasks_needing_work),
                            total_rollouts,
                            total_resolved,
                            elapsed,
                        )

    finally:
        await server.stop()

    # ── Final summary ─────────────────────────────────────────────
    total_elapsed = time.time() - total_start
    _log.info(
        "collection_complete tasks=%d rollouts=%d resolved=%d "
        "elapsed=%.1fs (%.1f min)",
        completed_tasks,
        total_rollouts,
        total_resolved,
        total_elapsed,
        total_elapsed / 60,
    )

    _print_summary(store, args)

    # ── Final hub upload ───────────────────────────────────────
    store.upload_now()

    # ── Export SFT data if requested ──────────────────────────────
    if args.export_sft:
        sft_path = Path(args.output_dir) / "sft_export.jsonl"
        count = store.export_for_sft(sft_path, resolved_only=True, max_turns=15)
        _log.info("Exported %d trajectories for SFT to %s", count, sft_path)

    return 0


def _print_summary(store: TrajectoryStore, args: argparse.Namespace) -> None:
    """Print final statistics."""
    print("\n" + "=" * 68)
    print("COLLECTION SUMMARY")
    print("=" * 68)
    print(store.summary())

    stats = store.pass_at_k_stats(k_values=[1, 4, 8])
    print(f"\n  Total tasks attempted: {stats['total_tasks']}")
    print(f"  Total trajectories: {stats['total_trajectories']}")
    print(f"  Overall resolve rate: {stats['resolve_rate']*100:.1f}%")
    for k in (1, 4, 8):
        key = f"pass@{k}"
        if key in stats:
            print(f"  {key}: {stats[key]*100:.1f}%")
    print("=" * 68)

    # Write stats to file
    stats_path = Path(args.output_dir) / "stats.json"
    # Remove per_task for cleaner top-level file
    stats_summary = {k: v for k, v in stats.items() if k != "per_task"}
    stats_path.write_text(json.dumps(stats_summary, indent=2))
    _log.info("Stats written to %s", stats_path)


def main() -> None:
    args = _build_parser().parse_args()
    raise SystemExit(asyncio.run(_run(args)))


if __name__ == "__main__":
    main()
