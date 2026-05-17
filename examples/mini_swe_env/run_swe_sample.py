#!/usr/bin/env python3
"""Run one real SWE-Gym interception rollout against an actual LLM.

This is a true end-to-end harness test:

  SWE-Gym task -> SWESessionFactory -> Pi in sandbox
  -> InterceptionServer -> real OpenAI-compatible LLM
  -> host-side answer tool grading -> session.verify()

Required env vars:
  SWE_LLM_BASE_URL   e.g. https://api.openai.com/v1
  SWE_LLM_API_KEY    bearer token
  SWE_LLM_MODEL      e.g. gpt-4o-mini

Example:
  SWE_LLM_BASE_URL=https://api.openai.com/v1 \
  SWE_LLM_API_KEY=... \
  SWE_LLM_MODEL=gpt-4o-mini \
  PYTHONPATH=src:envs python examples/mini_swe_env/run_swe_sample.py
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
from pathlib import Path

import httpx

_root = Path(__file__).resolve().parent.parent.parent
for _p in (_root / "src", _root / "envs"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from mini_swe_env.harness import SWEAgentConfig, SWESessionFactory
from mini_swe_env.task_loader_swegym import load_swegym_tasks
from openenv.core.harness.agents.interception_server import InterceptionServer
from openenv.core.harness.sandbox import create_sandbox_backend

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
_log = logging.getLogger("swe-e2e")

# Known easy SWE-Gym Lite task that has repeatedly produced reward=1.0
# in local end-to-end validation with qwen-3.6-27b.
DEFAULT_TASK_INDEX = 16  # getmoto__moto-5699


def _arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run one real SWE interception rollout")
    p.add_argument("--task-variant", default="lite", choices=["lite", "full"])
    p.add_argument(
        "--task-index",
        type=int,
        default=DEFAULT_TASK_INDEX,
        help=f"Task index in variant split (default: {DEFAULT_TASK_INDEX}, getmoto__moto-5699)",
    )
    p.add_argument("--sandbox-backend", default="docker", choices=["docker", "e2b", "hf"])
    p.add_argument("--interception-port", type=int, default=9090)
    p.add_argument("--interception-host", default="0.0.0.0")
    p.add_argument("--interception-base-url", default="")
    p.add_argument("--agent-timeout-s", type=float, default=1800.0)
    p.add_argument("--request-timeout-s", type=float, default=180.0)
    p.add_argument("--max-turns", type=int, default=50)
    p.add_argument("--assert-host-answer", action="store_true")
    return p


def _must_env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(f"Missing required env var: {name}")
    return value


async def _forward_to_llm(
    client: httpx.AsyncClient,
    *,
    intercept: dict,
    base_url: str,
    api_key: str,
    model: str,
    timeout_s: float,
) -> dict:
    body = dict(intercept.get("body") or {})
    body["model"] = model
    body["logprobs"] = True
    body["top_logprobs"] = 5
    body.pop("stream", None)
    body.pop("stream_options", None)

    r = await client.post(
        f"{base_url.rstrip('/')}/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json=body,
        timeout=timeout_s,
    )
    if r.status_code != 200:
        raise RuntimeError(f"LLM error {r.status_code}: {r.text[:500]}")
    return r.json()


async def _invoke_answer_tool(
    client: httpx.AsyncClient,
    *,
    server_port: int,
    secret: str,
    rollout_id: str,
) -> dict:
    r = await client.post(
        f"http://127.0.0.1:{server_port}/rollout/{rollout_id}/v1/tools/answer",
        headers={"Authorization": f"Bearer {secret}"},
        json={"arguments": {}},
        timeout=180.0,
    )
    if r.status_code != 200:
        raise RuntimeError(f"answer tool error {r.status_code}: {r.text[:500]}")
    return r.json()


async def _run(args: argparse.Namespace) -> int:
    llm_base_url = _must_env("SWE_LLM_BASE_URL")
    llm_api_key = _must_env("SWE_LLM_API_KEY")
    llm_model = _must_env("SWE_LLM_MODEL")

    tasks = load_swegym_tasks(args.task_variant)
    if not (0 <= args.task_index < len(tasks)):
        raise RuntimeError(
            f"task-index {args.task_index} out of range [0, {len(tasks) - 1}]"
        )
    gym_task = tasks[args.task_index]
    swe_task = gym_task.to_swe_task()

    server = InterceptionServer(port=args.interception_port, host=args.interception_host)
    await server.start()

    try:
        interception_base_url = args.interception_base_url.strip()
        if not interception_base_url:
            if args.sandbox_backend != "docker":
                raise RuntimeError(
                    "For non-docker backends, pass --interception-base-url "
                    "that sandboxes can reach (tunnel/public URL)."
                )
            interception_base_url = f"http://host.docker.internal:{server.port}"

        _log.info("Task: %s (%s)", gym_task.instance_id, gym_task.repo)
        _log.info("Sandbox image: %s", gym_task.instance_image)
        _log.info("Interception server: %s", interception_base_url)

        backend = create_sandbox_backend(args.sandbox_backend)
        cfg = SWEAgentConfig(
            base_url=interception_base_url,
            api_key=server.secret,
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

        episode_id = f"swe-e2e-{uuid.uuid4().hex[:8]}"
        t0 = time.time()
        session = factory.create(task=swe_task, episode_id=episode_id)
        _log.info("Session created in %.1fs", time.time() - t0)

        turns = 0
        answer_requested = False
        answer_bridge_invoked = False
        logprob_tokens = 0

        try:
            async with httpx.AsyncClient() as client:
                while turns < args.max_turns:
                    intercept = await session.next_request(timeout_s=args.agent_timeout_s)
                    if intercept is None:
                        _log.info("Agent exited")
                        break

                    turns += 1
                    llm_resp = await _forward_to_llm(
                        client,
                        intercept=intercept,
                        base_url=llm_base_url,
                        api_key=llm_api_key,
                        model=llm_model,
                        timeout_s=args.request_timeout_s,
                    )

                    choice0 = (llm_resp.get("choices") or [{}])[0]
                    msg = choice0.get("message") or {}
                    tool_calls = msg.get("tool_calls") or []
                    if any(
                        (tc.get("function") or {}).get("name") == "answer"
                        for tc in tool_calls
                    ):
                        answer_requested = True
                        answer_resp = await _invoke_answer_tool(
                            client,
                            server_port=server.port,
                            secret=server.secret,
                            rollout_id=str(intercept.get("rollout_id", "")),
                        )
                        answer_bridge_invoked = True
                        _log.info("host answer bridge response: %s", json.dumps(answer_resp, default=str))

                        # Replace tool-call response with plain assistant text so
                        # Pi doesn't need native /v1/tools client support.
                        llm_resp = dict(llm_resp)
                        choices = list(llm_resp.get("choices") or [])
                        if choices:
                            choice0 = dict(choices[0])
                            msg0 = dict(choice0.get("message") or {})
                            msg0.pop("tool_calls", None)
                            msg0["content"] = (
                                (msg0.get("content") or "")
                                + "\nSubmission received and graded on host."
                            ).strip()
                            choice0["message"] = msg0
                            choice0["finish_reason"] = "stop"
                            choices[0] = choice0
                            llm_resp["choices"] = choices

                    lp = (choice0.get("logprobs") or {}).get("content") or []
                    logprob_tokens += len(lp)

                    _log.info(
                        "turn=%d finish=%s tools=%s lp_tokens=%d",
                        turns,
                        choice0.get("finish_reason"),
                        [
                            (tc.get("function") or {}).get("name", "?")
                            for tc in tool_calls
                        ],
                        len(lp),
                    )

                    await session.deliver(intercept, llm_resp)

                    if answer_bridge_invoked:
                        break

            vr = session.verify(transcript=[])
            _log.info("verify reward=%s metrics=%s", vr.env_reward, json.dumps(vr.metrics, default=str))
            _log.info("turns=%d logprob_tokens=%d", turns, logprob_tokens)

            if turns == 0:
                try:
                    artifacts = session.collect_artifacts()
                    agent_log = artifacts.get("agent_log", "") if isinstance(artifacts, dict) else ""
                    if isinstance(agent_log, str) and agent_log.strip():
                        _log.warning("agent_log tail:\n%s", agent_log[-2000:])
                except Exception as exc:
                    _log.warning("Failed to collect agent log: %s", exc)

            if args.assert_host_answer:
                src = (vr.metrics or {}).get("reward_source")
                if src != "host_answer_tool":
                    raise RuntimeError(
                        "Expected reward_source=host_answer_tool, got "
                        f"{src!r}; model may not have called answer()."
                    )

            print("\n" + "=" * 68)
            print(f"Task            : {gym_task.instance_id}")
            print(f"Turns           : {turns}")
            print(f"Logprob tokens  : {logprob_tokens}")
            print(f"Answer called   : {answer_requested}")
            print(f"Answer bridged  : {answer_bridge_invoked}")
            print(f"Reward          : {vr.env_reward}")
            print(f"Reward source   : {(vr.metrics or {}).get('reward_source')}")
            print("=" * 68)
            return 0
        finally:
            session.close()
    finally:
        await server.stop()


def main() -> None:
    args = _arg_parser().parse_args()
    raise SystemExit(asyncio.run(_run(args)))


if __name__ == "__main__":
    main()
