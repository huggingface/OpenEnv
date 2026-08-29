from __future__ import annotations

import json
import os
import socket
import subprocess
import time
import uuid
from dataclasses import dataclass
from typing import Any, List, Optional

import requests
try:
    from openai import OpenAI
except Exception:  # pragma: no cover - optional at runtime
    OpenAI = None  # type: ignore


BENCHMARK = "email_triage_env"
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "meta-llama/Meta-Llama-3-8B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
API_KEY = os.getenv("API_KEY") or GROQ_API_KEY or HF_TOKEN or OPENAI_API_KEY
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")

IMAGE_NAME = LOCAL_IMAGE_NAME or "email-triage-env-openenv:latest"
TEMPERATURE = 0.0
MAX_TOKENS = 200
PORT = 8012
CONTAINER_NAME = "email-triage-inference-run"

TASKS = [("easy", 11), ("medium", 22), ("hard", 33)]

SYSTEM_PROMPT = (
    "You are an email triage assistant. Return only compact JSON with keys "
    "category, priority, should_escalate. category must be one of "
    "billing/support/spam/urgent/marketing/other; priority must be int 1-5; "
    "should_escalate must be true or false."
)


@dataclass
class ParsedAction:
    category: str
    priority: int
    should_escalate: bool


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)


def _strip_code_fences(text: str) -> str:
    out = text.strip()
    if out.startswith("```"):
        out = out.strip("`")
        if out.startswith("json"):
            out = out[4:]
    return out.strip()


def _heuristic_action(subject: str, body: str) -> ParsedAction:
    msg = f"{subject} {body}".lower()
    if any(k in msg for k in ["outage", "incident", "critical", "urgent", "production"]):
        return ParsedAction("urgent", 5, True)
    if any(k in msg for k in ["prize", "click", "offer", "winner", "reward"]):
        return ParsedAction("spam", 1, False)
    if any(k in msg for k in ["invoice", "billing", "payment", "refund", "charge"]):
        return ParsedAction("billing", 3, False)
    if any(k in msg for k in ["newsletter", "campaign", "promo", "partnership"]):
        return ParsedAction("marketing", 2, False)
    if any(k in msg for k in ["support", "error", "issue", "login", "bug"]):
        return ParsedAction("support", 3, False)
    return ParsedAction("other", 2, False)


def _parse_model_action(text: str, subject: str, body: str) -> ParsedAction:
    cleaned = _strip_code_fences(text)
    try:
        payload = json.loads(cleaned)
        category = str(payload.get("category", "other")).lower().strip()
        if category not in {"billing", "support", "spam", "urgent", "marketing", "other"}:
            category = "other"
        priority = int(payload.get("priority", 2))
        priority = max(1, min(5, priority))
        should_escalate = bool(payload.get("should_escalate", False))
        return ParsedAction(category, priority, should_escalate)
    except Exception:
        return _heuristic_action(subject, body)


def _build_openai_client() -> Optional[OpenAI]:
    if not API_KEY or OpenAI is None:
        return None
    return OpenAI(base_url=API_BASE_URL, api_key=API_KEY)


def _query_model(
    client: Optional[OpenAI],
    subject: str,
    body_snippet: str,
    sender_domain: str,
    task: str,
) -> ParsedAction:
    if client is None:
        return _heuristic_action(subject, body_snippet)

    user_prompt = (
        f"Task={task}. Sender domain={sender_domain}. Subject={subject}. "
        f"Body snippet={body_snippet}. Return JSON only."
    )

    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
        )
        content = (completion.choices[0].message.content or "").strip()
        return _parse_model_action(content, subject, body_snippet)
    except Exception:
        return _heuristic_action(subject, body_snippet)


def _docker_cleanup(container_name: str) -> None:
    try:
        subprocess.run(
            ["docker", "rm", "-f", container_name],
            capture_output=True,
            text=True,
            check=False,
        )
    except Exception:
        # Cleanup failures should never break score emission.
        pass


def _docker_available() -> bool:
    try:
        subprocess.run(["docker", "version"], check=True, capture_output=True, text=True)
        return True
    except Exception:
        return False


def _pick_port(preferred: int) -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        if sock.connect_ex(("127.0.0.1", preferred)) != 0:
            return preferred

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _candidate_images() -> List[str]:
    candidates = [
        IMAGE_NAME,
        "test:latest",
        "test",
        "email-triage-env-openenv:latest",
        "email-triage-env-opening:latest",
        "email-triage-env-openenv",
        "email-triage-env-opening",
    ]
    cwd_name = os.path.basename(os.getcwd()).replace("_", "-")
    if cwd_name:
        candidates.append(f"{cwd_name}:latest")

    deduped: List[str] = []
    seen = set()
    for item in candidates:
        if item and item not in seen:
            deduped.append(item)
            seen.add(item)
    return deduped


def _image_missing(stderr_text: str) -> bool:
    text = stderr_text.lower()
    return (
        "pull access denied" in text
        or "unable to find image" in text
        or "no such image" in text
    )


def _build_local_image(image_name: str) -> None:
    dockerfiles = [
        "Dockerfile",
        "server/Dockerfile",
        os.path.join("envs", "email_triage_env", "server", "Dockerfile"),
    ]
    dockerfile = next((path for path in dockerfiles if os.path.exists(path)), None)
    if not dockerfile:
        raise RuntimeError("Dockerfile_not_found_for_email_triage_env")

    build_res = subprocess.run(
        ["docker", "build", "-t", image_name, "-f", dockerfile, "."],
        capture_output=True,
        text=True,
    )
    if build_res.returncode != 0:
        msg = (build_res.stderr or build_res.stdout or "docker_build_failed").strip()
        raise RuntimeError(f"docker_build_failed:{msg}")


def _start_container(port: int, container_name: str) -> str:
    _docker_cleanup(container_name)
    errors: List[str] = []

    def _run_with_image(image_name: str) -> subprocess.CompletedProcess[str]:
        try:
            return subprocess.run(
                [
                    "docker",
                    "run",
                    "-d",
                    "--name",
                    container_name,
                    "-p",
                    f"{port}:8000",
                    image_name,
                ],
                capture_output=True,
                text=True,
                check=False,
            )
        except Exception as exc:
            return subprocess.CompletedProcess(
                args=["docker", "run", image_name],
                returncode=1,
                stdout="",
                stderr=str(exc),
            )

    candidates = _candidate_images()
    for candidate in candidates:
        run_res = _run_with_image(candidate)
        if run_res.returncode == 0:
            return candidate
        err = (run_res.stderr or run_res.stdout or "docker_run_failed").strip()
        errors.append(f"{candidate} -> {err}")

    build_target = candidates[0] if candidates else IMAGE_NAME
    try:
        _build_local_image(build_target)
        run_res = _run_with_image(build_target)
        if run_res.returncode == 0:
            return build_target
        err = (run_res.stderr or run_res.stdout or "docker_run_failed_after_build").strip()
        errors.append(f"{build_target} -> {err}")
    except Exception as exc:
        errors.append(str(exc))

    concise = " | ".join(errors[-3:]) if errors else "docker_run_failed"
    raise RuntimeError(f"container_start_failed:{concise}")


def _wait_for_health(base_url: str, timeout_s: float = 45.0) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            r = requests.get(f"{base_url}/health", timeout=2)
            if r.status_code == 200:
                return
        except requests.RequestException:
            pass
        time.sleep(0.5)
    raise RuntimeError("Environment did not become healthy in time")


def _run_task(base_url: str, client: Optional[OpenAI], task_name: str, seed: int) -> float:
    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

    reset_payload = {"difficulty": task_name, "seed": seed}
    reset_result = requests.post(f"{base_url}/reset", json=reset_payload, timeout=15).json()
    obs = reset_result["observation"]

    parsed = _query_model(client, obs["subject"], obs["body_snippet"], obs["sender_domain"], task_name)
    action = {
        "action": {
            "category": parsed.category,
            "priority": parsed.priority,
            "should_escalate": parsed.should_escalate,
        }
    }

    rewards: List[float] = []
    error: Optional[str] = None
    try:
        step_result = requests.post(f"{base_url}/step", json=action, timeout=15).json()
        reward = float(step_result.get("reward") or 0.0)
        done = bool(step_result.get("done", False))
    except Exception as exc:
        reward = 0.0
        done = True
        error = str(exc).replace(" ", "_")

    rewards.append(reward)
    action_repr = f"{parsed.category}|{parsed.priority}|{str(parsed.should_escalate).lower()}"
    log_step(step=1, action=action_repr, reward=reward, done=done, error=error)

    score = sum(rewards)
    log_end(success=score > 0.0, steps=len(rewards), score=score, rewards=rewards)
    return score


def main() -> None:
    client = _build_openai_client()
    runtime_port = _pick_port(PORT)
    runtime_container_name = f"{CONTAINER_NAME}-{uuid.uuid4().hex[:8]}"
    base_url = f"http://127.0.0.1:{runtime_port}"

    scores: List[float] = []
    try:
        if not _docker_available():
            raise RuntimeError("docker_not_available")
        _start_container(runtime_port, runtime_container_name)
        _wait_for_health(base_url)
        for task_name, seed in TASKS:
            scores.append(_run_task(base_url, client, task_name, seed))
    except Exception as exc:
        error_str = str(exc).replace(" ", "_")
        log_step(step=0, action="startup", reward=0.0, done=True, error=error_str)
    finally:
        _docker_cleanup(runtime_container_name)

    overall = sum(scores) / len(scores) if scores else 0.0
    print(f"FINAL_AVG_SCORE={overall:.3f}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        error_str = str(exc).replace(" ", "_")
        log_step(step=0, action="startup", reward=0.0, done=True, error=error_str)
        print("FINAL_AVG_SCORE=0.000", flush=True)
