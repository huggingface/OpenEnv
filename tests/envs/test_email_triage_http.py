"""End-to-end HTTP server test for Oversight Inbox Arena."""

from __future__ import annotations

import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import requests
import uvicorn

try:
    from envs.email_triage_env.server.app import app
except ImportError:
    from email_triage_env.server.app import app


def test_http_end_to_end() -> None:
    # Start server in background
    cfg = uvicorn.Config(app, host="127.0.0.1", port=8099, log_level="error")
    server = uvicorn.Server(cfg)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    time.sleep(3)

    base = "http://127.0.0.1:8099"

    # 1. Health
    r = requests.get(f"{base}/health", timeout=5)
    print(f"Health: {r.status_code} {r.json()}")
    assert r.status_code == 200

    # 2. Reset easy (backward compat)
    r = requests.post(f"{base}/reset", json={"difficulty": "easy", "seed": 42}, timeout=5)
    data = r.json()
    obs = data["observation"]
    eid = obs["email_id"]
    subj = obs["subject"][:40]
    print(f"Easy reset OK: email_id={eid} subject={subj}")

    # 3. Step easy
    action = {"action": {"category": "billing", "priority": 3, "should_escalate": False}}
    r = requests.post(f"{base}/step", json=action, timeout=5)
    data = r.json()
    print(f"Easy step OK: reward={data.get('reward', '?')} done={data.get('done', '?')}")
    assert data.get("done") is True, "Easy mode must be done in 1 step"

    # 4. Reset hard (multi-turn)
    r = requests.post(f"{base}/reset", json={"difficulty": "hard", "seed": 42}, timeout=5)
    data = r.json()
    obs = data["observation"]
    print(f"Hard reset OK: email_id={obs['email_id']}")

    # 5. Step hard (should NOT be done)
    r = requests.post(f"{base}/step", json=action, timeout=5)
    data = r.json()
    done_val = data.get("done", "?")
    print(f"Hard step 1: reward={data.get('reward', '?')} done={done_val}")

    # 6. State endpoint
    r = requests.get(f"{base}/state", timeout=5)
    state = r.json()
    resolved = state.get("tickets_resolved", "?")
    drift = state.get("drift_count", "?")
    print(f"State OK: tickets_resolved={resolved} drift_count={drift}")

    # 7. Loop until done
    steps = 1
    while not data.get("done", True):
        r = requests.post(f"{base}/step", json=action, timeout=5)
        data = r.json()
        steps += 1

    print(f"Hard episodes completed in {steps} steps")

    server.should_exit = True
    thread.join(timeout=10)
    print()
    print("=" * 50)
    print("HTTP SERVER END-TO-END TEST PASSED")
    print("=" * 50)


if __name__ == "__main__":
    test_http_end_to_end()
