# SPDX-License-Identifier: BSD-3-Clause

"""Live test of a whole unit. Runs inside the privileged integration container.

Build and run with `tests/core/test_openenvd/integration/run.sh`.
"""

import json
import os
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import httpx
import pytest
from websockets.sync.client import connect

pytestmark = pytest.mark.skipif(
    not Path("/opt/unit/openenv.yaml").exists(), reason="needs the integration image"
)

TIER = os.environ.get("OPENENVD_IT_TIER", "containers")
MANIFEST = (
    "/opt/unit/openenv.yaml"
    if TIER == "containers"
    else "/opt/unit/openenv.landlock.yaml"
)

BASE = "http://127.0.0.1:8100"
ORCH = "orchestrator-token-for-tests"
OBS = "observer-token-for-tests"
KEY = "trace-key-for-tests"
H = {"Authorization": f"Bearer {ORCH}"}
STATE = Path("/var/lib/openenvd")


MODEL_KEY = "real-model-key-never-in-a-zone"


class _StubModel(BaseHTTPRequestHandler):
    """An Anthropic-style upstream that insists on the real key."""

    def do_POST(self):
        self.rfile.read(int(self.headers.get("content-length", 0)))
        if self.headers.get("x-api-key") != MODEL_KEY:
            self.send_response(403)
            self.end_headers()
            return
        body = json.dumps(
            {
                "type": "message",
                "role": "assistant",
                "content": [
                    {"type": "thinking", "thinking": "secret plan"},
                    {"type": "text", "text": "hello"},
                ],
            }
        ).encode()
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


_MODEL = None


def _start(tmp: Path, *extra: str) -> tuple[subprocess.Popen, list[str]]:
    global _MODEL
    if _MODEL is None:
        _MODEL = ThreadingHTTPServer(("127.0.0.1", 9999), _StubModel)
        threading.Thread(target=_MODEL.serve_forever, daemon=True).start()
    model_key = tmp / "model-key"
    model_key.write_text(MODEL_KEY)
    extra = (
        "--model-url",
        "http://127.0.0.1:9999",
        "--model-key-file",
        str(model_key),
        *extra,
    )
    tokens = tmp / "tokens.json"
    tokens.write_text(json.dumps({"orchestrator": ORCH, "observer": OBS}))
    key = tmp / "key"
    key.write_text(KEY)
    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "openenv.core.openenvd",
            "--manifest",
            MANIFEST,
            "--assets",
            "/opt/unit",
            "--state",
            str(STATE),
            "--tokens",
            str(tokens),
            "--trace-key-file",
            str(key),
            *extra,
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    lines: list[str] = []
    threading.Thread(
        target=lambda: [lines.append(line) for line in proc.stdout], daemon=True
    ).start()
    return proc, lines


@pytest.fixture(scope="module")
def daemon(tmp_path_factory):
    proc, lines = _start(tmp_path_factory.mktemp("d"))
    deadline = time.time() + 90
    while not any("openenvd ready" in line for line in lines):
        if proc.poll() is not None or time.time() > deadline:
            pytest.fail("openenvd did not start:\n" + "".join(lines))
        time.sleep(0.2)
    yield lines
    proc.terminate()
    proc.wait(30)
    sys.stdout.write("".join(lines[-80:]))


def _crun(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["crun", "--root", str(STATE / "run" / "oci"), *args],
        capture_output=True,
        text=True,
    )


def _container(suffix: str) -> str:
    listed = json.loads(_crun("list", "--format", "json").stdout or "[]")
    return next(c["id"] for c in listed if c["id"].endswith(suffix))


def _exec(suffix: str, code: str) -> str:
    result = _crun("exec", _container(suffix), "/usr/local/bin/python3", "-c", code)
    assert result.returncode == 0, result.stderr
    return result.stdout.strip()


def test_enforcement_check(tmp_path):
    """Probe containers in every zone try what the zone must not be able to do."""
    proc, lines = _start(tmp_path, "--check", "--state", str(STATE / "check"))
    code = proc.wait(300)
    time.sleep(0.2)
    start = lines.index("{\n")
    end = len(lines) - lines[::-1].index("}\n")
    report = json.loads("".join(lines[start:end]))
    assert code == 0, "".join(lines)
    g = report["guarantees"]
    if TIER == "containers":
        assert all(v == "prevented" for v in g.values()), report
    else:
        for name in ("asset_isolation", "control_plane_isolation", "egress_control"):
            assert g[name] == "prevented", report


def test_full_episode(daemon):
    info = httpx.get(f"{BASE}/info", headers=H).json()
    assert info["tier"] == TIER
    expected = "prevented" if TIER == "containers" else "not_supported"
    assert info["guarantees"]["principal_isolation"] == expected
    assert httpx.get(f"{BASE}/info").status_code == 401

    info = httpx.post(f"{BASE}/reset_unit", headers=H, timeout=120).json()
    assert info["phase"] == "ready", info
    trace = STATE / "state" / "trace" / f"{info['episode_id']}.jsonl"

    # The scripted harness runs on its own; wait for its self-report.
    deadline = time.time() + 60
    while "harness.self_report" not in trace.read_text():
        assert time.time() < deadline, "harness never reported"
        time.sleep(0.2)
    time.sleep(0.5)

    # The agent surface is the unmodified env server.
    listed = httpx.post(
        f"{BASE}/mcp",
        json={"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}},
    ).json()
    names = {t["name"] for t in listed["result"]["tools"]}
    assert "echo_message" in names

    # Simulation controls are not reachable without the orchestrator token.
    assert httpx.post(f"{BASE}/reset", json={}).status_code == 401

    with connect(f"ws://127.0.0.1:8100/ws?token={ORCH}") as ws:
        ws.send(json.dumps({"type": "reset", "data": {}}))
        json.loads(ws.recv(timeout=30))
        for message in ("one", "two"):
            ws.send(
                json.dumps(
                    {
                        "type": "step",
                        "data": {
                            "type": "call_tool",
                            "tool_name": "echo_message",
                            "arguments": {"message": message},
                        },
                    }
                )
            )
            json.loads(ws.recv(timeout=30))
    assert httpx.get(f"{BASE}/info", headers=H).json()["phase"] == "running"

    if TIER == "containers":
        # Inside the env container: control-plane state and assets are masked,
        # nothing outside is reachable, and the hidden service answers only via its relay.
        out = _exec(
            "agent-env",
            "import os, socket, json, urllib.request\n"
            "r = {}\n"
            "def ls(p):\n"
            " try:\n  return os.listdir(p)\n"
            " except (FileNotFoundError, PermissionError):\n  return []\n"
            "r['state'] = ls('/var/lib/openenvd')\n"
            "r['assets'] = ls('/opt/unit/assets')\n"
            "r['pids'] = len([p for p in os.listdir('/proc') if p.isdigit()])\n"
            "s = socket.socket(); s.settimeout(2)\n"
            "try:\n s.connect(('1.1.1.1', 53)); r['egress'] = 'open'\n"
            "except OSError as e:\n r['egress'] = type(e).__name__\n"
            "r['svc'] = urllib.request.urlopen('http://airbnb.sim/', timeout=5).status\n"
            "open('/workspace/agent-note.txt', 'w').write('left by the agent')\n"
            "print(json.dumps(r))",
        )
        seen = json.loads(out)
        assert seen["state"] == [] and seen["assets"] == []
        assert seen["egress"] != "open"
        assert seen["pids"] < 10
        assert seen["svc"] == 200

    # Freeze and thaw the agent zone.
    assert (
        httpx.post(f"{BASE}/inspect", headers=H, timeout=60).json()["phase"] == "frozen"
    )
    assert httpx.post(f"{BASE}/resume", headers=H).json()["phase"] == "running"

    # The observer stream replays the trace so far.
    with connect(f"ws://127.0.0.1:8100/observe?token={OBS}") as ws:
        kinds = {json.loads(ws.recv(timeout=10))["kind"] for _ in range(5)}
    assert "phase" in kinds

    info = httpx.post(f"{BASE}/end_episode", headers=H, timeout=300).json()
    assert info["phase"] == "closed", info
    assert info["verdict"] == "graded" and info["seal"]

    from openenv.core.openenvd.trace import verify_trace

    result = verify_trace(trace, KEY.encode(), expected_head=info["seal"])
    assert result.ok, result.error
    records = [json.loads(line) for line in trace.read_text().splitlines()]
    kinds = [r["kind"] for r in records]
    assert "mcp.call" in kinds and "ws.in" in kinds and "kernel.exec" in kinds
    if TIER == "containers":
        assert "service.request" in kinds
        files = [r["data"] for r in records if r["kind"] == "kernel.file"]
        assert {"path": "agent-note.txt", "change": "written"}.items() <= files[
            0
        ].items()

    # The harness: the model proxy saw the thinking it later dropped, the real
    # key never entered the zone or the trace, and the mismatch was caught.
    assert MODEL_KEY not in trace.read_text()
    responses = [r["data"] for r in records if r["kind"] == "model.response"]
    assert responses[0]["thinking"] == "secret plan"
    probe = next(
        r["data"]
        for r in records
        if r["kind"] == "harness.self_report" and r["data"].get("kind") == "probe"
    )
    assert probe["tools"] and probe["other_key"] == 401
    assert probe["egress"] != "open" and probe["key_in_env"] is False
    assert probe["asset_readable"] is False and probe["state_readable"] is False
    mismatches = [
        r["data"]["mismatch"] for r in records if r["kind"] == "trace.mismatch"
    ]
    assert mismatches == ["thinking_dropped"]
    assert info["flags"] == ["trace_mismatch"]
    grading = [
        json.loads(line)
        for line in trace.with_suffix(".grading.jsonl").read_text().splitlines()
    ]
    results = {
        r["data"]["observer"]: r["data"]["result"]
        for r in grading
        if r["kind"] == "observer.result"
    }
    assert results["runner"]["assets_visible"] is False
    assert results["grader"]["score"] == 1.0
    assert results["grader"]["runner_saw_assets"] is False
    assert results["grader"]["sealed"] is True

    # The agent surface is closed once the episode is over.
    assert httpx.post(f"{BASE}/mcp", json={}).status_code == 503
