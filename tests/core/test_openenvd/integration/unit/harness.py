"""A scripted harness: it can see only MCP_URL, MODEL_BASE_URL and its report URL.

It lists tools, makes one model call, then files a self-report that drops the
model's thinking, which the control plane must catch.
"""

import json
import os
import socket
import urllib.error
import urllib.request


def post(url, body, headers=None):
    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode(),
        headers={"content-type": "application/json", **(headers or {})},
    )
    with urllib.request.urlopen(req, timeout=10) as r:
        return r.headers, json.loads(r.read())


probe = {"kind": "probe"}
_, tools = post(
    os.environ["MCP_URL"],
    {"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}},
)
probe["tools"] = sorted(t["name"] for t in tools["result"]["tools"])

auth = {"x-api-key": os.environ["ANTHROPIC_API_KEY"], "anthropic-version": "2023-06-01"}
message = {
    "model": "stub",
    "max_tokens": 16,
    "messages": [{"role": "user", "content": "hi"}],
}
headers, reply = post(os.environ["MODEL_BASE_URL"] + "/v1/messages", message, auth)
rid = headers["x-openenvd-request-id"]

try:
    post(
        os.environ["MODEL_BASE_URL"] + "/v1/messages",
        message,
        {"x-api-key": "sk-guess"},
    )
    probe["other_key"] = "accepted"
except urllib.error.HTTPError as e:
    probe["other_key"] = e.code

s = socket.socket()
s.settimeout(2)
try:
    s.connect(("1.1.1.1", 443))
    probe["egress"] = "open"
except OSError as e:
    probe["egress"] = type(e).__name__
probe["key_in_env"] = any("real-model-key" in v for v in os.environ.values())


def readable(path):
    try:
        if os.path.isdir(path):
            return bool(os.listdir(path))
        open(path, "rb").read(1)
        return True
    except OSError:
        return False


probe["asset_readable"] = readable("/opt/unit/assets/oracle.json")
probe["state_readable"] = readable("/var/lib/openenvd")

text = "".join(b.get("text", "") for b in reply["content"] if b["type"] == "text")
post(
    os.environ["OPENENVD_REPORT_URL"],
    [probe, {"request_id": rid, "text": text, "thinking": None}],
)
