"""Verdict observer: reads assets, the runner's output and the sealed trace."""

import json
import os

env = os.environ.get
oracle = json.load(
    open(os.path.join(env("OPENENVD_ASSETS", "/assets"), "oracle", "oracle.json"))
)
runner = json.load(
    open(os.path.join(env("OPENENVD_INPUTS", "/inputs"), "runner", "report.json"))
)
trace = os.path.join(env("OPENENVD_SOCKETS", "/run/openenvd"), "trace.jsonl")
records = [json.loads(line) for line in open(trace) if line.strip()]
steps = [
    r for r in records if r["kind"] == "ws.in" and '"step"' in json.dumps(r["data"])
]
verdict = {
    "score": 1.0 if len(steps) >= oracle["expected_messages"] else 0.0,
    "steps_seen": len(steps),
    "runner_saw_assets": runner["assets_visible"],
    "sealed": any(r["kind"] == "seal" for r in records),
}
with open(os.path.join(env("OPENENVD_OUT", "/out"), "verdict.json"), "w") as f:
    json.dump(verdict, f)
