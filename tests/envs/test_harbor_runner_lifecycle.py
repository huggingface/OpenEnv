# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""`openenv harbor rollout` must not strand its capture proxy when setup fails.

`run_batch` starts the capture server — which binds a port in a background thread — and only then
builds the tunnel the sandbox reaches it through. Anything that throws in between used to escape with
that thread still running, so the port stayed held and the *next* invocation died on a port conflict
that named nothing about the actual cause. Teardown had the mirror of the same gap: `forwarder.stop()`
ran before `capture.stop()` without a guard, so a tunnel that failed to shut down took the port with it.

These tests stub every external dependency; no engine, no sandbox and no network are involved.
"""

from __future__ import annotations

import pytest

runner = pytest.importorskip("openenv.harbor.runner")


class FakeCaps:
    llm = {"model": "m", "capture_level": "tokens"}
    available_sandboxes = ("e2b",)
    sandboxes: tuple = ()


def wire(monkeypatch, tmp_path, forwarder):
    """Replace startup, the capture server and the dataset with local stubs."""
    stopped: list[str] = []

    monkeypatch.setattr(runner, "resolve_task_dirs", lambda _d: [tmp_path])
    monkeypatch.setattr(
        "openenv.harbor.startup.prepare", lambda **_k: FakeCaps(), raising=False
    )

    class FakeCapture:
        registry = None
        inference = None

        def __init__(self, **kwargs):
            self.port = kwargs.get("port", 8100)

        def start(self):
            pass

        def stop(self):
            stopped.append("capture")

    monkeypatch.setattr(runner, "CaptureServer", FakeCapture)
    monkeypatch.setattr(
        "openenv.core.harness.capture.forwarding.make_forwarder",
        lambda _kind: forwarder(stopped),
        raising=False,
    )
    return stopped


def test_a_forwarder_that_cannot_start_releases_the_port(monkeypatch, tmp_path):
    class Exploding:
        name = "cloudflare"

        def __init__(self, _stopped):
            pass

        def start(self, _port):
            raise RuntimeError("cloudflared is not installed")

    stopped = wire(monkeypatch, tmp_path, Exploding)

    with pytest.raises(RuntimeError, match="cloudflared"):
        __import__("asyncio").run(
            runner.run_batch(llm_url="http://x/v1", dataset="d", task_indices=[0])
        )

    assert stopped == ["capture"], (
        "the capture server kept its port after a failed forwarder"
    )


def test_a_forwarder_that_cannot_stop_still_lets_the_port_go(monkeypatch, tmp_path):
    class BadTeardown:
        name = "gradio"

        def __init__(self, stopped):
            self._stopped = stopped

        def start(self, _port):
            return "https://tunnel.invalid"

        def stop(self):
            self._stopped.append("forwarder")
            raise RuntimeError("tunnel already gone")

    stopped = wire(monkeypatch, tmp_path, BadTeardown)
    monkeypatch.setattr(
        runner, "run_rollout", None
    )  # never reached: no indices are in range

    with pytest.raises(RuntimeError, match="tunnel already gone"):
        __import__("asyncio").run(
            runner.run_batch(llm_url="http://x/v1", dataset="d", task_indices=[99])
        )

    assert stopped == ["forwarder", "capture"], (
        "capture.stop() must run even when the forwarder's teardown raises"
    )
