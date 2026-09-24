# SPDX-License-Identifier: BSD-3-Clause

import json
import os

import pytest
from openenv.core.harness import HarnessRolloutResult, MCPHarnessAdapter, RolloutEvent
from openenv.core.openenvd import HarnessEventSink


def test_harness_emits_through_write_only_pipe():
    read_fd, write_fd = os.pipe()
    try:
        adapter = MCPHarnessAdapter(event_sink=HarnessEventSink(write_fd))
        result = HarnessRolloutResult()
        adapter._record_event(
            result, RolloutEvent(type="model_response", payload={"text": "hello"})
        )
        assert result.events[0].type == "model_response"
        assert json.loads(os.read(read_fd, 4096)) == {
            "type": "model_response",
            "payload": {"text": "hello"},
        }
        with pytest.raises(OSError):
            os.read(write_fd, 1)
    finally:
        os.close(read_fd)
        os.close(write_fd)


def test_harness_rejects_oversized_events_without_partial_writes():
    read_fd, write_fd = os.pipe()
    try:
        with pytest.raises(ValueError):
            HarnessEventSink(write_fd)({"text": "x" * 100000})
        os.set_blocking(read_fd, False)
        with pytest.raises(BlockingIOError):
            os.read(read_fd, 4096)
    finally:
        os.close(read_fd)
        os.close(write_fd)
