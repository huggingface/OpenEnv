# SPDX-License-Identifier: BSD-3-Clause

import asyncio
import json
import stat
import sys
import threading
from pathlib import Path

import pytest
from openenv.core.openenvd.trace import (
    canonical_json,
    cross_check,
    GENESIS,
    set_append_only,
    TraceRecord,
    TraceRecorder,
    TraceSealedError,
    TraceSubscriberOverflow,
    verify_trace,
)

KEY = b"k" * 32


def _clear_append_only(path: Path) -> None:
    import fcntl
    import struct

    with open(path, "rb") as f:
        buf = bytearray(8)
        fcntl.ioctl(f.fileno(), 0x80086601, buf, True)
        flags = struct.unpack_from("i", buf)[0] & ~0x20
        struct.pack_into("i", buf, 0, flags)
        fcntl.ioctl(f.fileno(), 0x40086602, buf, False)


@pytest.fixture
def path(tmp_path):
    return tmp_path / "trace.jsonl"


@pytest.fixture
def recorder(path):
    rec = TraceRecorder(path, KEY, append_only=False)
    yield rec
    rec.close()


def _fill(rec: TraceRecorder, n: int = 4) -> None:
    for i in range(n):
        rec.append("mcp.call", "env_relay", {"tool": "ls", "i": i}, zone="agent")


def _lines(path: Path) -> list[str]:
    return path.read_text().splitlines(keepends=True)


def _sealed_trace(rec: TraceRecorder, path: Path) -> list[str]:
    _fill(rec)
    rec.seal("done")
    rec.close()
    return _lines(path)


class TestRecorder:
    def test_chain_links_and_hashes(self, recorder, path):
        _fill(recorder, 3)
        records = [TraceRecord.model_validate_json(x) for x in _lines(path)]
        assert [r.seq for r in records] == [0, 1, 2]
        assert records[0].prev == GENESIS
        assert records[1].prev == records[0].hash
        assert all(r.computed_hash() == r.hash for r in records)
        assert recorder.head == records[-1].hash
        assert recorder.count == 3

    def test_file_mode_is_private(self, recorder, path):
        assert stat.S_IMODE(path.stat().st_mode) == 0o600
        assert stat.S_IMODE(recorder.grading_path.stat().st_mode) == 0o600

    def test_non_json_data_is_normalized_so_disk_matches_memory(self, recorder, path):
        record = recorder.append("kernel.file", "kernel", {"path": Path("/x")})
        assert record.data == {"path": "/x"}
        on_disk = TraceRecord.model_validate_json(_lines(path)[0])
        assert on_disk == record

    def test_refuses_existing_trace(self, path):
        path.write_text("{}\n")
        with pytest.raises(FileExistsError):
            TraceRecorder(path, KEY, append_only=False)

    def test_concurrent_appends_keep_a_valid_chain(self, recorder, path):
        def work():
            _fill(recorder, 50)

        threads = [threading.Thread(target=work) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        result = verify_trace(path, KEY)
        assert result.ok, result.error
        assert result.count == 400

    def test_append_only_flag_reported(self, path):
        rec = TraceRecorder(path, KEY)
        try:
            assert isinstance(rec.append_only_enforced, bool)
            if sys.platform != "linux":
                assert rec.append_only_enforced is False
        finally:
            rec.close()
            if rec.append_only_enforced:
                _clear_append_only(path)
                _clear_append_only(rec.grading_path)

    def test_set_append_only_never_raises_when_not_permitted(self, tmp_path):
        f = tmp_path / "f"
        f.write_text("x")
        result = set_append_only(f)
        if result:
            _clear_append_only(f)
        assert isinstance(result, bool)


class TestSeal:
    def test_seal_is_idempotent(self, recorder):
        _fill(recorder)
        first = recorder.seal("done")
        assert recorder.seal("timeout") == first
        assert first.count == 4
        assert first.head == recorder.head
        assert recorder.sealed == first

    def test_agent_side_sources_cannot_append_after_seal(self, recorder):
        _fill(recorder)
        recorder.seal("done")
        for source in ("env_relay", "model_proxy", "harness", "kernel"):
            with pytest.raises(TraceSealedError):
                recorder.append("mcp.call", source, {})
        assert recorder.count == 4

    def test_grading_segment_chains_from_seal_head(self, recorder, path):
        _fill(recorder)
        seal = recorder.seal("done")
        verdict = recorder.append("grader.verdict", "observer", {"score": 1.0})
        phase = recorder.append("phase", "control", {"to": "closed"})
        assert verdict.segment == "grading"
        assert verdict.prev == seal.head
        assert verdict.seq == seal.count + 1
        assert phase.prev == verdict.hash
        assert len(_lines(path)) == 5  # four records plus the seal record
        assert TraceRecord.model_validate_json(_lines(path)[-1]).hash == seal.head
        assert len(_lines(recorder.grading_path)) == 2
        recorder.close()
        result = verify_trace(path, KEY, require_seal=True)
        assert result.ok, result.error
        assert (result.count, result.head, result.sealed) == (4, seal.head, True)

    def test_seal_kind_is_reserved(self, recorder):
        with pytest.raises(ValueError):
            recorder.append("seal", "control", {})


class TestVerify:
    def test_wrong_key_fails_the_seal(self, recorder, path):
        _sealed_trace(recorder, path)
        result = verify_trace(path, b"other")
        assert not result.ok
        assert "seq 4" in result.error

    def test_edited_line(self, recorder, path):
        lines = _sealed_trace(recorder, path)
        lines[2] = lines[2].replace('"i":2', '"i":99')
        path.write_text("".join(lines))
        result = verify_trace(path, KEY)
        assert not result.ok
        assert "seq 2" in result.error and "edited" in result.error

    def test_edited_line_with_recomputed_hashes_fails_the_seal(self, recorder, path):
        lines = _sealed_trace(recorder, path)
        prev = GENESIS
        forged = []
        for line in lines[:-1]:
            fields = json.loads(line)
            fields["data"]["tool"] = "rm"
            fields["prev"] = prev
            fields.pop("hash")
            fields["hash"] = TraceRecord(**fields, hash="").computed_hash()
            prev = fields["hash"]
            forged.append(canonical_json(fields) + "\n")
        seal = json.loads(lines[-1])
        seal["prev"] = prev
        seal.pop("hash")
        seal["hash"] = TraceRecord(**seal, hash="").computed_hash()
        path.write_text("".join(forged) + canonical_json(seal) + "\n")
        result = verify_trace(path, KEY)
        assert not result.ok
        assert "seal does not match" in result.error

    def test_deleted_line(self, recorder, path):
        lines = _sealed_trace(recorder, path)
        del lines[1]
        path.write_text("".join(lines))
        result = verify_trace(path, KEY)
        assert not result.ok
        assert "expected seq 1" in result.error

    def test_swapped_lines(self, recorder, path):
        lines = _sealed_trace(recorder, path)
        lines[1], lines[2] = lines[2], lines[1]
        path.write_text("".join(lines))
        result = verify_trace(path, KEY)
        assert not result.ok
        assert "seq 1" in result.error

    def test_forged_line_appended_after_seal(self, recorder, path):
        lines = _sealed_trace(recorder, path)
        last = TraceRecord.model_validate_json(lines[-1])
        fields = {
            "seq": last.seq + 1,
            "ts": last.ts,
            "segment": "episode",
            "kind": "mcp.call",
            "source": "env_relay",
            "zone": None,
            "container": None,
            "data": {},
            "prev": last.hash,
        }
        fields["hash"] = TraceRecord(**fields, hash="").computed_hash()
        path.write_text("".join(lines) + canonical_json(fields) + "\n")
        result = verify_trace(path, KEY)
        assert not result.ok
        assert "after the seal" in result.error

    def test_truncated_mid_line(self, recorder, path):
        raw = "".join(_sealed_trace(recorder, path))
        path.write_text(raw[:-20])
        result = verify_trace(path, KEY)
        assert not result.ok
        assert "truncated" in result.error

    def test_dropping_the_seal_is_caught_by_require_seal(self, recorder, path):
        lines = _sealed_trace(recorder, path)
        path.write_text("".join(lines[:2]))
        assert verify_trace(path, KEY).ok
        result = verify_trace(path, KEY, require_seal=True)
        assert not result.ok and "not sealed" in result.error

    def test_whole_lines_cut_from_the_end_fail_expected_head(self, recorder, path):
        _fill(recorder)
        seal = recorder.seal("done")
        recorder.close()
        assert verify_trace(path, KEY, expected_head=seal.head).ok
        path.write_text("".join(_lines(path)[:2]))
        result = verify_trace(path, KEY, expected_head=seal.head)
        assert not result.ok
        assert "seq 1" in result.error

    def test_dropping_the_seal_with_grading_records_fails(self, recorder, path):
        _fill(recorder)
        recorder.seal("done")
        recorder.append("grader.verdict", "observer", {"score": 0})
        recorder.close()
        path.write_text("".join(_lines(path)[:-1]))
        result = verify_trace(path, KEY)
        assert not result.ok and "not sealed" in result.error

    def test_edited_grading_record(self, recorder, path):
        _fill(recorder)
        recorder.seal("done")
        recorder.append("grader.verdict", "observer", {"score": 0})
        recorder.close()
        g = recorder.grading_path
        g.write_text(g.read_text().replace('"score":0', '"score":1'))
        result = verify_trace(path, KEY)
        assert not result.ok
        assert "seq 5" in result.error

    def test_unsealed_trace_verifies(self, recorder, path):
        _fill(recorder)
        result = verify_trace(path, KEY)
        assert result.ok and not result.sealed
        assert (result.count, result.head) == (4, recorder.head)


class TestSubscribe:
    async def test_late_subscriber_replays_then_follows_a_thread(self, recorder):
        _fill(recorder, 3)
        sub = recorder.subscribe()

        def writer():
            _fill(recorder, 3)
            recorder.seal("done")
            recorder.append("grader.verdict", "observer", {})
            recorder.close()

        thread = threading.Thread(target=writer)
        thread.start()
        seen = [r async for r in sub]
        thread.join()
        assert [r.seq for r in seen] == list(range(8))
        assert seen[6].kind == "seal"
        assert seen[7].segment == "grading"

    async def test_live_delivery_without_close(self, recorder):
        sub = recorder.subscribe()
        threading.Thread(
            target=lambda: recorder.append("phase", "control", {"to": "running"})
        ).start()
        record = await asyncio.wait_for(sub.__anext__(), timeout=2)
        assert record.data == {"to": "running"}
        sub.close()

    async def test_slow_subscriber_is_closed_not_blocking(self, recorder):
        sub = recorder.subscribe(maxsize=5)
        _fill(recorder, 20)  # must not block the recorder
        assert recorder.count == 20
        with pytest.raises(TraceSubscriberOverflow):
            async for _ in sub:
                pass

    async def test_replay_is_not_counted_against_maxsize(self, recorder):
        _fill(recorder, 20)
        sub = recorder.subscribe(maxsize=5)
        recorder.close()
        assert len([r async for r in sub]) == 20


class TestCrossCheck:
    def _turn(self, rid, text, thinking=None):
        return TraceRecord(
            seq=0,
            ts=0.0,
            segment="episode",
            kind="model.response",
            source="model_proxy",
            data={"request_id": rid, "text": text, "thinking": thinking},
            prev=GENESIS,
            hash="",
        )

    def test_honest_report_matches(self):
        records = [self._turn("a", "hi", "hmm"), self._turn("b", "ok")]
        reports = [
            {"request_id": "a", "text": " hi\n", "thinking": "hmm"},
            {"request_id": "b", "text": "ok", "thinking": None},
        ]
        assert cross_check(records, reports) == []

    def test_each_kind_of_mismatch(self):
        records = [
            self._turn("dropped", "x"),
            self._turn("edited", "real"),
            self._turn("thought", "y", "secret plan"),
        ]
        reports = [
            {"request_id": "edited", "text": "cleaned up", "thinking": None},
            {"request_id": "thought", "text": "y", "thinking": ""},
            {"request_id": "ghost", "text": "never sent", "thinking": None},
        ]
        found = {(m.kind, m.request_id) for m in cross_check(records, reports)}
        assert found == {
            ("dropped", "dropped"),
            ("edited", "edited"),
            ("thinking_dropped", "thought"),
            ("fabricated", "ghost"),
        }

    def test_edited_and_thinking_dropped_on_same_turn(self):
        records = [self._turn("a", "real", "plan")]
        reports = [{"request_id": "a", "text": "fake", "thinking": None}]
        kinds = sorted(m.kind for m in cross_check(records, reports))
        assert kinds == ["edited", "thinking_dropped"]

    def test_ignores_non_model_records_and_duplicates(self):
        other = self._turn("a", "x").model_copy(update={"kind": "mcp.result"})
        records = [other, self._turn("b", "y"), self._turn("b", "y")]
        reports = [{"request_id": "b", "text": "y"}, {"request_id": "b", "text": "z"}]
        assert cross_check(records, reports) == []
