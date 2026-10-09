# SPDX-License-Identifier: BSD-3-Clause

"""The unit's tamper-evident trace.

Every record carries the sha256 of the record before it, so editing, removing,
reordering or inserting a line breaks the chain. Sealing the episode signs the
chain head with an HMAC key the agent never sees, so the agent can't rebuild a
consistent chain after the fact. Records written after the seal (observer and
control-plane output during grading) go to a separate grading file whose chain
starts at the sealed head.
"""

from __future__ import annotations

import asyncio
import errno
import hashlib
import hmac
import json
import os
import struct
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Literal

from pydantic import BaseModel, ConfigDict, ValidationError

GENESIS = "0" * 64
GRADING_SOURCES = frozenset({"observer", "control"})

_FS_IOC_GETFLAGS = 0x80086601
_FS_IOC_SETFLAGS = 0x40086602
_FS_APPEND_FL = 0x20
_NOT_PERMITTED = frozenset(
    {errno.EPERM, errno.EACCES, errno.ENOTTY, errno.EOPNOTSUPP, errno.ENOTSUP}
)


class TraceSealedError(RuntimeError):
    """An episode record was appended after the trace was sealed."""


class TraceSubscriberOverflow(RuntimeError):
    """A subscriber fell too far behind the recorder and was dropped."""


def canonical_json(obj: Any) -> str:
    return json.dumps(
        obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str
    )


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _seal_hmac(key: bytes, head: str, count: int) -> str:
    return hmac.new(key, f"{head}{count}".encode(), hashlib.sha256).hexdigest()


class TraceRecord(BaseModel):
    """One hash-chained trace record.

    Attributes:
        seq (`int`):
            Position in the unit's trace. Grading records continue after the seal.
        ts (`float`):
            Wall-clock time the record was written.
        segment (`str`):
            `episode` or `grading`.
        kind (`str`):
            What happened, e.g. `mcp.call`, `model.response`, `phase` or `seal`.
        source (`str`):
            Who recorded it, e.g. `env_relay`, `model_proxy`, `kernel` or `observer`.
        zone (`str`, *optional*):
            Zone the event came from.
        container (`str`, *optional*):
            Container the event came from.
        data (`dict[str, Any]`):
            The payload, normalized to plain JSON.
        prev (`str`):
            `hash` of the previous record in the chain (64 zeros for the first).
        hash (`str`):
            sha256 of the canonical JSON of every other field.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    seq: int
    ts: float
    segment: Literal["episode", "grading"]
    kind: str
    source: str
    zone: str | None = None
    container: str | None = None
    data: dict[str, Any]
    prev: str
    hash: str

    def computed_hash(self) -> str:
        return _sha256(canonical_json(self.model_dump(mode="json", exclude={"hash"})))


class Seal(BaseModel):
    """The signed end of an episode.

    Attributes:
        head (`str`):
            Hash of the seal record itself, the final head of the episode chain.
            The grading chain starts from it.
        count (`int`):
            Number of episode records before the seal.
        hmac (`str`):
            HMAC-SHA256, under the unit's trace key, of the seal record's `prev`
            (the last episode record's hash) followed by `str(count)`. It is
            stored in the seal record, so `head` covers it.
        reason (`str`):
            Why the episode was sealed.
        ts (`float`):
            When it was sealed.
    """

    model_config = ConfigDict(frozen=True)

    head: str
    count: int
    hmac: str
    reason: str
    ts: float


def _make_record(
    *,
    seq: int,
    segment: str,
    kind: str,
    source: str,
    data: dict[str, Any],
    prev: str,
    zone: str | None,
    container: str | None,
) -> TraceRecord:
    fields = {
        "seq": seq,
        "ts": time.time(),
        "segment": segment,
        "kind": kind,
        "source": source,
        "zone": zone,
        "container": container,
        "data": json.loads(canonical_json(data)),
        "prev": prev,
    }
    return TraceRecord(**fields, hash=_sha256(canonical_json(fields)))


def set_append_only(path: Path) -> bool:
    """
    Set the filesystem append-only flag (`chattr +a`) on a file.

    Once set, even the file's owner can only append to it; truncating, rewriting,
    renaming or deleting it needs `CAP_LINUX_IMMUTABLE` in the init user namespace.

    Args:
        path (`Path`):
            File to protect.

    Returns:
        `bool`: `True` if the flag is now set, `False` if this platform, filesystem
        or process can't set it.
    """
    if not sys.platform.startswith("linux"):
        return False
    import fcntl

    try:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    except OSError as e:
        if e.errno in _NOT_PERMITTED:
            return False
        raise
    try:
        buf = bytearray(8)
        fcntl.ioctl(fd, _FS_IOC_GETFLAGS, buf, True)
        flags = struct.unpack_from("i", buf)[0]
        if flags & _FS_APPEND_FL:
            return True
        struct.pack_into("i", buf, 0, flags | _FS_APPEND_FL)
        fcntl.ioctl(fd, _FS_IOC_SETFLAGS, buf, False)
        return True
    except OSError as e:
        if e.errno in _NOT_PERMITTED or e.errno == errno.EINVAL:
            return False
        raise
    finally:
        os.close(fd)


class TraceSubscription:
    """A live, replaying stream of trace records.

    Iterate with `async for`. The stream first replays every record written
    before it was opened, then delivers new records as they're appended. It
    ends when the recorder closes, and raises [`TraceSubscriberOverflow`] if
    the consumer falls more than `maxsize` records behind.
    """

    def __init__(
        self,
        recorder: TraceRecorder,
        loop: asyncio.AbstractEventLoop,
        replay: list[TraceRecord],
        maxsize: int,
    ):
        self._recorder = recorder
        self._loop = loop
        self._buf: deque[TraceRecord] = deque(replay)
        self._limit = len(replay) + maxsize
        self._lock = threading.Lock()
        self._wake = asyncio.Event()
        self._closed = False
        self._error: BaseException | None = None

    def _push(self, record: TraceRecord) -> None:
        with self._lock:
            if self._closed:
                return
            if len(self._buf) >= self._limit:
                self._buf.clear()
                self._closed = True
                self._error = TraceSubscriberOverflow(
                    f"subscriber fell behind at seq {record.seq}; stream closed"
                )
            else:
                self._buf.append(record)
        self._notify()

    def _finish(self) -> None:
        with self._lock:
            self._closed = True
        self._notify()

    def _notify(self) -> None:
        try:
            self._loop.call_soon_threadsafe(self._wake.set)
        except RuntimeError:
            pass

    def close(self) -> None:
        """Stop receiving records. Buffered records are dropped."""
        self._recorder._unsubscribe(self)
        with self._lock:
            self._closed = True
            self._buf.clear()
        self._notify()

    def __aiter__(self) -> TraceSubscription:
        return self

    async def __anext__(self) -> TraceRecord:
        while True:
            with self._lock:
                if self._buf:
                    return self._buf.popleft()
                if self._error is not None:
                    raise self._error
                if self._closed:
                    raise StopAsyncIteration
                self._wake.clear()
            await self._wake.wait()


class TraceRecorder:
    """
    Writes the unit's hash-chained trace.

    Episode records go to `path` and grading records to
    `path.with_suffix(".grading.jsonl")`, one canonical JSON line each, flushed and
    fsynced before `append` returns.

    Args:
        path (`Path`):
            The episode trace file. Must not exist or be empty.
        key (`bytes`):
            HMAC key for the seal. Must never be reachable from the agent zone.
        append_only (`bool`, *optional*, defaults to `True`):
            Try to set the filesystem append-only flag on both files.
    """

    def __init__(self, path: Path, key: bytes, *, append_only: bool = True):
        if not key:
            raise ValueError("trace key must not be empty")
        self.path = Path(path)
        self.grading_path = self.path.with_suffix(".grading.jsonl")
        self._key = key
        self._lock = threading.Lock()
        self._records: list[TraceRecord] = []
        self._subscribers: list[TraceSubscription] = []
        self._head = GENESIS
        self._count = 0
        self._grading_head = GENESIS
        self._next_seq = 0
        self._seal: Seal | None = None
        self._closed = False

        flags = os.O_APPEND | os.O_CREAT | os.O_WRONLY | os.O_NOFOLLOW
        self._fd = os.open(self.path, flags, 0o600)
        self._grading_fd = os.open(self.grading_path, flags, 0o600)
        for fd, p in ((self._fd, self.path), (self._grading_fd, self.grading_path)):
            if os.fstat(fd).st_size:
                os.close(self._fd)
                os.close(self._grading_fd)
                raise FileExistsError(f"trace file {p} already has records")
        self._append_only_enforced = append_only and all(
            [set_append_only(self.path), set_append_only(self.grading_path)]
        )

    @property
    def append_only_enforced(self) -> bool:
        """Whether the kernel enforces append-only on both trace files."""
        return self._append_only_enforced

    @property
    def head(self) -> str:
        """Hash of the last episode-file record; after sealing, the seal record."""
        return self._head

    @property
    def count(self) -> int:
        """Number of episode records (the seal record excluded)."""
        return self._count

    @property
    def sealed(self) -> Seal | None:
        """The seal, once the episode has been sealed."""
        return self._seal

    def append(
        self,
        kind: str,
        source: str,
        data: dict[str, Any],
        *,
        zone: str | None = None,
        container: str | None = None,
    ) -> TraceRecord:
        """
        Append one record. Thread-safe.

        Args:
            kind (`str`):
                What happened, e.g. `mcp.call` or `model.response`.
            source (`str`):
                Who recorded it. After the seal only `observer` and `control` may
                append; their records go to the grading segment.
            data (`dict[str, Any]`):
                The payload. Values JSON can't encode are stored as `str`.
            zone (`str`, *optional*):
                Zone the event came from.
            container (`str`, *optional*):
                Container the event came from.

        Returns:
            [`TraceRecord`]: the record as written.
        """
        if kind == "seal":
            raise ValueError("use seal() to seal the trace")
        with self._lock:
            self._check_open()
            if self._seal is None:
                record = self._write_episode(kind, source, data, zone, container)
                self._head = record.hash
                self._count += 1
                return record
            if source not in GRADING_SOURCES:
                raise TraceSealedError(
                    f"trace sealed ({self._seal.reason}); {source!r} can't append {kind!r}"
                )
            record = _make_record(
                seq=self._next_seq,
                segment="grading",
                kind=kind,
                source=source,
                data=data,
                prev=self._grading_head,
                zone=zone,
                container=container,
            )
            self._write(self._grading_fd, record)
            self._grading_head = record.hash
            return record

    def seal(self, reason: str) -> Seal:
        """
        Seal the episode. Idempotent: later calls return the first seal.

        Args:
            reason (`str`):
                Why the episode ended, e.g. `done` or `timeout`.

        Returns:
            [`Seal`]: the signed head and count.
        """
        with self._lock:
            if self._seal is not None:
                return self._seal
            self._check_open()
            data = {
                "count": self._count,
                "hmac": _seal_hmac(self._key, self._head, self._count),
                "reason": reason,
            }
            record = self._write_episode("seal", "control", data, None, None)
            self._seal = Seal(head=record.hash, ts=record.ts, **data)
            self._head = self._grading_head = record.hash
            return self._seal

    def subscribe(self, *, maxsize: int = 10_000) -> TraceSubscription:
        """
        Open a live stream of records. Call from inside a running event loop.

        Args:
            maxsize (`int`, *optional*, defaults to `10000`):
                How many undelivered live records the subscriber may lag by
                before it is closed with [`TraceSubscriberOverflow`].

        Returns:
            [`TraceSubscription`]: replays every record so far, then follows.
        """
        loop = asyncio.get_running_loop()
        with self._lock:
            sub = TraceSubscription(self, loop, list(self._records), maxsize)
            if self._closed:
                sub._finish()
            else:
                self._subscribers.append(sub)
            return sub

    def close(self) -> None:
        """Close both files and end every subscription."""
        with self._lock:
            if self._closed:
                return
            self._closed = True
            os.close(self._fd)
            os.close(self._grading_fd)
            subs, self._subscribers = self._subscribers, []
        for sub in subs:
            sub._finish()

    def _unsubscribe(self, sub: TraceSubscription) -> None:
        with self._lock:
            if sub in self._subscribers:
                self._subscribers.remove(sub)

    def _check_open(self) -> None:
        if self._closed:
            raise RuntimeError("trace recorder is closed")

    def _write_episode(self, kind, source, data, zone, container) -> TraceRecord:
        record = _make_record(
            seq=self._next_seq,
            segment="episode",
            kind=kind,
            source=source,
            data=data,
            prev=self._head,
            zone=zone,
            container=container,
        )
        self._write(self._fd, record)
        return record

    def _write(self, fd: int, record: TraceRecord) -> None:
        line = (canonical_json(record.model_dump(mode="json")) + "\n").encode()
        view = memoryview(line)
        while view:
            view = view[os.write(fd, view) :]
        os.fsync(fd)
        self._next_seq += 1
        self._records.append(record)
        for sub in self._subscribers:
            sub._push(record)


@dataclass
class VerifyResult:
    """The outcome of [`verify_trace`].

    Attributes:
        ok (`bool`):
            Whether both files are intact.
        count (`int`):
            Episode records before the seal (or all of them, if unsealed).
        head (`str`):
            Final head of the episode file (the seal record's hash, if sealed).
        sealed (`bool`):
            Whether a valid seal was found.
        error (`str`, *optional*):
            What was wrong, naming the first bad seq or line.
    """

    ok: bool
    count: int
    head: str
    sealed: bool
    error: str | None = None


def _read_chain(
    path: Path, segment: str, first_seq: int, first_prev: str
) -> tuple[list[TraceRecord], str | None]:
    records: list[TraceRecord] = []
    raw = path.read_bytes() if path.exists() else b""
    if raw and not raw.endswith(b"\n"):
        return records, f"{path.name}: last line is truncated"
    prev, seq = first_prev, first_seq
    for lineno, line in enumerate(raw.splitlines(), start=1):
        where = f"{path.name} line {lineno} (expected seq {seq})"
        try:
            record = TraceRecord.model_validate_json(line)
        except ValidationError as e:
            return records, f"{where}: not a trace record: {e.errors()[0]['msg']}"
        if record.seq != seq:
            return records, f"{where}: found seq {record.seq}"
        if record.segment != segment:
            return records, f"{where}: segment {record.segment!r} in {segment} file"
        if record.prev != prev:
            return records, f"{where}: prev does not link to seq {seq - 1}"
        if record.computed_hash() != record.hash:
            return records, f"{where}: hash mismatch, record was edited"
        records.append(record)
        prev, seq = record.hash, seq + 1
    return records, None


def verify_trace(
    path: Path,
    key: bytes,
    *,
    expected_head: str | None = None,
    require_seal: bool = False,
) -> VerifyResult:
    """
    Re-read a trace and check every hash, link and the seal.

    Args:
        path (`Path`):
            The episode trace file. The grading file next to it is checked too.
        key (`bytes`):
            The HMAC key the trace was sealed with.
        expected_head (`str`, *optional*):
            The head the recorder reported (e.g. [`Seal`]`.head`, kept outside the
            unit). If given, the episode file must end exactly there, which also
            catches whole lines cut from the end.
        require_seal (`bool`, *optional*, defaults to `False`):
            Treat a missing seal as a failure.

    Returns:
        [`VerifyResult`]: `ok=False` with an error naming the first bad record.
    """
    path = Path(path)
    records, error = _read_chain(path, "episode", 0, GENESIS)
    seal: Seal | None = None
    body = records
    if error is None:
        for i, record in enumerate(records):
            if record.kind == "seal":
                if i != len(records) - 1:
                    error = f"seq {records[i + 1].seq}: episode record after the seal"
                    break
                body = records[:i]
                try:
                    seal = Seal(head=record.hash, ts=record.ts, **record.data)
                except (TypeError, ValidationError):
                    error = f"seq {record.seq}: malformed seal"
                    break
                expected = _seal_hmac(key, record.prev, record.seq)
                if (
                    record.source != "control"
                    or seal.count != record.seq
                    or not hmac.compare_digest(seal.hmac, expected)
                ):
                    error = f"seq {record.seq}: seal does not match the chain"
                    seal = None
    head = records[-1].hash if records else GENESIS
    if error is None and expected_head is not None and head != expected_head:
        error = (
            f"episode ends at seq {len(records) - 1} with a head other than the "
            "expected one; records were removed or added at the end"
        )
    if error is None:
        grading_path = path.with_suffix(".grading.jsonl")
        if seal is None:
            if grading_path.exists() and grading_path.stat().st_size:
                error = "grading records exist but the episode is not sealed"
        else:
            grading, error = _read_chain(
                grading_path, "grading", seal.count + 1, seal.head
            )
            bad = next((r for r in grading if r.source not in GRADING_SOURCES), None)
            if error is None and bad is not None:
                error = f"seq {bad.seq}: source {bad.source!r} can't write grading"
    if error is None and require_seal and seal is None:
        error = "trace is not sealed"
    return VerifyResult(
        ok=error is None,
        count=len(body),
        head=head,
        sealed=seal is not None,
        error=error,
    )


@dataclass
class Mismatch:
    """A disagreement between the recorded model turns and the harness's report.

    Attributes:
        kind (`str`):
            `dropped`, `edited`, `thinking_dropped` or `fabricated`.
        request_id (`str`):
            The model request it concerns.
        recorded (`dict`, *optional*):
            What the model proxy recorded.
        reported (`dict`, *optional*):
            What the harness reported.
    """

    kind: Literal["dropped", "edited", "thinking_dropped", "fabricated"]
    request_id: str
    recorded: dict | None
    reported: dict | None


def cross_check(
    records: Iterable[TraceRecord], self_reports: Iterable[dict]
) -> list[Mismatch]:
    """
    Compare the model proxy's recorded turns with what the harness says happened.

    Args:
        records (`Iterable[TraceRecord]`):
            Trace records. Only `model.response` records are used; their `data` has
            `request_id`, `text` and `thinking`.
        self_reports (`Iterable[dict]`):
            The harness's claimed turns, each with `request_id`, `text` and
            `thinking`.

    Returns:
        `list[Mismatch]`: at most one per `(request_id, kind)`, recorded turns
        first in trace order, then fabricated ones in report order.
    """
    recorded: dict[str, dict] = {}
    for record in records:
        if record.kind == "model.response":
            recorded.setdefault(str(record.data.get("request_id")), record.data)
    reported: dict[str, dict] = {}
    for report in self_reports:
        reported.setdefault(str(report.get("request_id")), report)

    mismatches: list[Mismatch] = []
    for rid, rec in recorded.items():
        rep = reported.get(rid)
        if rep is None:
            mismatches.append(Mismatch("dropped", rid, rec, None))
            continue
        if (rec.get("text") or "").strip() != (rep.get("text") or "").strip():
            mismatches.append(Mismatch("edited", rid, rec, rep))
        if rec.get("thinking") and not rep.get("thinking"):
            mismatches.append(Mismatch("thinking_dropped", rid, rec, rep))
    for rid, rep in reported.items():
        if rid not in recorded:
            mismatches.append(Mismatch("fabricated", rid, None, rep))
    return mismatches
