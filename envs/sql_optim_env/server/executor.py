"""
executor.py — DuckDB In-Memory SQL Execution Engine
=====================================================
The core innovation of this environment: instead of keyword-matching
heuristics, we ACTUALLY execute both the original and optimized queries
against realistic synthetic data and measure real performance differences.

Tables populated:
  users    — 10,000 rows
  orders   — 500,000 rows
  products —  1,000 rows
  events   — 1,000,000 rows
"""

import atexit
import contextlib
import os
import re
import shutil
import tempfile
import threading
import time
from typing import Any, Dict, Iterator, List, Optional, Tuple

import duckdb

_instance: Optional["QueryExecutor"] = None
_lock = threading.Lock()

# Result sets larger than this are never pulled into Python: the row count and
# the correctness check are computed inside DuckDB instead. `SELECT * FROM
# events` is a million rows, and materializing that (three times, for the
# timing median) both dominated the measurement and held the whole result in
# memory. The cap matches the threshold `compare()` already used to pick the
# precise row-by-row comparison, so behaviour for small results is unchanged.
_MAX_MATERIALIZED_ROWS = 50_000
_FETCH_BATCH = 10_000
_TIMING_RUNS = 3
# `EXPLAIN ANALYZE` prints seconds with four decimals, so 0.1 ms is the smallest
# value it can express. Used as a floor so a sub-resolution query can never make
# the speedup ratio divide by zero.
_MIN_MEASURABLE_MS = 0.1
_TOTAL_TIME_RE = re.compile(r"Total Time:\s*([0-9]*\.?[0-9]+)\s*s")

# Opening the connection read-only stops the agent from changing the *database*,
# but on its own it leaves DuckDB's filesystem escape hatches open: `COPY ... TO`
# writes arbitrary files, `read_csv`/`read_text`/`glob` read them, `ATTACH`
# mounts other databases and `INSTALL`/`LOAD` pull in extensions. Since the query
# under measurement is agent-authored, external access is disabled outright, and
# the configuration is locked so a rewrite cannot `SET` its way back out (DuckDB
# also refuses to re-enable external access on a running database).
#
# The same reasoning applies to resources. The agent chooses the SQL, so a
# rewrite like `FROM events a, events b` is a trillion-row cross join: without
# a ceiling it can exhaust memory, fill the spill directory, and hold the
# execution lock for hours while every other session waits behind it. DuckDB has
# no statement-timeout option, so the limits below bound memory and spill space
# and `_deadline()` bounds time.
_DUCKDB_CONFIG = {
    "threads": "2",
    "enable_external_access": "false",
    "lock_configuration": "true",
    "memory_limit": "1GB",
    "max_temp_directory_size": "1GB",
}
# Generous next to the real tasks (the slowest is ~0.6s) and short enough that a
# runaway rewrite cannot wedge the server.
_QUERY_TIMEOUT_S = 15.0
_TIMEOUT_ERROR = (
    f"Query cancelled: exceeded the {_QUERY_TIMEOUT_S:.0f}s execution limit"
)


def _strip_terminators(query: str) -> str:
    """Trim surrounding whitespace and any trailing semicolons.

    Trailing ``;`` breaks queries that are wrapped in a subquery
    (``SELECT COUNT(*) FROM (<query>) t``) for correctness checksums.
    """
    return query.strip().rstrip(";").strip()


def _as_subquery(query: str, select: str) -> str:
    """Build ``SELECT <select> FROM (<query>) t``.

    The query goes on its own line: a rewrite ending in a ``--`` comment would
    otherwise swallow the closing parenthesis and turn a valid query into a
    parse error, which used to cost the checksum its result.
    """
    return f"SELECT {select} FROM (\n{query}\n) t"


class QueryExecutor:
    """
    Runs SQL against an in-memory DuckDB database with realistic
    synthetic data.  Provides execution timing, result correctness
    checks, and EXPLAIN plans — all used by the reward function.
    """

    def __init__(self) -> None:
        # Build the tables once on a writable connection to a temp file, then
        # reopen the database read-only. Every query — trusted task SQL and
        # agent rewrites alike — runs against the read-only connection, so the
        # DuckDB engine itself rejects any write (no keyword heuristics, and DML
        # inside a CTE cannot slip through on any DuckDB version) and, per
        # `_DUCKDB_CONFIG`, cannot reach the filesystem either. A lock
        # serializes access because a single DuckDB connection is not safe for
        # concurrent use by the server's worker pool.
        self._dir = tempfile.mkdtemp(prefix="sql_optim_")
        self._path = os.path.join(self._dir, "sql_optim.duckdb")
        builder = duckdb.connect(self._path, config=_DUCKDB_CONFIG)
        self._build_tables(builder)
        builder.close()

        self.conn = duckdb.connect(self._path, read_only=True, config=_DUCKDB_CONFIG)
        # Reentrant so `_run` can hold the lock across a whole measurement while
        # the helpers it calls still take it individually.
        self._exec_lock = threading.RLock()
        atexit.register(self._cleanup)

    def _cleanup(self) -> None:
        """Close the connection and remove the temp database directory."""
        try:
            self.conn.close()
        except Exception:
            pass
        shutil.rmtree(self._dir, ignore_errors=True)

    # ── Schema Setup ─────────────────────────────────────────────────────

    def _build_tables(self, conn: "duckdb.DuckDBPyConnection") -> None:
        """Create and populate all four synthetic tables on *conn*."""

        # users — 10k rows
        conn.execute("""
            CREATE TABLE users AS
            SELECT
                i                                                      AS id,
                'u' || i || '@mail.com'                                AS email,
                CASE i % 3
                    WHEN 0 THEN 'premium'
                    WHEN 1 THEN 'free'
                    ELSE 'enterprise' END                              AS tier,
                CASE i % 5
                    WHEN 0 THEN 'US'   WHEN 1 THEN 'EU'
                    WHEN 2 THEN 'IN'   WHEN 3 THEN 'UK'
                    ELSE 'AU' END                                      AS region,
                CASE i % 2 WHEN 0 THEN 'premium' ELSE 'basic' END     AS plan,
                DATE '2020-01-01' + CAST(i AS INTEGER)                 AS created_at
            FROM generate_series(1, 10000) t(i)
        """)

        # orders — 500k rows
        conn.execute("""
            CREATE TABLE orders AS
            SELECT
                i                                                      AS id,
                1 + (i % 10000)                                        AS customer_id,
                (i % 100) + 1                                          AS product_id,
                CASE i % 4
                    WHEN 0 THEN 'completed'  WHEN 1 THEN 'pending'
                    WHEN 2 THEN 'cancelled'  ELSE 'shipped' END        AS status,
                ROUND((i % 1000) * 1.5 + 49.99, 2)                   AS total,
                DATE '2023-01-01' + CAST(i % 730 AS INTEGER)          AS created_at
            FROM generate_series(1, 500000) t(i)
        """)

        # products — 1k rows
        conn.execute("""
            CREATE TABLE products AS
            SELECT
                i                                                      AS id,
                'Product_' || i                                        AS name,
                CASE i % 5
                    WHEN 0 THEN 'Electronics'  WHEN 1 THEN 'Clothing'
                    WHEN 2 THEN 'Food'         WHEN 3 THEN 'Books'
                    ELSE 'Sports' END                                  AS category,
                ROUND((i % 500) + 9.99, 2)                            AS price
            FROM generate_series(1, 1000) t(i)
        """)

        # events — 1M rows
        conn.execute("""
            CREATE TABLE events AS
            SELECT
                i                                                      AS id,
                1 + (i % 10000)                                        AS user_id,
                'sess_' || (i % 50000)                                 AS session_id,
                CASE i % 6
                    WHEN 0 THEN 'purchase'  WHEN 1 THEN 'view'
                    WHEN 2 THEN 'click'     WHEN 3 THEN 'signup'
                    WHEN 4 THEN 'logout'    ELSE 'search' END          AS event_type,
                DATE '2024-01-01' + CAST(i % 365 AS INTEGER)          AS occurred_at
            FROM generate_series(1, 1000000) t(i)
        """)

    # ── Execution helpers ─────────────────────────────────────────────────

    @contextlib.contextmanager
    def _deadline(self) -> Iterator[None]:
        """Cancel whatever is running on the connection after the time limit.

        DuckDB has no statement timeout, but ``interrupt()`` cancels the running
        query from another thread and leaves the connection usable, raising
        ``duckdb.InterruptException`` in the thread that issued it.

        Armed by every helper that executes agent SQL *and* around the whole of
        ``_run``, so a helper is bounded whoever calls it while the measurement
        as a whole is bounded too. Nesting is harmless: each level cancels its
        own timer, and the earliest one to fire wins.

        Must be entered while already holding ``_exec_lock`` — every call site
        does, as ``with self._exec_lock, self._deadline():``. That ordering is
        what makes the unavoidable cancel race benign: the timer can fire in the
        window between the statement finishing and ``cancel()`` landing, but
        because the lock is only released *after* this context exits, no other
        session can have a query in flight, so the stray ``interrupt()`` lands on
        an idle connection and does nothing.

        Callers must not treat the interrupt as a reason to try a different
        strategy: the timer has already fired, so a retry would run unbounded.
        """
        timer = threading.Timer(_QUERY_TIMEOUT_S, self.conn.interrupt)
        timer.daemon = True
        timer.start()
        try:
            yield
        finally:
            timer.cancel()

    def _probe(self, query: str) -> Tuple[Optional[List], Optional[str]]:
        """
        Fetch at most ``_MAX_MATERIALIZED_ROWS`` rows of *query*.

        Returns (rows, error). ``rows`` is None when the result set is larger
        than the cap — the caller then works from the in-engine row count and
        checksum instead of holding the full result in Python.

        This is also the authoritative error check: the query runs unwrapped on
        the read-only connection, so a write (DDL/DML, inside a CTE, or after a
        ``;``) still surfaces as a DuckDB error exactly as before.
        """
        try:
            with self._exec_lock, self._deadline():
                # `execute()` returns the connection itself, so the leftover rows
                # of an over-cap result cannot be released with a `close()`
                # (that would close the shared read-only connection). The next
                # statement on the connection supersedes the pending result,
                # and every caller issues one straight after.
                head = self.conn.execute(query).fetchmany(_MAX_MATERIALIZED_ROWS + 1)
        except duckdb.InterruptException:
            return None, _TIMEOUT_ERROR
        except Exception as exc:
            return None, str(exc)
        if len(head) > _MAX_MATERIALIZED_ROWS:
            return None, None
        return head, None

    def _row_count(self, query: str) -> Tuple[Optional[int], Optional[str]]:
        """
        Exact row count for *query*, computed inside DuckDB — no rows cross into
        Python. Falls back to a streaming drain (bounded to one batch of memory)
        for the rare query that cannot be wrapped in a subquery, but never after
        a timeout: the drain would be at least as expensive as the query that
        just ran out of time.
        """
        try:
            with self._exec_lock, self._deadline():
                return (
                    self.conn.execute(_as_subquery(query, "COUNT(*)")).fetchone()[0],
                    None,
                )
        except duckdb.InterruptException:
            return None, _TIMEOUT_ERROR
        except Exception:
            pass

        try:
            count = 0
            with self._exec_lock, self._deadline():
                cursor = self.conn.execute(query)
                while True:
                    batch = cursor.fetchmany(_FETCH_BATCH)
                    if not batch:
                        break
                    count += len(batch)
            return count, None
        except duckdb.InterruptException:
            return None, _TIMEOUT_ERROR
        except Exception as exc:
            return None, str(exc)

    def _engine_ms(self, query: str, runs: int) -> Optional[float]:
        """
        Median DuckDB-side execution time in ms, or None if unavailable.

        ``EXPLAIN ANALYZE`` really executes the query and discards the result
        inside the engine, so this measures execution alone: no rows are
        converted to Python objects and nothing is buffered client-side. Returns
        None when the profile footer cannot be parsed (a DuckDB version whose
        output differs), which makes the caller fall back to wall-clock timing.
        A timeout propagates instead, so the caller does not answer it by
        running the query yet again.
        """
        timings: List[float] = []
        for _ in range(runs):
            try:
                with self._exec_lock, self._deadline():
                    rows = self.conn.execute(f"EXPLAIN ANALYZE {query}").fetchall()
            except duckdb.InterruptException:
                raise
            except Exception:
                return None
            match = _TOTAL_TIME_RE.search("\n".join(str(r[-1]) for r in rows))
            if match is None:
                return None
            timings.append(float(match.group(1)) * 1000.0)

        timings.sort()
        return round(max(timings[len(timings) // 2], _MIN_MEASURABLE_MS), 3)

    def _wall_ms(self, query: str, runs: int) -> Tuple[float, Optional[str]]:
        """
        Wall-clock fallback timing: time a streaming drain that discards rows.

        Includes client-side conversion cost, so it is less faithful than
        ``_engine_ms``, but memory stays bounded to a single batch instead of
        the whole result set.
        """
        timings: List[float] = []
        for _ in range(runs):
            try:
                with self._exec_lock, self._deadline():
                    t0 = time.perf_counter()
                    cursor = self.conn.execute(query)
                    while cursor.fetchmany(_FETCH_BATCH):
                        pass
                    timings.append((time.perf_counter() - t0) * 1000.0)
            except duckdb.InterruptException:
                raise
            except Exception as exc:
                return 99_999.0, str(exc)

        timings.sort()
        return round(max(timings[len(timings) // 2], _MIN_MEASURABLE_MS), 3), None

    def _run(
        self, query: str, runs: int = _TIMING_RUNS
    ) -> Tuple[float, Optional[int], Optional[List], Optional[str]]:
        """
        Execute *query* on the read-only connection and measure it.

        Returns (median_ms, row_count, rows, error_or_None). ``row_count`` is
        exact whenever there is no error; ``rows`` is None for result sets above
        ``_MAX_MATERIALIZED_ROWS``. A write raises and is returned as the error
        string.

        The timing comes from DuckDB's own profiler rather than from wrapping a
        ``fetchall()``, so it reflects query execution instead of the cost of
        building Python tuples — for a 1M-row scan the latter was roughly 30x
        the former and swamped the signal the reward function is built on.

        The whole measurement shares one deadline, so an agent-authored query
        cannot hold the execution lock — and with it every other session — for
        longer than ``_QUERY_TIMEOUT_S``.
        """
        with self._exec_lock, self._deadline():
            try:
                rows, error = self._probe(query)
                if error is not None:
                    return 99_999.0, None, None, error

                if rows is not None:
                    row_count: Optional[int] = len(rows)
                else:
                    row_count, error = self._row_count(query)
                    if error is not None:
                        return 99_999.0, None, None, error

                median_ms = self._engine_ms(query, runs)
                if median_ms is None:
                    median_ms, error = self._wall_ms(query, runs)
                    if error is not None:
                        return 99_999.0, None, None, error
            except duckdb.InterruptException:
                return 99_999.0, None, None, _TIMEOUT_ERROR

        return median_ms, row_count, rows, None

    def _checksum(
        self, query: str
    ) -> Tuple[Optional[int], Optional[int], Optional[str]]:
        """
        Compute a deterministic (row-order-independent) checksum.
        Returns (row_count, checksum, error).

        BIT_XOR is commutative+associative — order-independent fingerprint.
        Falls back to count-only if the DuckDB version doesn't support the
        function, but not past the shared deadline: once a strategy has timed
        out, the cheaper ones would time out too.
        """
        with self._exec_lock, self._deadline():
            # Try BIT_XOR of a numeric hash (portable across DuckDB versions)
            selects = [
                # Option 1: BIT_XOR of md5 prefix cast to integer
                (
                    "COUNT(*) AS cnt, BIT_XOR(CAST(('0x' || "
                    "LEFT(md5(CAST(t AS VARCHAR)), 15)) AS UBIGINT)) AS chk"
                ),
                # Option 2: sum of hash (order-independent since sum is commutative)
                "COUNT(*) AS cnt, SUM(hash(CAST(t AS VARCHAR)) % 9999999999) AS chk",
                # Final fallback: count only
                "COUNT(*) AS cnt, NULL AS chk",
            ]
            error: Optional[str] = None
            for select in selects:
                try:
                    result = self.conn.execute(_as_subquery(query, select)).fetchone()
                    return result[0], result[1], None
                except duckdb.InterruptException:
                    return None, None, _TIMEOUT_ERROR
                except Exception as exc:
                    error = str(exc)
            return None, None, error

    # ── Public API ────────────────────────────────────────────────────────

    def compare(self, original: str, optimized: str) -> Dict[str, Any]:
        """
        Execute both queries, measure real timing, check correctness.

        Returns a dict with:
          original_ms, optimized_ms, speedup,
          results_match, original_rows, optimized_rows,
          original_error, optimized_error, verdict
        """
        # Normalize (drop trailing ``;``) so both direct execution and the
        # subquery-wrapped checksum are valid.
        original = _strip_terminators(original)
        optimized = _strip_terminators(optimized)

        orig_ms, orig_count, orig_rows, orig_err = self._run(original)

        # The optimized query is agent-authored, but the read-only connection is
        # the guarantee: any write (DDL/DML, in a CTE, or after a `;`) is rejected
        # by the DuckDB engine and surfaces as `optimized_error`. No structural
        # pre-check is needed, which also avoids false rejections of valid
        # rewrites (leading comments, a `;` inside a string literal, etc.).
        opt_ms, opt_count, opt_rows, opt_err = self._run(optimized)

        # ── Correctness: do both queries return the same data? ────────
        # Use a DuckDB-level checksum (order-independent) to avoid
        # false negatives from non-deterministic row ordering in parallel
        # window function queries on large tables.
        results_match = False
        if orig_err is None and opt_err is None:
            try:
                if orig_count != opt_count:
                    results_match = False
                elif orig_count == 0:
                    results_match = True
                elif orig_rows is not None and opt_rows is not None:
                    # Small/medium: full sorted comparison (precise). Equal row
                    # counts mean both sides are on the same side of the
                    # materialization cap, so either both rows are present or
                    # neither is.
                    orig_s = sorted(str(r) for r in orig_rows)
                    opt_s = sorted(str(r) for r in opt_rows)
                    results_match = orig_s == opt_s
                else:
                    # Large result sets: use SQL-level hash checksum
                    # (deterministic regardless of row ordering / thread count)
                    o_cnt, o_chk, o_err2 = self._checksum(original)
                    p_cnt, p_chk, p_err2 = self._checksum(optimized)
                    if o_err2 or p_err2 or o_chk is None or p_chk is None:
                        # No usable checksum. Equal row counts alone are NOT
                        # evidence of equal data on a result set this large, and
                        # treating them as a match handed out full correctness
                        # credit (and with it the speedup slice) for a rewrite
                        # that returned different rows. Withhold the credit
                        # instead of guessing.
                        results_match = False
                    else:
                        results_match = (o_cnt == p_cnt) and (o_chk == p_chk)
            except Exception:
                # Same reasoning as the missing-checksum case: an unverifiable
                # comparison earns no correctness credit.
                results_match = False

        # ── Speedup ratio ─────────────────────────────────────────────
        speedup = 1.0
        if opt_ms > 0 and orig_ms < 90_000:
            speedup = round(orig_ms / opt_ms, 3)

        # ── Human-readable verdict ────────────────────────────────────
        if opt_err:
            verdict = f"[FAIL] Optimized query error: {opt_err[:120]}"
        elif results_match and speedup >= 2.0:
            verdict = f"[OK] {speedup:.1f}x faster with correct results"
        elif results_match and speedup >= 1.0:
            verdict = (
                f"[WARN] Correct results but only {speedup:.1f}x speedup -- dig deeper"
            )
        elif not results_match and speedup >= 2.0:
            verdict = (
                f"[WARN] {speedup:.1f}x faster but results don't match -- fix the logic"
            )
        else:
            verdict = f"[FAIL] {speedup:.1f}x -- no meaningful improvement"

        return {
            "original_ms": orig_ms,
            "optimized_ms": opt_ms,
            "speedup": speedup,
            "results_match": results_match,
            "original_rows": orig_count if orig_count is not None else 0,
            "optimized_rows": opt_count if opt_count is not None else 0,
            "original_error": orig_err,
            "optimized_error": opt_err,
            "verdict": verdict,
        }

    def explain(self, query: str) -> str:
        """Return EXPLAIN output for a query.

        Plain ``EXPLAIN`` only plans the query, but it is still agent-authored,
        so it runs under the same deadline as everything else.
        """
        try:
            with self._exec_lock, self._deadline():
                rows = self.conn.execute(
                    f"EXPLAIN {_strip_terminators(query)}"
                ).fetchall()
            return "\n".join(str(r[1]) for r in rows)
        except duckdb.InterruptException:
            return f"EXPLAIN error: {_TIMEOUT_ERROR}"
        except Exception as exc:
            return f"EXPLAIN error: {exc}"

    @property
    def table_stats(self) -> Dict[str, int]:
        tables = ["users", "orders", "products", "events"]
        # No deadline: these are fixed, trusted queries, so arming one would only
        # add a window in which a stray interrupt could cancel them.
        with self._exec_lock:
            return {
                t: self.conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
                for t in tables
            }


# ── Singleton accessor ────────────────────────────────────────────────────


def get_executor() -> QueryExecutor:
    """Return the process-level singleton (lazy init, thread-safe)."""
    global _instance
    if _instance is None:
        with _lock:
            if _instance is None:
                _instance = QueryExecutor()
    return _instance
