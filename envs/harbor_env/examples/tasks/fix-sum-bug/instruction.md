# Fix `stats.py`

`stats.py` in your working directory provides two helpers, and both are wrong at
the edges:

- `total(values)` must return the sum of `values`, and `0` for an empty sequence.
- `mean(values)` must return the arithmetic mean, and `0.0` for an empty sequence
  (rather than raising `ZeroDivisionError`).

Fix `stats.py` so both behave as described. Keep the function names and
signatures unchanged — the grader imports them directly.

The grader awards partial credit: your reward is the fraction of its checks that
pass.
