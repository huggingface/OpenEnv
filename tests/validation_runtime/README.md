# Reproducing runtime validation

This project locks the shared fixture and test toolchain separately from the main
repository. It uses Python **3.12.13**, uv **0.9.3**, a digest-pinned Python image,
and a hash-checked dependency wheelhouse. Install those Python and uv versions
first; CI installs both explicitly. Run from the repository root:

```bash
uv sync --project tests/validation_runtime --frozen
uv run --project tests/validation_runtime --frozen python scripts/validation/reproduce.py --suite fast --require-complete
uv run --project tests/validation_runtime --frozen python scripts/validation/reproduce.py --suite protocol --require-complete
uv run --project tests/validation_runtime --frozen python scripts/validation/reproduce.py --suite docker --require-complete
```

The Docker suite requires a running Linux Docker engine. Missing Docker is a
failure. The reference job uses Linux x86-64; Docker Desktop arm64 uses the same
recipe but records its different platform. No host Python code from the subject
is imported by the validator.

The Docker suite snapshots the current source, builds its exact OpenEnv wheel,
and installs that wheel into this dedicated non-editable test environment. It
downloads only binary dependencies selected from the committed lock for the
Docker platform, verifies their hashes, and installs them in the fixture image
with networking disabled. A cold cache is supported. The tests run outside the
checkout with `PYTHONPATH` removed, exercising installed package data and the
production OpenEnv `/ws` endpoint. Each launch uses a fresh subject.

The image supports controlled `VALIDATION_FAULT` modes: `good`, `bad_reward`,
`bad_observation`, `missing_done`, `bad_state`, and `startup_failure`. All fault
switches and wire corruption remain inside test assets. They share one fixture
and one public runtime plan, so a defect changes one property at a time.

Evidence is written to `outputs/validation-runtime/<run-id>/`, including source,
fixture, lock and wheel hashes; the retained wheelhouse; platform and toolchain
details; test results and bounded logs; provider cleanup evidence; and checksums.
The validation subject never receives a host socket, host credentials or host
directory mount. Tests verify run-owned containers disappear after normal completion,
failed readiness and exec timeout. No global prune is used.

```bash
python scripts/validation/verify_artifacts.py outputs/validation-runtime/<run-id>
```

`--require-complete` requires every selected acceptance test to execute without a
skip. This checks the implemented slice's inventory, not completion of all RFC
008 checks. A successful image build is not evidence for reproducible-build,
containment or resource-validation graders. Those checks land in later slices.

For changes after the first sync, reinstall the local wheel before fast/protocol
tests: `uv sync --project tests/validation_runtime --frozen --reinstall-package
openenv`. The Docker suite always builds and reinstalls its captured source.
Clean commit runs are the review evidence; dirty developer runs record exact
wheel input hashes and identify themselves as dirty.
