# Reference Echo environment canary

The lab copies `envs/echo_env` from the tested checkout byte for byte and records
its source hashes. This directory supplies only the runtime execution declaration
and public action plan. It does not contain a substitute Echo implementation.

The canary reuses the shared probe's digest-pinned base, exact OpenEnv wheel,
hash-checked wheelhouse and offline installation. Its image recipe changes only
the subject COPY and entrypoint. The validator never imports Echo on the host.

The acceptance test requires real tool responses and a continuing episode with
state counts 0, 1 and 2. The current Echo implementation emits null step rewards
and resets with a base observation despite advertising CallToolObservation as its
schema. Consequently the expected validator result is FAIL with explicit reward
and schema findings, while startup and state checks pass. A passing acceptance
test proves those compatibility findings are observed; it does not certify Echo
or complete Level 2 validation. When the environment contract changes, update the
expected findings together with evidence of the intended behavior.
