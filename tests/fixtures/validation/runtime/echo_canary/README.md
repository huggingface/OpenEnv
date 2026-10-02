# Reference Echo environment canary

The lab copies `envs/echo_env` from the tested checkout byte for byte and records
its source hashes. This directory supplies only the runtime execution declaration
and public action plan. It does not contain a substitute Echo implementation.

The canary reuses the shared probe's digest-pinned base, exact OpenEnv wheel,
hash-checked wheelhouse and offline installation. Its image recipe changes only
the subject COPY and entrypoint. The validator never imports Echo on the host.

The acceptance test requires real tool responses and a continuing episode with
state counts 0, 1 and 2. Echo explicitly advertises its base reset observation
separately from CallToolObservation step responses. Its null rewards denote
unscored, nonterminal tool calls, permitted by the Level Two reward profile.
Startup, state, reward and observation schema checks must pass. Unimplemented
checks still produce SKIP and the report remains WARN; this does not certify Echo
or complete Level 2 validation. Terminal null rewards, malformed reset observations
and legacy single-schema mismatches remain failures in dedicated regressions.
