# Harbor provider qualification

An installed adapter is not evidence that a harness works with a particular model provider. Qualify the actual harness version, model route, sandbox, capture implementation, and task set together.

## Evaluation and training capture

Use explicit `purpose="eval"` for evaluation. Hosted OpenAI, native Anthropic, and Hugging Face routes can produce graded evaluation traces without engine token IDs. An eval trace must not export a training contract, even when its endpoint happens to provide token IDs.

Use `purpose="train"` only with a verified token-capable endpoint. Training export preserves engine prompt IDs, sampled completion IDs, processed log probabilities, and loss masks. `openenv.harbor.contract.to_trace_entries` rejects evaluation traces and fatal capture findings. Do not reconstruct token IDs by tokenizing rendered conversation text or fill missing log probabilities with zeros.

A prompt rewrite may create several training rows from one rollout. That does not by itself make the sampled tokens invalid. Report rows per rollout, repeated context, retained supervision, and downstream weighting separately. Capture correctness does not establish an efficient training configuration.

Native Anthropic requests retain their original signed blocks and supported native metadata. Translation to another harness protocol rejects output semantics that cannot be preserved. The native streaming bridge buffers the upstream response and replays SDK-compatible events; it does not provide upstream first-token streaming latency.

## Evidence and support tiers

A qualification report has one cell per harness/provider pair. The provider names are `openai`, `anthropic`, `hf`, and `vllm`. Keep capture artifacts and attempt configuration alongside the report, including exact model routes, available revision pins, harness versions, task identities, sampling, and source hashes.

The report distinguishes:

- `eval_pass`: a completed, graded rollout with captured calls, no fatal capture findings, and no training export. A task score of zero is still a valid evaluation; an infrastructure failure or missing grade is not a benchmark zero.
- `capture_and_reader_pass`: exact capture passed validation and the real training reader retained the expected supervision.
- `optimizer_pass`: the current capture artifacts were consumed by a real optimizer diagnostic. Record model revision, input fingerprints, consumed rows, finite losses, and finite nonzero gradients. Explicitly state whether this was diagnostic replay and whether weight synchronization was tested.
- `failed`, `blocked`, `in_progress`, and `not_run`: retain these outcomes rather than replacing them with a passing result from a different configuration.

`harness_maturity_rows` derives support tiers from validated report cells. Stable requires all three eval profiles and a current-capture optimizer pass on vLLM. Partial or pending support is experimental. Four failed or blocked profiles are unstable for the tested matrix. None of these labels claim universal compatibility or production-scale reliability beyond the recorded coverage.

Set `OPENENV_HARBOR_QUALIFICATION_REPORT` to the report JSON path to display evidence in Gradio. The UI defaults to stable harnesses, provides an experimental opt-in, and excludes unstable harnesses. With no report, adapters are unqualified and require the experimental opt-in. Changing the filter invalidates the prior selection. Recorded results do not certify a newly entered endpoint or automatically pin its harness installation. For profile-specific evidence, the UI passes the recorded profile to the rollout: ACP supports `opencode-1.18.30`; NeMo supports `shell-1.9.0` when the example workflow package is available in the checkout. The selected profile is displayed in the agent label. Profile selection creates a local seam copy and does not mutate the global adapter registry. Programmatic callers can pass `harness_profile=` to `run_rollout` or `build_trial_config`; unknown profiles fail explicitly.

## Recorded qualification: 15 September 2026

The completed qualification attempted all 29 adapters on four provider profiles, with two fixed tasks per pair (116 pairs). Results are compatibility smoke tests, not benchmark pass@1 scores. “Stable” means passing this recorded coverage; it does not certify arbitrary models, harness upgrades, or production-scale reliability.

| Provider profile | Model | Passing adapters |
|---|---|---:|
| OpenAI evaluation | `gpt-5.4-mini-2026-03-17` | 21/29 |
| Native Anthropic evaluation | `claude-sonnet-4-5-20250929` | 20/29 |
| Hugging Face evaluation | `Qwen/Qwen3.5-9B:together` | 19/29 |
| vLLM training capture and optimizer diagnostic | `Qwen/Qwen3.5-4B` | 21/29 |

The HF route is pinned, but its hosted weights are not an immutable revision. The vLLM model revision is `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`. That profile used vLLM 0.25.1, TP=1, DP=1, BF16, a 131072-token context, processed log probabilities, engine token IDs, Qwen3 XML tool parsing, Qwen3 reasoning parsing with thinking disabled, and no image/video inputs.

There are **14 stable, 9 experimental, and 6 unstable adapters**. A failed pair means the two-task qualification did not pass; it does not necessarily mean both tasks failed or that the adapter can never support that provider.

| Adapter | Tier | OpenAI | Anthropic | HF | vLLM |
|---|---|---|---|---|---|
| acp | experimental | failed | eval_pass | eval_pass | optimizer_pass |
| antigravity-cli | experimental | failed | failed | eval_pass | optimizer_pass |
| antigravity-sdk | unstable | failed | failed | failed | failed |
| claude-code | stable | eval_pass | eval_pass | eval_pass | optimizer_pass |
| cline-cli | stable | eval_pass | eval_pass | eval_pass | optimizer_pass |
| codex | experimental | eval_pass | eval_pass | failed | optimizer_pass |
| computer-1 | unstable | failed | failed | failed | failed |
| copilot-cli | stable | eval_pass | eval_pass | eval_pass | optimizer_pass |
| cursor-cli | unstable | failed | failed | failed | failed |
| devin | unstable | failed | failed | failed | failed |
| eve | unstable | failed | failed | failed | failed |
| gemini-cli | stable | eval_pass | eval_pass | eval_pass | optimizer_pass |
| goose | experimental | eval_pass | eval_pass | failed | optimizer_pass |
| grok-build | stable | eval_pass | eval_pass | eval_pass | optimizer_pass |
| kimi-cli | stable | eval_pass | eval_pass | eval_pass | optimizer_pass |
| mimo | stable | eval_pass | eval_pass | eval_pass | optimizer_pass |
| mini-swe-agent | stable | eval_pass | eval_pass | eval_pass | optimizer_pass |
| nemo-agent | experimental | eval_pass | eval_pass | failed | optimizer_pass |
| openclaw | experimental | eval_pass | eval_pass | failed | failed |
| opencode | stable | eval_pass | eval_pass | eval_pass | optimizer_pass |
| openhands | experimental | eval_pass | eval_pass | eval_pass | failed |
| openhands-sdk | stable | eval_pass | eval_pass | eval_pass | optimizer_pass |
| pi | stable | eval_pass | eval_pass | eval_pass | optimizer_pass |
| qwen-coder | stable | eval_pass | eval_pass | eval_pass | optimizer_pass |
| rovodev-cli | unstable | failed | failed | failed | failed |
| swe-agent | experimental | eval_pass | failed | eval_pass | optimizer_pass |
| terminus-2 | stable | eval_pass | eval_pass | eval_pass | optimizer_pass |
| trae-agent | experimental | eval_pass | failed | eval_pass | optimizer_pass |
| vibe | stable | eval_pass | eval_pass | eval_pass | optimizer_pass |

### Scope and known limitations

The optimizer diagnostics consumed 99 current capture rows across 21 adapters using the real `AsyncGRPOTrainer`, with finite losses and finite nonzero gradients. They used a diagnostic advantage of +1 and did not synchronize weights. This establishes capture consumption by the trainer, not reward-normalized learning, long-run stability, or correct weighting when a rollout produces multiple rows. Claude Code and other prompt-rewriting harnesses still need row-budget and weighting checks for a particular training configuration.

ACP qualification applies only to the `opencode-1.18.30` profile, and NeMo qualification only to `shell-1.9.0`. ACP has partial native usage evidence; NeMo lacks independent native token counts. Engine capture remains authoritative, and these results do not qualify arbitrary ACP agents or NeMo workflows.

Codex, Goose, and NeMo retain HF failures. Antigravity CLI retains OpenAI and Anthropic failures. SWE-agent timed out on Anthropic; Trae-agent captured no Anthropic calls. OpenClaw and OpenHands retain training trajectory reconciliation failures. Antigravity SDK also failed strict reconciliation despite executing tools. Missing vendor credentials or application prerequisites prevented qualification of Cursor, Devin, Rovo Dev, and Eve. Computer-1 needs a separate desktop/vision qualification. Keep these failures visible; do not relax token checks to promote an adapter.

The final combined regression run passed 567 tests with two skips; native Anthropic SDK streaming replay was also checked separately. Live qualification and optimizer replay used separate services and source snapshots. Updating this documentation or a qualification report does not restart training, change an existing training snapshot, or deploy the adapter changes. A running process continues to use its configured source and services.

## Regression and live validation

Run the deterministic Harbor tests from the repository root:

```bash
PYTHONPATH=src:envs python -m pytest tests/envs/test_harbor*.py -q
```

These tests cover provider conversion, capture graphs, export masks, reconciliation, routing, lifecycle behavior, and evidence gates. They are not a replacement for live harness execution.

For live qualification, use isolated services and immutable source snapshots. Fix the task set and versions before launch; bound sandbox concurrency; save each result before proceeding. Resume by scheduling only missing cases into a new attempt directory, preserving prior failures and provenance. Reusing a capture in optimizer evidence requires its exact source hash and row fingerprint to match; a newer retry must not inherit an older optimizer pass.

Some adapters require a separately supplied application, workflow, vision input, or vendor account. Report the missing prerequisite or restriction. Do not substitute a different agent, silently remove observations, relax token checks, or claim success merely because an endpoint is reachable.
