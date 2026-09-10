# Pathway analysis env — failure codes (v1)

Observations use **`metadata["failure_code"]`** for stable, machine-readable failure labels. Success steps may omit this field or set it to `null` in future versions.

See `server/pathway_environment.py` for where each code is set.

## v1 taxonomy

| `failure_code` | When |
|----------------|------|
| `episode_already_done` | `step` after the episode ended (`submit` or strict failure). |
| `unknown_action_type` | `action_type` is not a recognized pathway action. |
| `design_partial_contrast` | `understand_experiment_design` with only one of reference/alternate. |
| `design_invalid_contrast_names` | Proposed reference/alternate not in `conditions`, or ref equals alt. |
| `design_insufficient_samples_per_arm` | Pipeline case: a contrast arm has no samples in `sample_metadata`. |
| `de_missing_contrast` | Pipeline DE: no reference/alternate from action, validated design, or `default_contrast`. |
| `de_pydeseq2_unavailable` | PyDESeq2 not installed (non-strict: recoverable; strict: episode ends). |
| `de_deseq2_failed` | `run_deseq2_contrast` returned an error string. |
| `de_invalid_counts_matrix` | `validate_counts_case` failed. |
| `de_too_few_genes_after_filter` | Prefilter leaves too few genes for stable DESeq2. |
| `case_sample_metadata_mismatch` | `build_sample_metadata` raised (e.g. sample id missing from metadata). |
| `ora_de_prerequisite` | Pipeline ORA before DE has been run. |
| `ora_no_pathway_definitions` | Case has no `pathway_genes` for ORA. |
| `compare_missing_pathway_names` | `compare_pathways` without both `pathway_a` and `pathway_b`. |
| `expert_disabled` | `expert_budget` is 0. |
| `expert_budget_exhausted` | Expert calls used ≥ budget. |
| `submit_incorrect_hypothesis` | `submit_answer` with wrong pathway (analytics; optional). |
| `strict_termination` | Strict mode ended the episode; prefer the specific code above when also set. |

## Strict mode

When **`strict_mode`** ends an episode, observations include **`metadata["strict_failure"]: true`** and a specific **`failure_code`** (e.g. `de_missing_contrast`) when applicable, or `strict_termination` as a fallback.

## Implementation map (v1)

| Code | Where set in `pathway_environment.py` |
|------|----------------------------------------|
| `episode_already_done` | `step` when `s.is_done` |
| `unknown_action_type` | `step` fallback |
| `design_partial_contrast` | `_step_understand_experiment_design` (partial contrast) |
| `design_invalid_contrast_names` | `_validate_contrast_proposal` |
| `design_insufficient_samples_per_arm` | `_validate_contrast_proposal` |
| `de_missing_contrast` | `_step_de` (no ref/alt) |
| `de_pydeseq2_unavailable` | `_step_de` |
| `de_deseq2_failed` | `_step_de` after `run_deseq2_contrast` error |
| `de_invalid_counts_matrix` | `_step_de` after `validate_counts_case` |
| `de_too_few_genes_after_filter` | `_step_de` after prefilter |
| `case_sample_metadata_mismatch` | `_step_de` `build_sample_metadata` `ValueError` |
| `ora_de_prerequisite` | `_step_enrichment` (no DE rows, pipeline) |
| `ora_no_pathway_definitions` | `_step_enrichment` |
| `compare_missing_pathway_names` | `_step_compare` |
| `expert_disabled` | `_step_expert` |
| `expert_budget_exhausted` | `_step_expert` |
| `submit_incorrect_hypothesis` | `_step_submit` when `correct` is false |
| `strict_termination` | `_fail_strict` default only if no other code passed |

## Constants

Python constants live in **`server/failure_codes.py`** — import these instead of hard-coding strings in new code.
