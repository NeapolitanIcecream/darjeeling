# 2026-06-09 L2 evolution notes

## Goal

本轮目标是验证并使用 L4 coding-agent evolve L2；如果 agent patch 不能解决 L2
结构能力不足，则进入普通开发修复并重跑实验。

## Plan executed

1. Add slot-level failure context for L2 coding-agent jobs.
2. Run a real L2 Codex agent job against the observed slot-missing wrong accept.
3. If the agent job times out or produces an insufficient patch, reduce context
   and rerun the agent job.
4. Replay the agent patch on the same 3k cached setup and compare against the
   baseline.
5. If a non-L2 infrastructure issue blocks the experiment, fix it, test it, and
   rerun the experiment.
6. If the agent patch does not fix the L2 issue, make the smallest L2 design
   change needed, then rerun the same 3k experiment.

## Changes

- `da51b73` added `slot_error_summary.json` to L2 agent context, so agent jobs can
  see teacher-visible L2 wrong accepts and slot-level mismatch summaries.
- `ec858ac` is the successful compact L2 Codex agent patch. It made slot-pattern
  prefix matching treat question `is` contractions like `what's` as equivalent to
  `what is`.
- `f50a4e0` fixed experiment infrastructure: offline promotion replay now honors
  `L1_WORKER_TIMEOUT_S` instead of hardcoding a 5s Rust L1 worker timeout.
- `920340e` added a guard-protected `list_name` lexical fallback for singular
  `list` markers, for example `to do list` -> `list_name=to do`.

## Agent jobs

| job | context | model | result |
| --- | --- | --- | --- |
| `runs/l2-agent-slot-aware-job-r1` | 2200 teacher train traces + 99 hard cases | `gpt-5.5` | timed out after 900s, no diff |
| `runs/l2-agent-slot-aware-job-r2` | 145 slot-focused train traces + 4 hard cases | Codex default | succeeded, generated `ec858ac` patch |

The successful r2 job used 659,100 input tokens, 595,712 cached input tokens,
10,310 output tokens, and 5,648 reasoning output tokens as reported by Codex CLI.

## Experiment results

All runs used cached teacher labels, `LOCAL_SLM_MODE=disabled`,
`L1_AGENT_MODE=disabled`, `L4_PROPOSAL_MODE=disabled`, and
`L2_AGENT_MODE=disabled`. The successful post-timeout-fix runs used
`L1_WORKER_TIMEOUT_S=30`.

| run | relevant change | L2 accepts | L2 correct | L2 wrong | frame EM | `train-7270` |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| `runs/l2-agent-patch-tuned-3k-r1` | pre slot-agent context baseline | 12 | 11 | 1 | 0.999333 | L2 wrong accept, missing `list_name` |
| `runs/l2-agent-slot-pattern-tuned-3k-r2` | Codex agent contraction patch | 12 | 11 | 1 | 0.999333 | unchanged |
| `runs/l2-list-fallback-tuned-3k-r1` | guarded `list_name` fallback | 13 | 13 | 0 | 0.999667 | L2 correct accept |

Final run summary:

- Layer counts: `{'L0': 185, 'L1': 9, 'L2': 13, 'L3': 0, 'L4': 2793}`.
- L2 accepted accuracy: `1.000`.
- L2 wrong accept rate: `0.000`.
- Gold frame exact match: `1.000`.
- L2 p50/p95: `2.440/3.549 ms`.

## Conclusions

- L4 coding-agent evolve L2 is operational, but context size matters. The broad
  2200-trace job timed out before producing a patch; the compact failure-focused
  context completed and produced a tested patch.
- The agent-generated contraction patch was locally valid but did not fix the
  observed wrong accept, because the failing early artifact had not learned a
  transferable `list_name` pattern.
- The remaining issue was not just threshold tuning. A narrow slot postprocess
  capability was needed so the guard could evaluate the more exact frame.
- After the fallback, the known wrong accept became a correct L2 accept in the
  end-to-end 3k replay, while L2 coverage increased by one request and wrong
  accepts dropped to zero.

## Follow-ups

- Generalize slot-name-derived lexical fallbacks only when a replay-backed hard
  case justifies them; avoid turning L2 into an unbounded rule system.
- Continue treating artifact promotion as an open issue: whole-artifact
  promotion can hide single-layer regressions.
- Keep L1 hardware/runtime timeout configurable in future experiment commands.

## Autoresearch-style L2 workspace follow-up

Later on 2026-06-09, the L2 agent harness was redesigned around an
autoresearch-style isolated workspace:

- `a6f36b7` introduced `workspace/l2_research/` with stable `program.md`,
  editable `candidate/`, fixed `system/darjeeling/`, dynamic teacher-visible
  `data/`, local `tools/`, and `workspace_manifest.json`.
- The Codex stdin prompt is now the stable one-line instruction to read
  `program.md`; trace/context/metrics/objective files are no longer embedded
  in prompt text.
- `L2_AGENT_MODEL` now defaults to `gpt-5.5`, `L2_AGENT_TIMEOUT_S` defaults to
  `7200`, and Codex runs with `--ignore-user-config`, `--ignore-rules`,
  `--ephemeral`, and `--skip-git-repo-check`.
- A dry-run validation smoke confirmed that generated `tools/run_checks.py`
  overlays `candidate/` into `system/darjeeling/` and runs focused L2
  pytest/ruff successfully.

All follow-up runs used `--max-requests 500 --compile-every 500 --teacher cache`
with a copied oracle teacher cache. They were smoke runs intended to validate
the new harness and patch-adoption loop, not final 3k/10k quality claims.

| run | repo state evaluated | candidate L2 accepts | candidate frame EM | candidate wrong accept | promotion | agent patch |
| --- | --- | ---: | ---: | ---: | --- | --- |
| `runs/l2-agent-research-workspace-smoke-r1` | new workspace harness only | 12 | 0.986486 | 0.013514 | rejected: objective did not improve | lower safe threshold tie-break |
| `runs/l2-agent-research-workspace-smoke-r2` | lower-threshold patch applied | 16 | 0.977064 | 0.022936 | rejected: accuracy regression exceeds epsilon | utterance length guard feature |
| `runs/l2-agent-research-workspace-smoke-r3` | length-bucket feature applied, lower-threshold reverted | 18 | 0.977273 | 0.022727 | rejected: accuracy regression exceeds epsilon | QA slot lexical fallback |

Agent/provenance observations:

- r1 Codex job succeeded with 885,689 input tokens, 821,504 cached input tokens,
  7,852 output tokens, and 1,752 reasoning output tokens.
- r2 Codex job succeeded with 1,494,176 input tokens, 1,345,664 cached input
  tokens, 10,538 output tokens, and 2,813 reasoning output tokens.
- r3 Codex job succeeded with 1,271,731 input tokens, 1,182,080 cached input
  tokens, 11,006 output tokens, and 2,118 reasoning output tokens.
- The short prompt plus workspace files made the jobs complete reliably and
  gave high cached-input ratios, but each real GPT-5.5 job still took several
  minutes and substantial tokens.

Patch-adoption outcome:

- `75647e0` applied the r1 lower-threshold patch, then `e7a8d66` reverted it
  after r2 showed worse replay accuracy.
- `df1e4e6` applied the r2 utterance-length guard feature, then `e082923`
  reverted it after r3 still failed promotion with accuracy regression.
- The r3 QA lexical fallback was not applied. It hardcodes MASSIVE-specific
  QA intent/slot names and would move L2 toward dataset-specific rules without
  replay evidence that it fixes the current promotion failure.
- After the smoke runs, the generated `program.md` rules were tightened to make
  replay/promotion success explicit and to reject raw-coverage trades or
  dataset-specific slot hardcoding by default.
- `tools/run_checks.py` was adjusted to prefer pytest/ruff from the current
  Python environment, falling back to `uv run` only when those modules are
  unavailable. This keeps outer validation compatible while giving sandboxed
  agents a path around dependency-cache misses when a usable venv is available.

Conclusion for this follow-up:

- The redesigned L2 coding-agent harness works as intended: it isolates
  workspace writes, keeps prompt stable, records auditable artifacts, validates
  generated patches, and supports apply-commit-rerun evaluation.
- The L2 quality bottleneck remains unsolved in these 500-request smoke runs.
  Simple guard-threshold or single-feature changes trade accuracy for coverage
  and fail the promotion gate.
- Further L2 evolution should make the agent objective more explicit: prefer
  frame exactness and promotion success over raw L2 coverage, reject
  dataset-specific lexical patches by default, and consider a bounded agent
  budget/stop policy before spending another GPT-5.5 job.

## Target-dependent inner loop correction

The previous `l2_research/candidate` harness was useful for patch generation,
but it still mixed two concerns:

- It tied L2 evolution to outer replay generations, so the number of L4 agent
  attempts was limited by `compile_every` and replay throughput.
- It treated dataset-specific L2 code as if it were a Darjeeling-core patch,
  which incorrectly applied dataset-independence rules to target runtime code.

The corrected design is now implemented as `edge-mvp l2 target-evolve`:

- Outer Darjeeling creates a fixed target split and owns provenance, private
  promotion holdout, artifact adoption, and core invariants.
- Inner L2 target evolution runs many rounds inside one isolated
  `workspace/l2_target/`.
- `target/` is the only writable target-dependent runtime code area.
- `system/darjeeling/` is a read-only evaluator/core copy.
- `data/train.jsonl` and `data/inner_validation.jsonl` are visible to the
  agent; `promotion_holdout.jsonl` is stored under the outer job `private/`
  directory and is not copied into the agent workspace.

Smoke command:

```bash
uv run edge-mvp l2 target-evolve \
  --traces runs/l2-list-fallback-tuned-3k-r1/traces.jsonl \
  --out-dir runs/l2-target-evolve-inner-smoke-r2 \
  --rounds 5 \
  --max-traces 500
```

Smoke result:

- Runtime: about 8 seconds for 5 dry-run rounds over 500 trace rows.
- Split: train 300, inner validation 100, private promotion holdout 100.
- Inner validation stayed at 1 accepted / 1 correct / 0 wrong.
- Promotion holdout stayed at 0 accepted.
- Workspace manifest exposes only `train.jsonl` and `inner_validation.jsonl`;
  holdout privacy was verified by file layout.

Conclusion:

- The architectural issue is fixed: L2 inner evolution can run many fast rounds
  without waiting for a new replay prefix, and target-dependent code no longer
  needs to be judged as Darjeeling-core code.
- This dry-run smoke does not claim L2 quality improvement. The next meaningful
  experiment must run `codex-cli` or target patches against this inner loop and
  compare promotion-holdout success.
