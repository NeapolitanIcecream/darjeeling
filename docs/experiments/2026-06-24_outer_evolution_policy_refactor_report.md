# Outer Evolution Policy Refactor Report

Date: 2026-06-24

Branch/worktree:

```text
branch: codex/outer-evolution-policy
worktree: /Users/chenmohan/gits/darjeeling-outer-evolution-policy
```

## Summary

Implemented a small target-independent outer evolution policy module:

```text
src/darjeeling/compiler/evolution_policy.py
```

The module owns shared vocabulary and accounting for:

- budget profile names: `standard`, `fixed-inner`, `smoke`;
- requested rounds;
- resolved live agent launch caps;
- timeout and live agent command metadata;
- agent budget counters;
- budget/profile-guidance payload rendering;
- evidence-strength classification;
- common stop-reason interpretation for incomplete fixed snapshots and agent
  sessions.

No NLU intents, frames, slots, request ids, CLINC150 labels, dataset fields, or
failure examples moved into core. Core policy helpers receive only generic
strings, counters, profile names, and opaque caller-provided text.

## Control Flow

Before:

```text
L1 one-shot coding-agent job
L2 target-evolution controller with private budget/evidence helpers
L3 one-session prompt evolution with local session summary
```

After:

```text
outer evolution policy
  -> common profile, launch cap, timeout, budget, stop, evidence payloads
  -> L1 ProgramBank coding-agent adapter
  -> L2 target-evolution adapter
  -> L3 prompt-evolution adapter
```

L1, L2, and L3 now all emit policy-shaped `budget_policy`, `agent_budget`,
`stop_reason`, and evidence payloads at their stable summary/provenance
boundaries.

## Adapter Boundaries

Stayed in L1:

- Rust ProgramBank workspace creation;
- teacher-visible context/focus-task writing;
- prompt and command text;
- cargo validation;
- diff/transcript/provenance files;
- L1 workspace scope rules.

Stayed in L2:

- target trace scopes and visible/private split policy;
- local-search objective and Optuna budget extensions;
- target workspace files and tools;
- NLU target diagnostics, safety backlogs, slot/intent risk reports;
- visible/private selection and adoption gates;
- promoted target snapshot selection.

Stayed in L3:

- prompt artifact schema and validation;
- local SLM replay;
- prompt workspace files and tools;
- private holdout separation;
- prompt selection and adoption gates.

## Compatibility

L2's public target-evolution schema remains `l2-target-evolution-v1`.
Compatibility wrappers preserve existing L2 helper names and summary fields:

- `_target_budget_policy_payload`
- `_target_evidence_policy_payload`
- `_agent_budget_payload`
- `_effective_max_agent_rounds`

The L2 budget profile compatibility payload still includes the legacy
`rounds_are_l2_train_eval_iterations` key and the existing L2-specific guidance
text. `main_cli.py` now delegates live-agent cap resolution to the shared
resolver while keeping L2-local defaults for local search trials, visible folds,
cross-audit folds, and target split policy.

L1 gained policy fields without changing default behavior. The default remains
one job, and existing dry-run/codex-cli/agent-session artifact paths are still
written. L1 provenance and compiler candidate metrics now record stop reason,
round count, and agent budget.

L3 gained `budget_policy`, `agent_budget`, `evidence_policy`,
`rounds_requested`, and `rounds_completed` while retaining the existing
`agent_session` field for compatibility. The CLI gained an optional
`--budget-profile` flag; default behavior remains `standard`.

## Remaining Duplication

Command launch helpers remain layer-local. L1, L2, and L3 build different Codex
commands, prompts, working directories, report paths, and environment policies.
Extracting those now would add more abstraction than it removes.

Workspace scope checks also remain layer-local. The hashing mechanics are
similar, but writable/protected roots and violation messages are layer-owned and
small enough to keep explicit.

L2 local-search budget fields stay L2-owned because Optuna trials, cross-audit
top-k, and search spaces are target/layer semantics rather than outer policy
requirements.

## Validation

Baseline before migration:

```bash
uv run --extra dev pytest \
  tests/targets/nlu/test_l1_coding_agent.py \
  tests/targets/nlu/test_l2_target_evolution.py \
  tests/targets/nlu/test_l3_prompt_optimizer.py \
  tests/targets/nlu/test_clinc150_phase1.py \
  -q
# 82 passed
```

Focused validation after migration:

```bash
uv run --extra dev pytest \
  tests/test_evolution_policy.py \
  tests/targets/nlu/test_l1_coding_agent.py \
  tests/targets/nlu/test_l2_target_evolution.py \
  tests/targets/nlu/test_l3_prompt_optimizer.py \
  -q
# 61 passed
```

```bash
uv run --extra dev ruff check \
  src/darjeeling/compiler/evolution_policy.py \
  src/darjeeling/targets/nlu/compiler/l1_program_compiler.py \
  src/darjeeling/targets/nlu/compiler/l2_target_evolution.py \
  src/darjeeling/targets/nlu/compiler/l3_prompt_optimizer.py \
  src/darjeeling/targets/nlu/compiler/loop.py \
  src/darjeeling/targets/nlu/main_cli.py \
  src/darjeeling/targets/nlu/settings.py \
  tests/test_evolution_policy.py \
  tests/targets/nlu/test_l1_coding_agent.py \
  tests/targets/nlu/test_l3_prompt_optimizer.py
# passed
```

Final validation:

```bash
uv run --extra dev pytest \
  tests/targets/nlu/test_l1_rust_worker.py \
  tests/targets/nlu/test_l2_target_evolution.py \
  tests/targets/nlu/test_l3_prompt_optimizer.py \
  tests/targets/nlu/test_clinc150_phase1.py \
  -q
# 89 passed
```

```bash
uv run --extra dev --extra massive pytest -q
# 313 passed
```

```bash
uv run --extra dev ruff check src tests
# passed
```

```bash
git diff --check
# passed
```

The full ruff command exposed three small pre-existing lint issues outside the
refactor path: `zip(..., strict=...)` in `src/darjeeling/targets/nlu/replay.py`
and two line-length violations in `tests/targets/nlu/test_patch_runtime.py`.
They were fixed mechanically so the required full lint command passes.

## Cost

No paid benchmark calls were run. This was a local refactor and test task only.
