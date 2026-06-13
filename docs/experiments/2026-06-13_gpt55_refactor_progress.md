# GPT-5.5 Review Refactor Progress

Date: 2026-06-13

Scope: repository-level implementation for the GPT-5.5 review refactor plan.

Constraints observed:

- Darjeeling core remains target-, dataset-, and application-independent.
- NLU frame, intent, slot, audit, expert, and residual details stay in the NLU target.
- Generated L1/L2/L3 task workspaces are not edited as the repo fix.
- New wiring uses plain Python helpers, explicit artifacts, and static manifests.

## Progress

- Started from the verified design gaps in `docs/design/05_gpt55_review_refactor_plan.md`.
- Existing working tree already contained the review verification and refactor plan docs as untracked user-provided context.
- Completed implementation slices:
  1. NLU patch/composer runtime and replay metrics.
  2. Lower-layer teacher audit metadata and hard-buffer disagreement evidence.
  3. L2 target evolution in main candidate replay/promotion.
  4. L2 expert-bank artifact and runtime patch emission.
  5. Ranked focus-task contexts for L4 proposals and L1 agent workspaces.
  6. L3 residual value gate and reduced decode budget.
  7. Compiler-loop helper split.
  8. Focused tests, docs, and target/core boundary regression coverage.

## Outcome Notes

- Core router/contracts were not changed.
- New NLU terms (`FramePatch`, field metrics, audit disagreement, focus tasks, experts, residual gate) live in `darjeeling.targets.nlu`.
- L3 guarded runtime is deliberately disabled when residual replay evidence does not show accuracy-safe cost/latency value.

## Verification Log

- `uv run ruff check src/darjeeling/targets/nlu tests/targets/nlu` passed.
- `uv run pytest tests/targets/nlu -q` passed: 213 tests in 41.99s.
- `uv run pytest tests -q` passed: 237 tests in 38.58s.
- `git diff --check` passed.
