# Outer Evolution Policy Simplification Plan

Date: 2026-06-24

Purpose: repair and simplify the `codex/outer-evolution-policy` refactor before
it is merged. The current branch moved useful mechanics into a shared policy
module, but it also introduced forward-compatible controls and report fields
that are not fully implemented or are only useful for historical artifacts.

This plan replaces that shape with a smaller contract:

```text
one evolution round =
  produce or modify one candidate
  evaluate that candidate
  record feedback, artifacts, metrics, and stop/adoption signals
```

The default executor for a round is an L4/Codex agent. `dry-run`, fake command,
and `local-search` are internal/testing executors, not separate first-class
budget models. Every executor still produces a round result and goes through
target-owned evaluation.

## Required Context Files

Read these before changing code:

- `AGENTS.md`
- `docs/experiments/2026-06-24_outer_evolution_policy_refactor_plan.md`
- `docs/experiments/2026-06-24_outer_evolution_policy_refactor_report.md`
  from the existing worktree, if present:
  `/Users/chenmohan/gits/darjeeling-outer-evolution-policy/docs/experiments/2026-06-24_outer_evolution_policy_refactor_report.md`
- `src/darjeeling/compiler/evolution_policy.py`
- `src/darjeeling/targets/nlu/compiler/l1_program_compiler.py`
- `src/darjeeling/targets/nlu/compiler/l2_target_evolution.py`
- `src/darjeeling/targets/nlu/compiler/l3_prompt_optimizer.py`
- `src/darjeeling/targets/nlu/compiler/loop.py`
- `src/darjeeling/targets/nlu/main_cli.py`
- `src/darjeeling/targets/nlu/settings.py`
- `tests/test_evolution_policy.py`
- `tests/targets/nlu/test_l1_coding_agent.py`
- `tests/targets/nlu/test_l2_target_evolution.py`
- `tests/targets/nlu/test_l3_prompt_optimizer.py`

## Execution Location

Continue in the existing worktree and branch if available:

```text
worktree: /Users/chenmohan/gits/darjeeling-outer-evolution-policy
branch: codex/outer-evolution-policy
base commit before repair: 767f113c20129f992b676cc59781f813626eec98
```

If that worktree is missing, recreate it from the branch. Do not implement this
repair directly in the main worktree.

When complete, create a new commit on `codex/outer-evolution-policy` and report
the worktree path, branch, commit hash, and whether the tracked worktree is
clean.

## Design Rules

### No Partial Control Points

If a control point is exposed by the shared policy, settings, or CLI, every
target/layer adapter that accepts it must implement it. Do not leave a parameter
that is recorded in summaries but ignored by execution.

Examples to fix:

- `L1_AGENT_ROUNDS` currently records requested rounds but L1 runs one job.
- L3 `--max-agent-sessions` currently accepts values greater than one but runs
  at most one session.

### No Historical Compatibility Burden

Do not preserve compatibility fields only for old experiment artifacts or old
reports. Historical artifacts can be understood using the commit and report that
created them. New code should not keep fields that future runs will not use.

Delete legacy fields if there is no current or future runtime/report consumer.
This includes L2 summary fields that existed only to keep old experiment JSON
shape readable.

### Core Does Not Decide Target Quality

Core may manage run mechanics and round accounting. Core must not claim that a
candidate has sufficient quality. Target/layer adapters and outer replay own
quality decisions because they know private gates, promotion gates, replay
results, and target semantics.

Do not keep a core field named `quality_claim_supported` unless it is backed by
actual private gate and outer replay results. Prefer deleting it.

## Simplified Boundary Contract

Use a small contract. Exact names are implementation details, but the shape
should be this simple.

### Inputs: Core/Runner To Target Adapter

| Input | Meaning | Owner |
|---|---|---|
| `max_rounds` | Maximum candidate rounds to run. | Shared policy controls loop. |
| `round_timeout_s` | Per-round timeout. | Shared policy passes through; adapter/launcher enforces. |
| `patience_rounds` | Stop after this many non-improving completed rounds. | Shared policy applies to adapter result. |
| `round_executor` | Round implementation. Default is `agent`; internal/test values may include `dry-run`, fake command, or `local-search`. | Adapter implements. |
| executor-specific options | Command path, model, local-search trials, dry-run patch list, etc. | Target/layer adapter owns; keep out of generic policy when possible. |

Do not expose separate `max_agent_rounds`, `max_agent_calls`, or
`max_agent_sessions`. A round is the budget unit. If a target uses an agent
inside a round, that is the default execution path, not a second budget axis.

### Outputs: Target Adapter To Core/Report

| Output | Meaning |
|---|---|
| `round_index` | One-based round number. |
| `status` | `completed`, `failed`, `timeout`, `scope_violation`, `validation_failed`, or another small shared status. |
| `candidate_ref` | Path or artifact reference for the round candidate, if any. |
| `metrics` | Opaque target-owned metrics. |
| `diagnostics` | Opaque target-owned diagnostics. |
| `improved` | Target-owned boolean used by shared patience logic. |
| `adoptable` | Target-owned signal that the candidate may be adopted or promoted. |
| `stop_reason` | Optional target-requested stop reason. |

Shared summaries should contain:

- `max_rounds`;
- `rounds_completed`;
- `stop_reason`;
- `round_results`;
- target-owned quality/adoption decision payloads.

Do not keep `agent_budget`. If future cost accounting is needed, derive it from
round/action logs. Do not keep `local_search_consumes_llm`.

## Required Deletions Or Replacements

Remove these from the shared core policy vocabulary:

- `max_agent_rounds`;
- `agent_budget`;
- `local_search_consumes_llm`;
- `agent_session_requires_one_completed_session`;
- `quality_claim_supported`;
- `requires_private_selection_gate`;
- `requires_private_promotion_gate`;
- `requires_outer_replay`;
- `quality_profile`;
- `quality_evidence_class`, unless replaced by a small fixed run-evidence enum
  that does not claim target quality;
- `outer_replay_cadence_bound`;
- `profile_guidance`;
- `prompt_strategy`;
- `cost_policy`;
- `metadata`;
- `layer_name`, unless it is used immediately in a real report or log;
- `fixed_trace_snapshot_inner_loop` and related `inner_loop` terminology.

Remove these target-facing legacy or partial controls unless they are replaced
with real `max_rounds` behavior:

- `L1_AGENT_BUDGET_PROFILE`;
- `L1_AGENT_MAX_AGENT_ROUNDS`;
- `L3 --budget-profile`;
- `L3 --max-agent-sessions`;
- L2 `profile_intent`;
- L2 `rounds_are_l2_train_eval_iterations`;
- L2 legacy helper names used only to preserve old summary shape.

If L2 still needs a convenience preset for active future workflows, keep it
target-local and make it simply resolve explicit values such as `max_rounds`,
`patience_rounds`, visible folds, and local-search trials. Do not carry the
preset name into generic policy or summary payloads unless future reports will
actually consume it.

## Implementation Steps

### 1. Reshape The Shared Policy Module

Replace the current `OuterEvolutionPolicy` and evidence helpers with the
smallest useful run policy and summary helpers.

Expected shape:

```text
EvolutionRunPolicy
  max_rounds
  round_timeout_s
  patience_rounds
  round_executor

EvolutionRoundResult
  round_index
  status
  candidate_ref
  metrics
  diagnostics
  improved
  adoptable
  stop_reason

EvolutionRunSummary
  max_rounds
  rounds_completed
  stop_reason
  round_results
```

The exact dataclass names can differ, but keep the vocabulary at this level.
Do not encode NLU, CLINC150, private gates, promotion, or replay semantics in
the shared module.

### 2. Make L1 Rounds Real

Implement real policy-controlled L1 multi-round execution.

Minimum acceptable behavior:

- `L1_AGENT_ROUNDS` or the replacement `L1_AGENT_MAX_ROUNDS` causes L1 to
  execute that many candidate rounds unless stopped by failure or patience.
- Each round has its own job/round directory, command transcript, report, diff,
  validation result, and round result payload.
- Round 1 starts from the input crate. Later rounds start from the best or latest
  successful candidate crate, using target-owned feedback from previous rounds.
- `dry-run` supports multiple patches as multiple rounds.
- fake/codex test paths can verify multiple invocations without paid calls.
- compiler metrics record `max_rounds`, `rounds_completed`, `stop_reason`, and
  round result artifacts, not fake agent budget.

The first implementation can use a simple target-owned improvement rule, such
as "candidate validated and changed files" or benchmark improvement. Do not add
a large scoring framework.

### 3. Make L3 Rounds Real

Replace L3 `max_agent_sessions` with the same round model.

Minimum acceptable behavior:

- L3 accepts `max_rounds` through config/CLI.
- Each round launches one agent session by default, validates the prompt, runs
  the existing L3 candidate evaluation, records a candidate snapshot and round
  result, and then either continues with feedback or stops.
- `max_rounds > 1` actually runs more than one session unless stopped.
- `max_rounds=0` may be allowed only as an explicit workspace-preparation smoke
  mode if current tests require it. If kept, document it as test/internal.
- Remove `--budget-profile`; use explicit `--max-rounds` and timeout.

Keep L3 prompt schema, local SLM replay, private holdout separation, and
adoption gates target-owned.

### 4. Simplify L2 Without Historical Compatibility

L2 already has real rounds. Keep the behavior but remove the legacy payload
surface:

- Replace shared policy calls with the simplified `max_rounds` policy.
- Delete `profile_intent` unless a current future-facing report consumes it.
- Delete `rounds_are_l2_train_eval_iterations`.
- Delete core-provided `quality_claim_supported`; L2 target summary should
  report its own selection/adoption/private evidence.
- If `budget_profile` remains as a target-local CLI convenience, convert it to
  explicit resolved values and avoid carrying profile names into shared policy.
- Keep local-search trials, visible folds, cross-audit settings, target scope,
  and private gates in L2 target code.

### 5. Update CLI And Settings

Expose only controls that are real:

- L1: use one max-rounds setting and timeout. Remove budget profile and agent
  max-agent settings.
- L3: use `--max-rounds` and timeout. Remove budget profile and max sessions.
- L2: prefer explicit `--rounds`/`--max-rounds`, timeout, patience, and
  target-local local-search options.

Update help text so it describes rounds as candidate-evaluation rounds, not
agent calls or inner lifecycle.

### 6. Update Tests

Add or update focused tests:

- shared policy module has only the simplified fields and no deleted fields;
- L1 with `max_rounds=3` and fake command actually invokes three rounds and
  writes three round results;
- L1 dry-run with multiple patches maps patches to rounds;
- L3 with `max_rounds=2` and fake command actually invokes two sessions/rounds;
- L2 summaries no longer contain deleted compatibility fields;
- no summary contains `quality_claim_supported`, `requires_private_*`,
  `requires_outer_replay`, `profile_intent`, or `fixed_trace_snapshot_inner_loop`
  unless a target-specific future consumer is explicitly documented.

Do not test old artifact compatibility.

### 7. Report The Repair

Update the existing refactor report or add a short repair report:

```text
docs/experiments/2026-06-24_outer_evolution_policy_simplification_report.md
```

The report should explain:

- the final core/target boundary;
- which fields were removed and why;
- how L1/L2/L3 now implement real round controls;
- what tests prove multi-round behavior;
- any target-local convenience knobs intentionally kept.

## Validation

Run focused tests first:

```bash
uv run --extra dev pytest \
  tests/test_evolution_policy.py \
  tests/targets/nlu/test_l1_coding_agent.py \
  tests/targets/nlu/test_l2_target_evolution.py \
  tests/targets/nlu/test_l3_prompt_optimizer.py \
  -q
```

Then run:

```bash
uv run --extra dev --extra massive pytest -q
uv run --extra dev ruff check src tests
git diff --check
```

No paid benchmark calls are expected.

## Done Criteria

This repair is complete when:

- the shared policy has one unified round concept;
- there is no separate shared agent budget/call budget concept;
- L1 and L3 actually execute the exposed round count;
- L2 keeps real round behavior but drops historical compatibility fields;
- core no longer claims target quality;
- deleted fields are absent from new summaries and tests;
- focused tests, full tests with extras, ruff, and `git diff --check` pass;
- the branch has a new commit and a clean tracked worktree.

## Escalation Rules

Do not stop for routine implementation details. Make a conservative choice,
verify it with tests, and document it.

Escalate only if the repair would require changing the product goal, changing
target/core semantic boundaries, reintroducing historical compatibility fields,
or running paid benchmark calls.
