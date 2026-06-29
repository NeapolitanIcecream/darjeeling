# Interactive Compile Loop Runbook

This runbook describes the missing implementation slice that lets one target
adaptation agent keep working inside one workspace while Core evaluates
candidates and writes validation feedback back into the workspace.

The active design documents remain the source of truth. If this runbook and
`docs/design/reboot/` disagree, update the design document first or change this
runbook to match it before changing code.

## Problem

The current implementation has the right lower-level pieces, but the high-level
compile path does not yet connect them into a live optimization loop.

Existing pieces:

- `Agent Workspace` creates an isolated writable workspace with `scaffolding/`,
  `runtime/`, `tests/`, `journal/`, and `submissions/`.
- The agent can write local search scripts, generated code, logs, and candidate
  artifact packages inside that workspace.
- `receive_candidate_submission(...)` accepts candidates from
  `submissions/<candidate>/artifacts/l1|l2|l3`.
- `evaluate_candidate_on_validation(...)` consumes raw validation rows inside
  Core and returns a validation `Report` plus `AgentFeedback`.
- `provide_validation_feedback(...)` writes agent-safe feedback to
  `journal/feedback-<candidate>.json`.
- `run_compile_loop(...)` can batch-scan submissions and write feedback.

Current gap:

- `start_compile_launch(...)` launches one agent command synchronously and then
  returns.
- Core does not keep running while the agent is alive.
- Core therefore does not consume new candidate submissions during the agent
  session, run validation, and write feedback for the agent to use in the next
  local-search round.

The result is that an agent can run Optuna-like or hand-written search inside
the workspace, but only against train/proxy signals unless it waits until after
the session for Core evaluation. That is not the intended compile-time feedback
loop.

## Desired Behavior

One compile launch should support this lifecycle:

1. Core builds a Snapshot and mounts only agent-visible inputs.
2. Core starts one target adaptation agent session.
3. The agent writes one or more candidate directories under `submissions/`.
4. While the agent is still running, Core notices each new candidate.
5. Core runs validation through Candidate Evaluation.
6. Core writes `AgentFeedback` into `journal/feedback-<candidate>.json`.
7. The agent reads those feedback files and continues local search.
8. The loop stops when the agent exits, time budget expires, candidate budget
   expires, cost budget expires, or Core/user stops the compile.
9. Core closes the attempt.
10. Test evaluation and release decision happen only after the attempt is
    closed.

This can be a single agent session. It must still interact with Core for
official validation feedback. The agent must not read validation or test rows
directly.

## Scope

Implement a thin Core-managed interactive compile loop.

In scope:

- Async or long-running launch for the target adaptation agent.
- Polling or watching `submissions/` for new candidate directories.
- Deduplicating already evaluated submissions.
- Calling `evaluate_candidate_on_validation(...)`.
- Writing `AgentFeedback` through `provide_validation_feedback(...)`.
- Recording validation failures as agent-safe feedback rather than crashing the
  whole compile when possible.
- Enforcing `CompileBudget.max_agent_seconds`, `CompileBudget.max_candidates`,
  and `CompileBudget.max_cost` where available.
- Closing the attempt with a clear reason.
- Tests that prove validation rows remain Core-only.

Out of scope:

- Letting the agent see validation/test raw rows.
- Letting the agent call the reference/L4 broker directly.
- Nested autonomous coding agents launched from generated target code.
- Making Optuna a required dependency.
- Reworking Snapshot, Candidate Evaluation, Release Runtime, or telemetry
  evidence architecture.
- Test/promotion automation beyond the closed-attempt boundary.

## Module Placement

Keep the split simple:

- `compile_orchestration.py` should own the end-to-end driver because it already
  owns compile scheduling and call order.
- `agent_workspace.py` should own workspace operations, agent launch mechanics,
  candidate submission intake, feedback file writing, and attempt closure.
- `candidate_evaluation.py` should remain the only owner of validation
  evaluation and `AgentFeedback` construction.

Do not add a new framework, plugin layer, scheduler abstraction, or target
specific hook for this slice.

## Implementation Steps

### 1. Add a non-blocking agent launch path

Add an Agent Workspace function that starts the target adaptation command and
returns while the process is still running.

Suggested shape:

```python
def launch_target_adaptation_agent_async(
    attempt: AgentAttempt,
    brief_path: Path,
    agent_runtime: dict[str, Any],
) -> AgentSessionHandle:
    ...
```

It should reuse the same sandbox setup as `launch_target_adaptation_agent(...)`.

It must:

- write a session record under `journal/agent_session.json`;
- record pid, command, start time, and sandbox mode;
- keep stdout/stderr in `journal/agent.log` or separate log files;
- preserve existing sync launch behavior for callers that still need it.

If preserving a live process handle inside `AgentSessionHandle` would make the
dataclass messy, keep process management in a small internal runtime object in
`compile_orchestration.py` and keep `AgentSessionHandle` as the durable record.

### 2. Add a submission ledger

Core needs to avoid evaluating the same submission repeatedly.

Use a small JSON file in the attempt journal, for example:

```text
journal/evaluated_submissions.json
```

Record:

- submission id;
- submission path digest or workspace commit;
- validation status: `feedback_written`, `evaluation_failed`, or `skipped`;
- feedback path when present;
- error class and safe error message when evaluation fails;
- timestamp.

This is compile-time bookkeeping, not an artifact.

### 3. Add the interactive driver

Add a function in Compile Orchestration.

Suggested shape:

```python
def run_interactive_compile_loop(
    compile_run: CompileRun,
    attempt: AgentAttempt,
    agent_handle: AgentSessionHandle,
    definition: TargetDefinition,
    contract: TargetRuntimeContract,
    snapshot: Snapshot,
    base_release: Release,
    reference_qualification: ReferenceQualificationReport,
    reference_usage: ReferenceUsageLedger,
    baseline_cost: dict[str, Any],
    evaluation_options: dict[str, Any],
) -> dict[str, Any]:
    ...
```

Keep the return value plain:

- compile id;
- attempt id;
- evaluated submission count;
- feedback count;
- skipped/failed submission count;
- stop reason;
- elapsed seconds;
- total candidate cost if available.

The driver should:

1. start from the existing `CompileRun` and `AgentAttempt`;
2. poll `submissions/` at a small interval;
3. call `receive_candidate_submission(...)` for new candidate directories;
4. call `evaluate_candidate_on_validation(...)`;
5. write `AgentFeedback` through `provide_validation_feedback(...)`;
6. write safe failure feedback when candidate evaluation fails before producing
   a report;
7. stop on budget, time, candidate limit, user stop marker, or agent exit;
8. close the attempt with the matching reason.

Do not expose raw validation examples, row ids, expected outputs, or
reconstructable holdout details in any feedback file.

### 4. Decide how the agent discovers feedback

Keep this file-based.

The agent should learn from the brief that:

- candidates go under `submissions/<candidate>/artifacts/...`;
- official validation feedback appears under
  `journal/feedback-<candidate>.json`;
- the agent may keep running and watch `journal/`;
- the agent may run its own local search in `scaffolding/` or `tests/`;
- the agent must not access validation/test data or call the reference broker.

Update `write_agent_brief(...)` accordingly.

### 5. Add a CLI only if it is useful for manual operation

If a CLI is added, keep it thin and target-independent.

Possible command:

```bash
darjeeling compile run-interactive /path/to/target
```

The CLI should call the same Core driver. It should not add a separate code
path for candidate evaluation or feedback.

If adding the CLI would force too much unrelated setup work, skip it and expose
the driver through tests first.

## Optuna And Similar Tools

Do not hardcode Optuna into Core.

The correct support model is:

- Core provides workspace, candidate submission, official validation feedback,
  and budgets.
- The agent may choose Optuna, grid search, custom scripts, generated programs,
  or hand-written search inside `scaffolding/`.
- If a target adaptation plan wants Optuna, it should either use an already
  available dependency or explicitly add it with a clear reason.

Darjeeling's claim is the compile loop and validation boundary, not Optuna.

## Tests

Add focused tests before broad integration tests.

Required tests:

1. A fake long-running Python agent writes candidate `c1`, waits for
   `journal/feedback-c1.json`, writes candidate `c2`, then exits.
2. The interactive loop evaluates both candidates and writes two feedback files.
3. The feedback files contain aggregate metrics and no raw validation rows,
   request ids, row ids, expected outputs, split indices, or reconstructable
   holdout material.
4. `max_candidates=1` stops after one evaluated submission and closes with
   `candidate_limit`.
5. `max_agent_seconds` stops a still-running agent and closes with
   `time_limit` or `budget_exhausted`.
6. A broken candidate produces agent-safe failure feedback and does not crash
   the whole compile loop unless the failure is a scope/invariant violation.
7. Test evaluation is not run while the attempt is open.
8. The sync launch path and existing demo still pass.

Useful integration test:

- A toy target starts from cold start, runs the interactive loop, uses feedback
  to produce an improved second candidate, closes the attempt, then runs test
  evaluation and release creation through the existing flow.

Run:

```bash
uv run pytest tests/test_04_agent_workspace.py tests/test_10_compile_orchestration.py tests/test_11_end_to_end_thin_target.py -q
uv run pytest tests -q
uv run ruff check src tests
git diff --check
```

## Done Criteria

The implementation is complete when:

- one compile launch can keep one agent session alive while Core evaluates
  multiple candidate submissions;
- validation is consumed by Core, not by the agent;
- the agent receives only `AgentFeedback`;
- the loop is bounded by candidate, time, and cost budgets;
- the attempt is closed with a clear stop reason;
- no validation/test raw rows or row identifiers appear in workspace feedback;
- existing release/test/promotion boundaries remain unchanged;
- CI passes.

## Review Checklist

When reviewing the implementation, check these points first:

- Does any agent-visible file contain validation/test rows, row ids, expected
  outputs, split indices, or reconstructable holdout membership?
- Can the agent call the reference/L4 broker directly?
- Does Core evaluate each candidate at most once unless its content changes?
- Does the loop keep running while the agent is alive?
- Are failed candidate evaluations turned into safe feedback when possible?
- Are test evaluation and release decisions still after attempt closure?
- Did the implementation add new architecture terms or framework objects that
  are not necessary for this loop?
