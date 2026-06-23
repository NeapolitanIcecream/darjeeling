# CLINC150 Teacher Reliability Repair Plan

Date: 2026-06-23

Purpose: repair the CLINC150 live teacher gate so its result is trustworthy
after the completion-token cap fix. This plan is about teacher-call reliability,
incremental artifacts, resume behavior, and cost accounting. It is not a new
benchmark selection pass and not an L2/cascade experiment.

## Context

The CLINC150 Phase 1 gate failed mainly because live teacher calls produced
parse/schema failures:

- `clinc150-intent-v1`: 71/500 parse/schema failures, including 70 empty
  responses and 1 timeout.
- `clinc150-intent-v2-label-cards`: 59/500 parse/schema failures, including 58
  empty responses and 1 connection error.
- Among parsed v2 rows, only 5/441 were wrong.

A later code fix removed an unsafe `max_completion_tokens <= 64` cap from the
CLINC teacher and the generic `teacher-v2-intent-first` intent call. That cap was
likely starving visible JSON because reasoning tokens share the completion
budget on reasoning models. The next step is to rerun the teacher gate through a
more robust live-eval harness.

Relevant current files:

- `AGENTS.md`
- `docs/experiments/2026-06-23_clinc150_phase1_experiment_plan.md`
- `docs/experiments/2026-06-23_clinc150_phase1_report.md`
- `src/darjeeling/targets/nlu/layers/l4_cloud_llm.py`
- `src/darjeeling/targets/nlu/teacher_eval.py`
- `src/darjeeling/targets/nlu/clinc150_phase1.py`
- `src/darjeeling/targets/nlu/main_cli.py`
- `tests/targets/nlu/test_l4_teacher.py`
- `tests/targets/nlu/test_teacher_eval.py`
- `tests/targets/nlu/test_clinc150_phase1.py`

## Decision To Support

After this repair, make one of these decisions:

- Continue CLINC150 Phase 1 if the repaired teacher gate passes the planned
  quality and reliability thresholds.
- Keep CLINC150 paused if teacher quality is promising but infrastructure still
  shows transient reliability problems that need a small, bounded follow-up.
- Reject CLINC150 as the Phase 1 benchmark only if the repaired live path is
  reliable enough to trust and the teacher still fails semantically.

Do not reject CLINC150 based on artifacts polluted by avoidable empty-response
or interrupted-run failures.

## Scope

In scope:

- Preserve and strengthen retry behavior for empty live teacher responses.
- Record attempt-level diagnostics for live teacher calls.
- Preserve completed paid rows through incremental JSONL writes.
- Add resume support for CLINC150 teacher gate/eval runs.
- Rerun a bounded CLINC150 teacher gate after the repair.
- Update experiment docs and commit the completed work.

Out of scope:

- No open-ended prompt search.
- No L2, L3, cascade, stream, or promotion work.
- No new benchmark selection.
- No database, queue, service, job framework, plugin framework, or broad core
  abstraction.
- No CLINC150-specific semantics in Darjeeling core.

## Hypotheses

1. The previous empty responses were mostly caused by the too-small completion
   budget, not by CLINC150 being unsuitable.
2. Empty responses should remain retryable because API/model transient failures
   can still happen even with a correct token cap.
3. Attempt-level telemetry will separate output-budget issues, transient API
   issues, timeouts, parse/schema failures, and true semantic label mistakes.
4. Incremental artifact writes plus resume will make paid live gates recoverable
   without losing completed rows or undercounting cost.

## Design Direction

Keep the design simple and target-local where possible.

### Empty-Response Retry

- Keep SDK retries disabled in the OpenAI client and keep the project-level
  retry wrapper as the single retry policy.
- Keep empty message content retryable. The final failure after all retries must
  still be reported as a hard parse/schema failure.
- Add attempt-level records for live calls. Plain dictionaries are fine. Capture
  at least:
  - attempt number;
  - latency;
  - success/failure;
  - error type and message;
  - response model;
  - usage payload when present;
  - finish reason when available;
  - visible content length when available.
- Make recovered empty responses visible in metrics, for example
  `retry_recovered_rows`, `empty_response_attempts`, and
  `final_empty_response_failures`.
- Do not hide semantic parse failures by retrying arbitrary bad labels forever.
  If the implementation chooses to retry schema parse failures, keep it bounded
  and report them separately from empty responses.

### Cost Accounting

- Count all completed paid attempts, including empty attempts that returned
  usage.
- Report row-level final response cost and row-level total attempt cost if they
  differ.
- Build a run-level cost ledger from detail JSONL/attempt data, not only from a
  summary estimate.
- Avoid both large overestimates and undercounts. If a failed attempt has no
  usage payload, record that explicitly as unknown/zero-observed usage.

### Incremental Landing And Resume

- Write live eval rows incrementally to JSONL as rows complete.
- Flush after each row so an interrupted paid run preserves completed work.
- Rebuild summary JSON, CSV, and CLINC metrics from JSONL rows at the end.
- Add resume support that loads existing JSONL rows, validates run identity, and
  evaluates only missing request IDs.
- Validate at least prompt version, split, stream/sample request IDs, model, and
  schema version before resuming. Refuse to resume a mismatched run.
- Preserve existing failed rows as completed rows by default. Retrying completed
  failed rows should require an explicit option if implemented.
- For parallel runs, write from the main thread/process after futures complete;
  do not let worker threads append to the same file directly.
- Final artifacts should be sorted by the original request index even if rows
  completed out of order.

### CLI Shape

Add resume support to the CLINC150 paid live paths first:

- `edge-mvp-nlu clinc150 teacher-gate --resume-existing`
- `edge-mvp-nlu clinc150 teacher-eval --resume-existing`

If a small generic helper in `teacher_eval.py` reduces duplication, use it, but
do not create a general job system.

## Experiment

Use a new run directory such as:

```text
runs/clinc150-teacher-reliability-20260623/
```

Recommended sequence:

1. Run focused fake-client tests for retry, attempt telemetry, incremental
   JSONL, resume, and mismatch rejection.
2. Run a small paid live smoke on the fixed teacher path, preferably 20-50
   CLINC150 validation rows, with `TEACHER_MAX_TOKENS` at least 256,
   `OPENAI_TIMEOUT_S=120`, and bounded retries.
3. If the smoke has no material final empty-response failure, rerun the 500-row
   teacher gate on the planned prompt versions.
4. If final empty responses persist, inspect attempt telemetry before changing
   prompts. Possible next moves are higher `TEACHER_MAX_TOKENS`, lower
   concurrency, or a tighter retry/backoff policy. Treat streaming as a later
   hypothesis, not the first repair.
5. Update the CLINC150 Phase 1 report or write a short reliability report with
   the new metrics and decision.

Paid LLM calls are allowed for this repair. Keep new observed spend under $10
unless the evidence clearly shows a larger rerun is necessary; in all cases stay
under the previously discussed $100 ceiling and report observed spend from
artifacts.

## Validation

Run at least:

```bash
uv run pytest tests/targets/nlu/test_l4_teacher.py tests/targets/nlu/test_teacher_eval.py tests/targets/nlu/test_clinc150_phase1.py -q
uv run pytest -q
uv run ruff check <touched python files>
git diff --check
```

The paid live run is part of validation unless `OPENAI_API_KEY` is unavailable.
If live calls cannot run, document that as a blocker rather than claiming the
repair is validated.

## Done Criteria

- Empty response retry remains covered by tests and attempt telemetry.
- Incremental JSONL artifacts survive interruption and resume without duplicate
  rows.
- Resume rejects mismatched prompt/model/sample runs.
- Cost accounting includes all observed completed attempts.
- A bounded live CLINC150 teacher gate has either passed, or failed with enough
  telemetry to distinguish infrastructure failures from semantic teacher
  mistakes.
- Experiment docs are updated.
- All checks pass.
- Changes are organized into a git commit.
