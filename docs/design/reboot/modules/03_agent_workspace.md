# Agent Workspace Module

## Purpose

The Agent Workspace module creates and controls the isolated workspace where one target adaptation agent writes compile-time scaffolding and L1/L2/L3 runtime source. It gives the agent enough target context to improve artifacts, but never gives it validation/test rows, evaluator authority, production credentials, or release authority.

There is one logical agent role per adaptation attempt: the target adaptation agent. Core may launch multiple isolated attempts in parallel, but generated code must not launch another autonomous coding agent.

This module follows the [System Invariants](../00_overall_design.md#system-invariants), especially the agent boundary.

## Boundary

Inputs:

- `TargetDefinition` and read-only target view from Target Definition.
- `Snapshot` and train view from Snapshot And Reference.
- Current `Release` from Release Runtime, with source snapshot and artifact metadata when it is a compiled Release.
- `AgentVisibleReport` summaries from Candidate Evaluation.
- Agent-visible telemetry summaries from Telemetry Evidence And Recompile.
- Runtime protocol docs from Artifact Worker.
- Compile budget, time boundary, and launch decision from Compile Orchestration.

Outputs:

- Writable workspace with `scaffolding/`, `runtime/`, `submissions/`, `proposals/`, `journal/`, and `tests/`.
- `CandidateSubmission` directories to Artifact Worker and Candidate Evaluation.
- Agent command log and usage ledger to Report.
- `WorkspaceBaselineUpdate` after an accepted Release or explicit carry-forward decision.
- Optional proposals for target definition changes to user review.

This module must not output raw validation/test rows or allow the agent to write active target definition, snapshot, evaluator, registry, or telemetry stores.

## Workspace Layout

There is one long-term agent-managed workspace per target. Compile attempts are isolated clones of that workspace at a known baseline commit:

```text
workspaces/<target>/main/
  scaffolding/
  runtime/
    l1/
    l2/
    l3/
  proposals/
  journal/
  tests/

workspaces/<target>/attempts/<compile-id>/<attempt-id>/
  scaffolding/
  runtime/
    l1/
    l2/
    l3/
  submissions/
  proposals/
  journal/
  tests/
```

Core only enforces the top-level writable directories:

```text
scaffolding/
runtime/
submissions/
proposals/
journal/
tests/
```

`main/` is not a production artifact and is not directly evaluated. It is the source baseline for future attempts. An attempt's `final_commit` may advance the long-term baseline only after the related Candidate becomes an accepted Release or after an explicit user/Core carry-forward decision.

Subdirectories under `scaffolding/` are conventions, not Core protocols. The agent may create miners, profilers, trainers, search scripts, code generators, benchmarks, or report reducers as ordinary files.

## Data Types

### `CompileRun`

Fields:

- `compile_id: str`
- `target_name: str`
- `contract_hash: str`
- `snapshot_id: str`
- `snapshot_digest: str`
- `base_release_id: str`
- `workspace_baseline_commit: str`
- `started_at: datetime`
- `budget: CompileBudget`
- `status: Literal["running", "closing", "closed", "failed"]`

### `TargetWorkspace`

Fields:

- `target_name: str`
- `workspace_path: Path`
- `baseline_commit: str`
- `contract_hash: str`
- `last_accepted_release_id: str | None`
- `status: Literal["active", "archived"]`

### `WorkspaceBaselineUpdate`

Fields:

- `target_name: str`
- `previous_commit: str`
- `new_commit: str`
- `source_attempt_id: str`
- `source_release_id: str | None`
- `reason: Literal["accepted_release", "explicit_carry_forward"]`

### `AgentAttempt`

Fields:

- `attempt_id: str`
- `compile_id: str`
- `target_name: str`
- `contract_hash: str`
- `snapshot_id: str`
- `snapshot_digest: str`
- `workspace_path: Path`
- `source_workspace_commit: str`
- `initial_commit: str | None`
- `final_commit: str | None`
- `agent_model: str`
- `status: Literal["running", "closed", "failed", "timed_out"]`

### `AgentSearchGuidance`

Fields:

- `preferred_strategies: list[str]`
- `preferred_tools: list[str]`
- `extra_instructions: str | None`

The strategy and tool strings are user guidance only. Core renders them into the
brief and does not interpret them as an optimizer framework, plugin registry, or
target-specific rule set.

### `AgentWorkspacePermissions`

Fields:

- `network_access: bool`
- `dependency_install: bool`

Defaults are conservative: no network research and no workspace-local
dependency installation. When a user explicitly enables either permission, the
agent launch policy records that choice and relaxes the matching runtime
restriction for the attempt workspace.

Current target-adaptation agent execution support is the macOS
`sandbox-exec` launch path. If `sandbox-exec` is unavailable, Core fails the
agent launch clearly instead of falling back to a Python-level sandbox; portable
or non-macOS execution needs a future external runner/container design.

### `AgentAttemptOptions`

Fields:

- `agent_model: str`
- `agent_command: list[str]`
- `agent_timeout_seconds: int | None`
- `permissions: AgentWorkspacePermissions`

### `ClosedAgentAttempt`

Fields:

- `attempt_id: str`
- `compile_id: str`
- `target_name: str`
- `contract_hash: str`
- `snapshot_id: str`
- `snapshot_digest: str`
- `source_workspace_commit: str`
- `workspace_path: Path`
- `final_commit: str`
- `status: Literal["closed", "failed", "timed_out"]`
- `usage: dict`

### `CandidateSubmission`

Fields:

- `submission_id: str`
- `compile_id: str`
- `attempt_id: str`
- `submission_path: Path`
- `workspace_commit: str`
- `submitted_at: datetime`
- `declared_layers: list[Literal["L1", "L2", "L3"]]`

The submission is an input to Core evaluation, not an evaluation result.

### `AgentFeedback`

Fields:

- `candidate_id: str`
- `summary: dict`
- `requirement_results: list[dict]`
- `metrics: dict`
- `safe_slice_summaries: list[dict]`
- `latency_cost_summary: dict`
- `raw_rows_included: Literal[False]`

## Functions

### `create_compile_run`

Input:

- `definition: TargetDefinition`
- `target_check: TargetCheckReport`
- `snapshot: Snapshot`
- `base_release: Release`
- `budget: CompileBudget`
- `workspace: TargetWorkspace`
- `reference_qualification: ReferenceQualificationReport`
- `compile_options: CompileOptions`

Output:

- `CompileRun`

Purpose:

- Create a compile run record.
- Fix contract hash, snapshot id, base release id, workspace baseline commit, budget, and access rules.
- Refuse to start if the target check or ReferenceQualificationReport is not acceptable.
- Allow the base Release to have no lower-layer artifacts; in that case there are no inherited artifacts or prior release source files.

Used by:

- Compile Orchestration module.

### `load_target_workspace`

Input:

- `target_name: str`
- `contract_hash: str`
- `workspace_store: WorkspaceStore`

Output:

- `TargetWorkspace`

Purpose:

- Load or initialize the long-term agent-managed target workspace.
- Verify that its baseline commit matches the active target contract or has an explicit migration record.

Used by:

- `create_compile_run`
- `create_agent_workspace`

### `create_agent_workspace`

Input:

- `compile_run: CompileRun`
- `target_workspace: TargetWorkspace`
- `attempt_options: AgentAttemptOptions`

Output:

- `AgentAttempt`

Purpose:

- Clone or copy the target workspace baseline into an isolated attempt workspace.
- Initialize source control for the attempt workspace if needed.
- Create required top-level directories.
- Configure sandbox limits and filesystem mounts.

Used by:

- `launch_target_adaptation_agent`

### `advance_target_workspace_baseline`

Input:

- `target_workspace: TargetWorkspace`
- `closed_attempt: ClosedAgentAttempt`
- `release: Release | None`
- `accepted_candidate: Candidate | None`
- `reason: Literal["accepted_release", "explicit_carry_forward"]`

Output:

- `WorkspaceBaselineUpdate`

Purpose:

- Advance the long-term target workspace baseline to an attempt final commit.
- Require either an accepted Release for a Candidate produced by the closed attempt or an
  explicit carry-forward decision.
- When `reason == "accepted_release"`, hard-fail unless `release.candidate_id` matches
  `accepted_candidate.candidate_id` and the Candidate's `compile_id` and `attempt_id`
  match `closed_attempt`.
- Keep rejected attempts available for inspection without making them the next compile baseline by default.

Used by:

- Release Runtime after release approval.
- CLI/user carry-forward.

### `mount_readonly_inputs`

Input:

- `attempt: AgentAttempt`
- `target_view: TargetViewManifest`
- `train_view: TrainViewManifest`
- `base_release_view: ReleaseSourceView`
- `report_views: list[AgentVisibleReport]`
- `telemetry_summaries: list[AgentVisibleTelemetrySummary]`
- `protocol_docs: ProtocolDocs`

Output:

- `WorkspaceMountManifest`

Purpose:

- Mount or copy allowed read-only inputs into the workspace.
- When the base Release has no lower-layer artifacts, mount only release metadata and no inherited lower-layer artifact source.
- Mount only `AgentVisibleReport` summaries, never full `Report` records.
- Mount agent-visible telemetry summaries as aggregate, redacted context only.
- Ensure validation/test views, registry credentials, production secrets, and active evaluator code are absent.

Used by:

- `launch_target_adaptation_agent`

### `write_agent_brief`

Input:

- `attempt: AgentAttempt`
- `compile_run: CompileRun`
- `mount_manifest: WorkspaceMountManifest`
- `objective: CompileObjective`
- `agent_guidance: AgentSearchGuidance`
- `workspace_permissions: AgentWorkspacePermissions`

Output:

- `AgentBriefPath`

Purpose:

- Write a concise brief telling the agent the target, scorecard, budget, allowed files, forbidden actions, and candidate submission format.
- Render the objective, preferred search strategies/tools, extra user
  instructions, and whether network research or workspace-local dependency
  installation is allowed.
- Include that compile-time scaffolding and runtime source are both writable.
- Include mounted telemetry summaries as aggregate context only, not as raw production examples.
- Include that generated code must not launch another autonomous coding agent.

Used by:

- `launch_target_adaptation_agent`

### `launch_target_adaptation_agent`

Input:

- `attempt: AgentAttempt`
- `brief_path: Path`
- `agent_runtime: AgentRuntimeConfig`

Output:

- `AgentSessionHandle`

Purpose:

- Start the single target adaptation agent for this attempt.
- Stream logs and command events to the compile run store.
- Apply time, filesystem, network, process, and API broker restrictions. Network
  and workspace-local dependency installation remain disabled by default and are
  enabled only when the attempt permissions explicitly allow them. In the macOS
  profile, network access is allowed by omitting `(deny network*)`; workspace
  dependency installation is allowed only within the existing attempt workspace
  write surface.

Used by:

- `run_compile_loop`

### `run_compile_loop`

Input:

- `compile_run: CompileRun`
- `attempts: list[AgentAttempt]`
- `candidate_limit: int`
- `time_limit: Duration`
- `validation_feedback: Callable[[CandidateSubmission], AgentFeedback]`

Output:

- `CompileLoopResult`

Purpose:

- Let the agent work until budget, candidate count, or time boundary is reached.
- Receive submitted Candidates.
- Request Core validation feedback for each Candidate.
- Provide only aggregate feedback back to the same agent run.
- Stop before test evaluation.

Used by:

- Compile Orchestration module.

### `receive_candidate_submission`

Input:

- `attempt: AgentAttempt`
- `submission_path: Path`

Output:

- `CandidateSubmission`

Purpose:

- Register a submitted candidate directory.
- Record workspace commit and declared changed layers.
- Perform initial path escape checks before Artifact Worker validation.

Used by:

- Candidate Evaluation preflight.

### `provide_validation_feedback`

Input:

- `attempt: AgentAttempt`
- `feedback: AgentFeedback`

Output:

- `FeedbackDeliveryRecord`

Purpose:

- Write validation feedback into an agent-visible channel.
- Ensure feedback contains aggregate metrics, safe slice summaries, and requirement-check reasons only.
- Prevent raw validation/test inputs, expected outputs, request ids, or split indices from being exposed.

Used by:

- Candidate Evaluation module.

### `close_agent_attempt`

Input:

- `attempt: AgentAttempt`
- `reason: Literal["budget_exhausted", "candidate_limit", "time_limit", "user_stop", "ready_for_test", "failure"]`

Output:

- `ClosedAgentAttempt`

Purpose:

- Stop the agent session and all child processes.
- Capture final commit, diff summary, transcript, command log, and usage ledger.
- Freeze the workspace before test evaluation.

Used by:

- Candidate Evaluation module before test.
- CLI stop.

### `record_agent_usage`

Input:

- `attempt: AgentAttempt`
- `usage_event: AgentUsageEvent`

Output:

- `AgentUsageLedger`

Purpose:

- Record paid API/model calls launched by the compile experiment.
- Include target adaptation agent calls when they are paid API calls.
- Do not count the human-facing Codex session that is merely executing the plan.

Used by:

- Reports and budget checks.

### `write_agent_journal_entry`

Input:

- `attempt: AgentAttempt`
- `entry: JournalEntry`

Output:

- `JournalPath`

Purpose:

- Store human-readable hypotheses, changes, local evidence, submitted candidates, and remaining uncertainty.
- Use ordinary Markdown rather than a new research-log schema.

Used by:

- Agent workspace.
- Final reports.

### `collect_target_change_proposals`

Input:

- `attempt: AgentAttempt`

Output:

- `list[TargetChangeProposal]`

Purpose:

- Collect agent-written proposals under `proposals/`.
- Mark them as user-review items that can create a new target definition version.
- They do not affect the active compile run.

Used by:

- CLI/user review.

## Invariants

- One target adaptation agent per attempt.
- Multiple attempts are isolated full attempts, not L1/L2/L3 agents.
- Attempts start from the long-term target workspace baseline and do not advance it automatically.
- Only accepted Release work or explicit carry-forward can become the next workspace baseline.
- The agent can edit `scaffolding/` and `runtime/`, but not active target definition, evaluator, snapshot, registry, or telemetry.
- Generated scaffolding may call trainers, tuners, compilers, and Core brokers, but not another autonomous coding agent.
- User-enabled network research or workspace-local dependency installation does
  not grant validation/test rows, release authority, evaluator authority,
  registry credentials, production credentials, or direct reference/L4 broker
  access.
- Test evaluation starts only after the relevant agent attempt is closed.
