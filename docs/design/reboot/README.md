# Darjeeling Architecture Design

This directory contains the active architecture design for Darjeeling. The
implementation in `src/darjeeling/` is expected to follow these documents.

The directory name is historical. Do not treat `reboot` as a user-facing product
concept or a separate runtime mode.

The design assumes:

- Darjeeling core is target-independent.
- Users provide and review a small target definition.
- The system can cold start with a Release that has no lower-layer artifacts before any compile succeeds.
- The first lower-layer compile is a `RecompileRequest`, not a separate cold-start compile path.
- Core launches one target adaptation agent per compile attempt.
- That agent owns both compile-time scaffolding and L1/L2/L3 runtime source inside an isolated workspace.
- Core alone owns train/validation/test boundaries, official evaluation, release, runtime fallback, trace, audit, and rollback.
- L1/L2/L3 are runtime artifact levels behind one `accept`/`abstain` worker protocol.
- Cache is a runtime optimization, not an L0 capability layer.

## Documents

- [00 Overall Design](00_overall_design.md)
- [01 Target Definition Module](modules/01_target_definition.md)
- [02 Snapshot And Reference Module](modules/02_snapshot_reference.md)
- [03 Agent Workspace Module](modules/03_agent_workspace.md)
- [04 Artifact Worker Module](modules/04_artifact_worker.md)
- [05 Candidate Evaluation Module](modules/05_candidate_evaluation.md)
- [06 Release Runtime Module](modules/06_release_runtime.md)
- [07 Runtime Feedback Overview](modules/07_runtime_feedback_overview.md)
- [08 Runtime Trace And Metrics Module](modules/08_runtime_trace_metrics.md)
- [09 Audit Monitoring Module](modules/09_audit_monitoring.md)
- [10 Telemetry Evidence And Recompile Module](modules/10_telemetry_evidence_recompile.md)
- [11 Compile Orchestration Module](modules/11_compile_orchestration.md)

## Review Discipline

When these documents are changed, update them as a set:

1. Start from the overall module map and data flow.
2. Define module-level function inputs and outputs.
3. If a module-level document changes an input, output, function name, or ownership boundary, update the overall design.
4. Re-check the design against the source feedback file before considering the change complete.

The main drift risks are:

- Reintroducing one Agent per layer.
- Letting target-specific code move back into core.
- Letting an artifact call L4 or the next layer.
- Counting cache hits as L1/L2/L3 coverage.
- Letting the agent decide validation/test data, correctness, or release.
- Turning compile-time scaffolding into deployed runtime code.
