# artifacts 模块

模块：`darjeeling.artifacts`

## 职责

- 管理 candidate 和 promoted artifact。
- 保证原子写入。
- 记录 lineage。
- 支持 report 读取历史。

## 目录

```text
runs/<id>/artifacts/
  manifest.current.json
  generations/
    gen_000/
      manifest.json
      hard_buffer.jsonl
      candidate_metrics.csv
      promotion.json
      l0_cache.json
      l1/
        source/
        target/
        binary/
        l1_metrics.json
      l2/
        l2_config.json
        l2_student.joblib
        l2_guard.joblib
        target/
          target_l2.py
      guard/
        guard_candidate.json
      l3/
        l3_prompt.json
        l3_prompt.replay.json
        l3_prompt.candidate.json
      router_config.json
```

## Manifest

```json
{
  "artifact_set_id": "gen_001_abcd",
  "generation": 1,
  "parent_artifact_set_id": "gen_000_...",
  "created_at": "...",
  "schema_versions": {},
  "artifact_paths": {},
  "candidate_metrics": {},
  "per_layer_deltas": {},
  "promoted_with_layer_regression": false,
  "regressed_layers": [],
  "promoted": true,
  "promotion_reason": "objective improved within gates"
}
```

`candidate_metrics["promotion_block_layer_regressions"]` 记录该 generation 是否启用默认 per-layer regression gate。启用时，显著单层 regression 会拒绝整组 promotion，`promotion_reason` 写 `per-layer regression gate failed: ...`，`regressed_layers` 仍保留被阻塞的层用于诊断。

## Atomic promotion

1. 写入 generation 目录。
2. 完成 build/test/replay metrics。
3. 写 `manifest.json`。
4. 用 atomic replace 更新 `manifest.current.json`。

## L1 native artifact

## L2 target artifact

当 `edge-mvp l2 promote-target` 提升一个通过 target-evolve adoption gate 的候选时，manifest 同时记录：

- `artifact_paths["l2_student"]`: 用 target workspace 的 visible train split 和 target config 重新训练得到的 L2 bundle。
- `artifact_paths["l2_target"]`: copied target runtime module，通常是 `generations/gen_*/l2/target/target_l2.py`。

Runtime replay 和 compiler offline replay 都必须在加载 `l2_student` 后检查 `l2_target`。若存在，L2 运行时使用 target wrapper：先执行 bundle prediction，再执行 `postprocess_frame(...)`，再应用 guard accept，最后允许 `accept_prediction(...)` veto。`l2_target` 不进入 Darjeeling core；它是 target-dependent artifact。

Promoted manifest 必须保留 `candidate_metrics["l2_target_loop_cadence"]` 和 `candidate_metrics["l2_target_code_policy"]`。前者证明 target rounds 运行在固定 trace snapshot 上，不等待新的 outer stream prefix；后者证明 target-specific code 的合法边界是 `target/` + visible train/validation folds，而不是 Darjeeling core 的 dataset-independent 规则。

## Hard buffer artifact

`hard_buffer.jsonl` 使用 `hard-buffer-v1` schema。每条 hard case 必须包含 `visibility`：

- `train_visible`: 可进入 L1 coding-agent context、L2/guard training 相关上下文和 replay pressure。
- `replay_only`: 只能进入 promotion replay evaluator，不能进入任何 candidate-generation prompt/context。

该字段用于避免 holdout/regression-derived hard cases 被反向塞进 prompt，同时保留 replay pressure。

## L1 native artifact

L1 artifact 必须保存：

- Rust source snapshot。
- Cargo.lock。
- build profile。
- binary 或可重建说明。
- native benchmark 结果。
- replay metrics。
- Codex CLI transcript/diff/provenance。

这让后续 report 能解释“哪个 native program 被提升，以及为什么”。

L1 coding-agent artifact 中的 `provenance.json` 使用 `l1-agent-provenance-v1` schema，保存 mode、return code、路径、Codex event type summary、外层命令摘要和 diff stats。Raw transcript 不被 provenance 替代，仍应作为单独 artifact 保留。

`reports/l1_benchmark.json` 是 report-time 观测产物：它可以复用 generation 内的 native benchmark，也可以在 report 阶段对当前 promoted/default L1 crate 重新跑一次 worker benchmark。若要做跨 generation benchmark 曲线，应把每代 benchmark 结果同时归档到 generation artifact 中。

当 L1 coding-agent 成功产出 candidate crate 时，compiler 会在 generation 目录写 `l1/l1_benchmark.json` 并在 manifest 中记录 `artifact_paths["l1_benchmark"]`。这是跨 generation L1 worker benchmark 表格的数据源。

`reports/l3_benchmark.json` 是显式 preflight 观测产物，由 `edge-mvp l3 bench --out ...` 写入。它记录本地 SLM 单配置 benchmark 的 status/error、backend、actual device、load/generation latency、parse/repair、would-accept 和 throughput。Report 只读取该文件，不自动加载本地模型。

## L3 mode artifact

Artifact manifest 必须记录 L3 mode：

```text
disabled | shadow | guarded
```

若 L3 disabled 或 shadow，report 中不能把 L3 计为主路由覆盖贡献。

若 manifest 包含 `artifact_paths["l3_prompt"]`，runtime 必须加载该 `L3PromptArtifact` 作为 L3 prompt；若缺失，则使用 settings/default prompt。

`artifact_paths["l3_prompt_candidate"]` 只表示 legacy L4 direct API 生成的候选 prompt artifact，不能被 runtime 自动加载。Compiler 只有在 L3 prompt candidate 能被 regenerated replay 或 shadow replay 评估后，才应把它提升为 runtime `l3_prompt`。

`edge-mvp l3 prompt-evolve` 生成的 prompt candidate snapshot 存在该 job 的 `candidates/candidate_l3_prompt.json` 中。它同样不能绕过 replay/promotion；只有 outer harness 的 visible validation、private selection、private promotion 和后续显式 promotion 通过后，runtime manifest 才能记录为 `artifact_paths["l3_prompt"]`。

`artifact_paths["l3_prompt_replay"]` 保存 `l3-prompt-replay-v1` promotion 证据。该 artifact 由显式 `edge-mvp l3 replay-prompt` 生成，并由 `edge-mvp l3 promote-prompt` 校验后归档到 generation 目录。Replay artifact 必须包含 `prompt_version` 和 prompt canonical JSON 的 `prompt_sha256`，promotion 时必须与待提升的 `L3PromptArtifact` 匹配。Promotion manifest 必须把 replay 聚合指标写入 `candidate_metrics`，但不应把 replay request 明细塞进后续 L4 candidate-generation context。
