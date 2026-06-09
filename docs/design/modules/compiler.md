# compiler 模块

模块根：`darjeeling.compiler`

## 职责

- 每 K requests 从 trace 中取 teacher-visible data。
- 生成候选 artifact。
- 调用 replay evaluator。
- 只在通过 promotion gate 后更新 live artifact set。

## Generation 流程

1. 加载 current artifact manifest。
2. 加载 recent teacher traces。
3. 按 split policy 切分 `teacher_train`、`teacher_promotion_holdout`、`teacher_regression_sample`。
4. 从 candidate-generation 可见 split 构建 hard buffer。
5. Mining hot utterances、hot intents、embedding clusters、failure cases。
6. 构造 L0 candidates。
7. 运行 L1 coding-agent compiler job，产生 Rust candidate artifact。
8. 请求 L4 direct API 提议 L2 config candidates。
9. 训练 L2 candidates。
10. 请求 L4 direct API 提议 L3 prompt candidates。
11. 根据 L3 mode 决定 disabled/shadow/guarded 评估路径。
12. 搜索 guard thresholds。
13. 组合 candidate artifact sets。
14. Replay。
15. Promotion decision。
16. 写 artifact lineage 和 metrics。

## 当前开发切片

已实现最小可运行 compiler generation 与离线 promotion gate：

- `compile_every` 到达时，从当前 run 的 `traces.jsonl` 构造 `TeacherTrace` 视图。
- compiler 输入会移除 `gold_frame`，并执行 teacher-visible-only 检查。
- 按 deterministic split 切分 `teacher_train`、`teacher_promotion_holdout` 和小规模 `teacher_regression_sample`。
- 从 current manifest 的上一版 hard buffer、本轮 `teacher_train`、以及 replay-only evaluation split 中合并 hard buffer，记录 weak wrong accept、L4 fallback、slow path 等 teacher-visible 困难样本。
- 只用 `teacher_train` 生成 L0 exact cache artifact、训练 L2 student artifact，并在 `L1_AGENT_MODE` 非 disabled 时运行 L1 coding-agent job；L1 agent 只能看到 `teacher_train` 和 `visibility=train_visible` hard cases。
- L1 coding-agent 成功生成 candidate crate 后，compiler 会运行小型 Rust worker benchmark，写入 generation-scoped `l1/l1_benchmark.json` 并记录到 manifest。
- `L4_PROPOSAL_MODE=live` 时，compiler 通过 `L4ProposalAdapter` 请求 bounded L2 config proposal；默认 disabled，失败时记录错误并回退 deterministic config。
- `L2_ENABLED=false` 时，compiler 不训练、不写入 L2 candidate artifact，并从 candidate manifest 中移除 inherited `l2_student`。
- `L2_GUARD_MODE=always_accept` 时，compiler 仍训练 L2 student，但跳过 learned-threshold search，将 threshold 固定为 0。`experiment no-guard` 是隔离的诊断性 ablation，会设置 `FORCE_PROMOTE_ARTIFACTS=true`，使无 guard 的 L2 artifact 能实际进入 runtime 并暴露真实错误率；主实验不使用这个设置。
- `L4_PROPOSAL_MODE=live` 时，compiler 请求 bounded guard search proposal，写入 `guard/guard_candidate.json`，并用 proposal 中的 threshold grid 与 wrong-accept 上限驱动 deterministic local search。
- L2 训练后在 `teacher_train` 上执行 deterministic guard threshold grid search，选择满足 wrong-accept 上限且覆盖率最高的 threshold，并写入 L2 artifact 与 candidate metrics。
- `L4_PROPOSAL_MODE=live` 时，compiler 也会请求 bounded L3 prompt proposal，展开 teacher-visible few-shot trace IDs，并写入 `l3/l3_prompt.candidate.json`。该 candidate 记录为 `l3_prompt_candidate`，不会自动成为 runtime `l3_prompt`。
- 用 `teacher_promotion_holdout + teacher_regression_sample + hard_buffer` 对 current artifact set 与 candidate artifact set 做离线 replay；hard buffer 包含 `train_visible` 与 `replay_only` 两类 replay pressure，但不能替代独立 holdout，若没有 holdout/regression 覆盖则拒绝 promotion。
- promotion 使用 `decide_artifact_set_promotion`，检查 objective、accuracy epsilon 和 wrong-accept 上限。
- 若 `FORCE_PROMOTE_ARTIFACTS=true`，compiler 仍先计算正常 promotion decision，并在 candidate metrics 中记录 `force_promote_original_reason`，随后强制 promotion。该开关只用于诊断实验，不属于主线 evolution 策略。
- 每个 generation 写 `hard_buffer.jsonl`、`candidate_metrics.csv`、`promotion.json` 和 `manifest.json`。
- 候选即使被拒绝，也写入 `artifacts/generations/gen_*/manifest.json`；只有 gate 通过才更新 `manifest.current.json`。
- replay 在 promotion 后重新加载 L0/L1/L2，使下一窗口能实际使用 promoted artifacts。

当前离线 replay gate 的范围：

```text
L0 -> L1 -> L2 -> recorded L3 accept -> teacher fallback
```

其中 recorded L3 accept 使用 trace 中已经记录的 L3 result，不在 compiler 中加载本地模型；teacher fallback 直接使用 evaluator 可见的 `teacher_frame`，用于评估候选弱层是否安全吸收。这个切片已经移除了无条件 bootstrap promotion，并把 L1 candidate 纳入 gate，但仍不是 proposal 的完整 replay evaluator。

非阻塞后续项：

- embedding clusters、更复杂的 guard feature/search family。
- compiler 主循环内自动对 `l3_prompt_candidate` 做本地 SLM 重新生成式 replay 并参与 artifact-set promotion；当前显式 CLI 路径已经支持 L3 prompt replay/promotion，但 compile generation 不自动加载本地模型。
- 多实验对比已有表格、rank 和 bottleneck summary；趋势图、显著性区间和更完整的 plot/report 页面作为后续增强。
- L1 worker benchmark 已有跨 generation 表格；Criterion/cargo bench microbenchmark 仍未接入。

## Hard buffer 设计

MVP hard buffer 是 `hard-buffer-v1` JSONL，记录：

- `request_id`
- primary `reason`
- all `reasons`
- `severity`
- `chosen_layer`
- `total_latency_ms`
- teacher-visible `trace`

当前 reason 集：

- `weak_wrong_accept`
- `teacher_final_mismatch`
- `fallback_after_weak_abstain`
- `l4_fallback`
- `slow_path`

当前 hard buffer 会跨 generation 持久化：compiler 读取 current manifest 中的 `artifact_paths["hard_buffer"]`，与本轮 `teacher_train` 新挖掘出的 hard cases 合并，按 severity 去重截断后写入新 generation。

Hard buffer 的可见性分两层：

- generation-visible buffer：只从 `teacher_train` 派生，可以进入 L1 coding-agent context。
- replay-only buffer：从 `teacher_promotion_holdout + teacher_regression_sample` 派生，只能进入 evaluator，不能进入 L1/L2/L3 candidate-generation context。
- replay pressure：在存在独立 replay coverage 时，将两类 hard buffer 并入 evaluator，让候选不要牺牲已知困难样本。

每条 hard case 都带 visibility class：`train_visible` 或 `replay_only`。Compiler 持久化两类 hard cases，但调用 L1 coding-agent 和 direct L4 proposal 时只传入 `train_visible` 数据，避免把 holdout 或最终 eval 信息反向塞进 prompt。

## L1 compiler job

L1 compiler 不生成 DSL。它启动 `L4CodingAgentAdapter`：

```text
current L1 Rust source tree
+ teacher-visible traces
+ hard cases
+ objective/gates
+ allowed commands
-> Codex CLI agent edits Rust source
-> cargo test / cargo bench / local replay
-> candidate source snapshot + binary + report
```

Agent 产出的 artifact 仍必须经过外层 replay。

## L2/L3/guard compiler jobs

这些 job 的 L4 参与方式分层：

- L2: 轻量路径仍支持 bounded config JSON；主 evolve 路径是 L4 coding agent 修改 L2 代码/search space，并调用 Optuna 等本地工具做超参搜索。
- L3: prompt artifact JSON，few-shot 只能引用 teacher trace IDs。
- Guard: threshold/search proposal，最终通过 deterministic grid search/replay 选择。

L2 Optuna tuning 在 compiler 中是显式开关：

```text
L2_TUNING_MODE=optuna
teacher_train
-> internal train/validation split
-> Optuna trials over L2StudentConfig
-> l2/l2_tuning.json
-> train final L2 candidate on full teacher_train with best config
-> deterministic guard search
-> outer replay promotion gate
```

Optuna tuning 不能读取 `teacher_promotion_holdout`、gold eval 或 future stream。它产出的 best config 仍只是 candidate 输入，不能绕过外层 replay。

## Objective

默认 score：

```text
+ 100.0 * frame_exact_match
- 200.0 * wrong_accept_rate
-   1.0 * cost_usd_per_100_requests
-   0.01 * p95_latency_ms
-   0.001 * artifact_complexity
```

权重可配置，并写入 run report。

## Promotion gate

```text
candidate_score > current_score
candidate_end_to_end_accuracy >= current_accuracy - epsilon
wrong_accept_rate <= configured_limit
candidate replay covers recent holdout + hard buffer + older trace sample
```

L4、Codex CLI agent、local tests 都不能绕过这个 gate。

## Promotion 粒度与遗留问题

MVP promotion 单位是 artifact set。单层 candidate 可以单独度量，但 live manifest 以整组更新。

已知遗留问题：整组 promotion 可能掩盖单层 regression。Compiler 必须输出 per-layer delta，并将 `promoted_with_layer_regression` 写入 promotion record。后续可引入 per-layer hard gates、shadow promotion 或 layer quarantine。
