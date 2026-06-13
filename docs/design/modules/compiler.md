# compiler 模块

模块根：core `darjeeling.compiler` 目前只保留 package boundary；concrete NLU
compiler 已迁到 `darjeeling.targets.nlu.compiler.*`，并通过
`NluTargetCompiler.propose_artifacts(...)` 接入 target compiler entry。

本页的 L1/L2/L3 训练、intent/slot mining、frame objective 和 prompt evolution
描述的是 NLU target compiler，不是 target-independent core 默认语义。

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
5. 调用 target compiler 提供的 mining/diagnostic/proposal 入口。
6. 构造 L0 candidates。
7. 运行 L1 coding-agent compiler job，产生 Rust candidate artifact。
8. 请求 L4 direct API 提议 L2 config candidates。
9. 训练 L2 candidates。
10. 可选请求 legacy L4 direct API 提议 L3 prompt candidates。
11. 根据 L3 mode 决定 disabled/shadow/guarded 评估路径。
12. 搜索 guard thresholds。
13. 组合 candidate artifact sets。
14. Replay。
15. Promotion decision。
16. 写 artifact lineage 和 metrics。

## 当前开发切片

已实现最小可运行 compiler generation 与离线 promotion gate：

- `compile_every` 到达时，从当前 run 的 `traces.jsonl` 构造 `TeacherTrace` 视图。
- compiler 输入会移除 hidden gold label，并执行 teacher-visible-only 检查。
- 按 deterministic split 切分 `teacher_train`、`teacher_promotion_holdout` 和小规模 `teacher_regression_sample`。
- 从 current manifest 的上一版 hard buffer、本轮 `teacher_train`、以及 replay-only evaluation split 中合并 hard buffer，记录 weak wrong accept、L4 fallback、slow path 等 teacher-visible 困难样本。
- 只用 `teacher_train` 生成 L0 exact cache artifact、训练 L2 student artifact，并在 `L1_AGENT_MODE` 非 disabled 时运行 L1 coding-agent job；L1 agent 只能看到 `teacher_train` 和 `visibility=train_visible` hard cases。
- L1 coding-agent 成功生成 candidate crate 后，compiler 会运行小型 Rust worker benchmark，写入 generation-scoped `l1/l1_benchmark.json` 并记录到 manifest。
- `L4_PROPOSAL_MODE=live` 时，compiler 通过 `L4ProposalAdapter` 请求 bounded L2 config proposal；默认 disabled，失败时记录错误并回退 deterministic config。
- `L2_ENABLED=false` 时，compiler 不训练、不写入 L2 candidate artifact，并从 candidate manifest 中移除 inherited `l2_student`。
- `L2_GUARD_MODE=always_accept` 时，compiler 仍训练 L2 student，但跳过 learned-threshold search，将 threshold 固定为 0。`experiment no-guard` 是隔离的诊断性 ablation，会设置 `FORCE_PROMOTE_ARTIFACTS=true`，使无 guard 的 L2 artifact 能实际进入 runtime 并暴露真实错误率；主实验不使用这个设置。
- `L4_PROPOSAL_MODE=live` 时，compiler 请求 bounded guard search proposal，写入 `guard/guard_candidate.json`，并用 proposal 中的 threshold grid 与 wrong-accept 上限驱动 deterministic local search。
- L2 训练 scope 由 `L2_TRAINING_SCOPE` 决定：默认 `teacher_train`，实验开关 `lower_miss` 只使用本轮 `teacher_train` 中 L0/L1 未接收的 traces。Scope 定义 L2 可见训练分布，不等同于最终 validation 口径。
- Optuna internal validation 和 final guard calibration 都优先使用 residual validation：从 train window 中做内部 split，用 train prefix 模拟 L0 exact cache，并过滤 validation 中 exact repeat 与已记录 L0/L1 accepted 请求。这样 tuning/threshold 优化的是“真实会到达 L2 的 future residual”，而不是 L2-only 或 already-covered 样本。
- L2 训练后执行 deterministic guard threshold grid search，选择满足 wrong-accept 上限且覆盖率最高的 threshold，并写入 L2 artifact 与 candidate metrics。若 residual calibration 不可用，compiler 回退到 training-scope search 并记录 `l2_guard_calibration.fallback_reason`。
- L2 target-dependent runtime changes do not run through compiler generation of core artifacts. 显式 `edge-mvp-nlu l2 promote-target` 会在 manifest 中写入 `artifact_paths["l2_target"]`；compiler offline replay 加载 current/candidate artifact set 时必须与 runtime replay 一样应用 target wrapper。若 compiler 在普通 generation 中重新训练 core L2 bundle，但没有做 target-aware adoption，则必须删除继承的 `l2_target` path，并在 candidate metrics 中记录 `l2_target_dropped_reason`，避免旧 target code 与新 bundle 混用。
- `L4_PROPOSAL_MODE=live` 时，compiler 也会请求 legacy bounded L3 prompt proposal，展开 teacher-visible few-shot trace IDs，并写入 `l3/l3_prompt.candidate.json`。该 candidate 记录为 `l3_prompt_candidate`，不会自动成为 runtime `l3_prompt`；当前真实 L3 prompt evolve 主路径是显式 `edge-mvp-nlu l3 prompt-evolve`。
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

其中 recorded L3 accept 使用 trace 中已经记录的 L3 result，不在 compiler 中加载本地模型；teacher fallback 直接使用 evaluator 可见的 teacher label，用于评估候选弱层是否安全吸收。这个切片已经移除了无条件 bootstrap promotion，并把 L1 candidate 纳入 gate，但仍不是 proposal 的完整 replay evaluator。

非阻塞后续项：

- embedding clusters、更复杂的 guard feature/search family。
- compiler 主循环内自动对 `l3_prompt_candidate` 做本地 SLM 重新生成式 replay 并参与 artifact-set promotion；当前显式 CLI 路径已经支持 `l3 prompt-evolve`、prompt replay 和 prompt promotion，但 compile generation 不自动加载本地模型。
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
teacher_train or lower_miss subset
-> internal train/validation split, default chronological future-like holdout
-> Optuna trials over L2StudentConfig
-> l2/l2_tuning.json
-> train final L2 candidate on selected L2 training scope with best config
-> deterministic guard search
-> outer replay promotion gate
```

Optuna tuning 不能读取 `teacher_promotion_holdout`、gold eval 或 future stream。默认 `L2_TUNING_SPLIT_POLICY=chronological`，用 teacher-visible training window 的尾部做 future-like validation；`stratified_random` 只作为 ablation。Optuna 产出的 best config 仍只是 candidate 输入，不能绕过外层 replay。

`L2_TUNING_MIN_EXAMPLES` 控制 Optuna 的最低样本数。样本不足时 compiler 跳过 tuning、记录 `l2_tuning_skipped_reason`，但仍按 deterministic config 训练 L2 candidate，以保证 generation 继续产生可比较 artifact。

## Objective

默认 score：

```text
+ 100.0 * correctness
- 200.0 * wrong_accept_rate
-   1.0 * cost_usd_per_100_requests
-   0.01 * p95_latency_ms
-   0.001 * artifact_complexity
```

`artifact_complexity` 不是原始源码行数或原始数据行数，而是每类 artifact
定义的复杂度单位：

- L0 exact cache 贡献 `sqrt(cache_entries)`。它是简单查表的数据 artifact，
  维护风险随规模增长但不应按每条 cache line 线性惩罚，否则会拒绝低风险的
  L4 call elimination。
- L1 Rust programbank 存在时贡献 1 个 artifact unit。
- L2 student bundle 存在时贡献 1 个 artifact unit。

权重可配置，并写入 run report。

## Promotion gate

```text
candidate_score > current_score
candidate_end_to_end_accuracy >= current_accuracy - epsilon
wrong_accept_rate <= configured_limit
candidate replay covers recent holdout + hard buffer + older trace sample
no significant per-layer regression, unless explicitly disabled for diagnosis
```

L4、Codex CLI agent、local tests 都不能绕过这个 gate。

## Promotion 粒度与遗留问题

MVP promotion 单位是 artifact set。单层 candidate 可以单独度量，但 live manifest 以整组更新。

当前主线已把最小 per-layer hard gate 接入 promotion decision。Compiler 必须输出 per-layer delta；若某个仍在使用或使用占比上升的层出现显著 accepted accuracy 下降、wrong accept 上升或 p95 latency 上升，`PROMOTION_BLOCK_LAYER_REGRESSIONS=true` 时拒绝整组 promotion，并把 `regressed_layers` 写入 promotion record。若上层 share 下降是因为请求被更低层正确吸收，该上层不因自身样本减少而触发 regression。`promoted_with_layer_regression=true` 只应出现在显式关闭该 gate 或 `FORCE_PROMOTE_ARTIFACTS=true` 的诊断实验中。

仍然保留为遗留问题的是更细粒度的恢复策略：当前 gate 能阻止被掩盖的退化，但 live manifest 仍按 artifact set 更新，尚未支持 shadow promotion、layer quarantine 或 per-layer rollback。
