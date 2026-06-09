# replay 与 promotion 模块

模块：`darjeeling.compiler.replay`、`darjeeling.compiler.objective`

## 数据切分协议

MVP 使用三层 teacher-visible split：

```text
teacher_train
  给 L1 coding agent、L2 training、L3 prompt optimization、guard training 使用。
  generation-visible hard buffer 只能从这一 split 派生。

teacher_promotion_holdout
  只给外层 replay evaluator 使用。
  L1 coding agent 和 direct L4 proposal context 不可见。

teacher_regression_sample
  从较早 trace 抽样，用于避免 artifact 只优化最近窗口。
  MVP 中不进入 L1 coding-agent context。
```

最终 report 另外使用 MASSIVE gold：

```text
gold_eval
  只在 eval/report 阶段使用。
  不参与 compiler、promotion 或 guard training。
```

## 质量定义

Promotion 阶段的 correctness 以 `teacher_frame` 为准。Gold label 只用于最终 report。这避免弱层训练和 artifact promotion 直接接触 MASSIVE gold。

## Promotion 单位

MVP 采用 artifact set promotion：

```text
candidate artifact set = current artifacts + one or more candidate layer artifacts
```

整组 replay 后，如果系统 objective 改善并通过 gate，则提升整组 manifest。

## Promotion gate

```text
candidate_score > current_score
candidate_end_to_end_accuracy >= current_accuracy - epsilon
wrong_accept_rate <= configured_limit
candidate replay covers teacher_promotion_holdout + hard_buffer + teacher_regression_sample
```

其中 hard buffer 是 replay pressure，不是独立验收集。若 `teacher_promotion_holdout + teacher_regression_sample` 为空，即使 hard buffer 非空也不能 promotion。

## 当前开发切片

已实现 deterministic teacher-visible split 和第一版离线 replay gate：

- `teacher_train` 用于生成 L0 exact cache、训练 L2 student，以及为 L1 coding-agent job 提供 context。
- `teacher_promotion_holdout` 不进入 candidate generation，只用于 evaluator。
- `teacher_regression_sample` 从较早 trace 中抽样，并入 evaluator。
- hard buffer 从 `teacher_train` 中挖掘 `train_visible` cases，从 `teacher_promotion_holdout + teacher_regression_sample` 中挖掘 `replay_only` cases，写入同一个 `hard_buffer.jsonl`。
- L1 coding-agent 只可见 `train_visible` hard cases；promotion replay 在存在独立 replay coverage 时会把 `train_visible + replay_only` hard buffer 并入 evaluation traces。
- evaluator 对 current artifact set 与 candidate artifact set 运行 `L0 -> L1 -> L2 -> teacher fallback`。
- correctness 以 `teacher_frame` 为准，不读取 MASSIVE gold。
- objective、wrong accept gate、accuracy epsilon 和 per-layer regression gate 都会进入 promotion decision。
- rejected candidate 写 generation manifest，但不更新 `manifest.current.json`。
- L3 prompt candidate 已有显式 regenerated replay/promotion 路径：`edge-mvp l3 replay-prompt` 生成 `l3-prompt-replay-v1`，`edge-mvp l3 promote-prompt` 校验 prompt hash、accepted accuracy、wrong accept rate 和非空 coverage gate 后创建新的 promoted generation。

当前 evaluator 覆盖范围：

- L0 exact cache。
- L1 Rust ProgramBank。
- L2 student + guard。
- Recorded L3 accept：offline replay 不在 compiler 中加载本地模型，而是使用 trace 中已记录的 L3 result；若 recorded L3 accepted，则按 L3 弱层接收计入 correctness、latency、cost 和 wrong accept。
- Teacher fallback。

当前 evaluator 尚未覆盖：

- compiler 主循环内对新 `l3_prompt_candidate` artifact 的自动重新生成式 replay；因此 candidate 不会在 compile generation 中自动提升为 runtime `l3_prompt`。
- full cascade latency/cost 实测；当前使用固定 layer cost/latency estimates 作为 deterministic gate 的第一版近似。

当前 hard buffer 尚未覆盖：

- embedding cluster 或语义相似样本扩展。

## 单层 regression 记录

整组 promotion 的已知问题是可能掩盖单层 regression。当前主线先采用最小 per-layer hard gate：仍以 artifact set 为 promotion 单位，但如果某个弱层出现显著 accepted accuracy 下降、wrong accept 上升或 p95 latency 上升，默认拒绝整组 promotion。`PROMOTION_BLOCK_LAYER_REGRESSIONS=false` 只用于诊断或 ablation。

每次 promotion decision 必须输出 per-layer delta：

- coverage delta
- accepted accuracy delta
- wrong accept delta
- p50/p95 latency delta
- cost delta
- layer share delta

若某层显著 regression 且 gate 开启，promotion reason 必须标记：

```text
promotion_reason = "per-layer regression gate failed: L1, L2"
regressed_layers = [...]
```

若某层显著 regression 但整组仍被提升，说明这是显式诊断路径，report 标记：

```text
promoted_with_layer_regression = true
regressed_layers = [...]
```

## 遗留问题

后续仍可能引入：

- shadow promotion
- layer quarantine
- per-layer rollback
- Pareto frontier selection
- multi-objective promotion

这些不阻塞 MVP，但需要在 report 中保留足够信息支持后续设计。
