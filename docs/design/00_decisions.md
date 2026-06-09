# 00 用户决策与优先级

本文档记录已经对齐的顶层决策。后续 Codex session 看到它们与 proposal 冲突时，应优先遵守本文档。

## 决策 1：L1 使用 Rust native CPU program

**状态：用户决策。**

L1 从第一版开始使用 Rust 编写的 native CPU ProgramBank。Python 不能作为真实 L1 backend，因为 Python latency 无法证明“hot path 被编译成便宜 CPU program”的 thesis。

设计含义：

- L1 真实指标以 Rust binary 或 Rust native worker 为准。
- Python 只负责 orchestration、JSONL I/O、测试驱动或报告。
- L1 report 至少区分：
  - `native_inner_latency`: Rust 内部处理单条 request 的计时。
  - `integration_latency`: Python runtime 调用 L1 worker 的端到端计时。
- MVP 初期可使用 long-lived Rust subprocess，避免 Python extension ABI 复杂度。

## 决策 2：L1 主路径不是 DSL

**状态：用户决策。**

L1 不以 DSL 作为架构主路径。原 proposal 中 DSL 可以作为一种辅助表示，但不能限制 L1 的实现形态。

设计含义：

- L1 artifact 是一个 agent-maintained Rust source tree。
- 允许实现 if/else tree、tight loop、trie、perfect hash、regex automata、小型 state machine、表驱动 matcher 等。
- 代码可以较大，可以存在 dead code 或错误分支；质量由 replay/evaluator 和 benchmark 决定。
- 系统不要求 L4 或外层 compiler 完全理解每个分支。
- 外层只要求固定 I/O contract、可构建、可测试、可 replay、可测 latency。

## 决策 3：L4 evolve L1 时就是 coding agent

**状态：用户决策。**

L4 是一个层次，不只是一次模型 API call。L4 在 L1 compiler mode 下通过 Codex CLI coding-agent harness 运行。强模型位于 coding agent 内部，利用 agent 的多轮对话、局部读写文件、运行构建和测试的能力来 evolve L1 Rust 代码。

这不是以下形态：

```text
L4 model -> 写一条自然语言指令 -> 另一个 agent 执行
```

而是：

```text
L4CodingAgentAdapter -> 启动 Codex CLI -> 在隔离工作区内修改 L1 Rust source tree
```

设计含义：

- L1 evolution job 内部允许多轮 agent session。
- session 只在一个 compiler generation/job 内有效。
- agent transcript、diff、测试命令、benchmark 结果都要落盘。
- agent 产物仍然只是 candidate，不能 self-certify。
- 外层 replay evaluator 仍是唯一 promotion authority。

## 决策 4：Direct L4 API calls 不使用长期多轮 session

**状态：设计决策。**

Teacher/fallback、L2 config proposal、L3 prompt proposal、guard proposal 不维护长期对话 session。每次调用由代码确定性组装 prompt/context。

设计含义：

- teacher 不看历史，只看 task schema 和当前 utterance。
- L2/L3/guard proposal 可以看 teacher-visible 历史摘要，但不是 session transcript。
- retry/repair 是显式 stateless request，不依赖隐藏对话状态。
- 每次 request 都记录 prompt version、context hash、source trace ids、usage 和 cached tokens。

## 决策 5：Prompt prefix 稳定性是一等目标

**状态：设计决策。**

Direct L4 API prompt 必须采用 stable-prefix + dynamic-tail 布局，以最大化 prompt/KV cache 命中。

设计含义：

- 静态 instructions、schema、约束放在 prompt 前缀。
- 动态 trace、hard cases、当前任务放在末尾。
- block 顺序确定性排序。
- 超 token budget 时从低优先级动态 block 的尾部裁剪，不截断 stable prefix。
- 记录 `cached_tokens` 并在 report 中展示 prompt cache 效果。

## 决策 6：Promotion 先采用整组提升，但记录单层 regression 遗留问题

**状态：设计决策，带遗留问题。**

MVP promotion 单位先采用 artifact set，而不是单层 artifact。候选 L0/L1/L2/L3/guard 组合后统一 replay，整组 objective 变好才提升。

设计含义：

- 单层 candidate 可以单独生成和单独度量。
- live manifest 更新以 artifact set 为单位。
- promotion report 必须记录每一层的 coverage、accepted accuracy、wrong accept、latency delta。
- 如果整组提升但某一层显著 regression，必须在 report 中显式标红。

遗留问题：

整组 promotion 可能掩盖单层 regression。例如 L1 退化但 L2/L4 fallback 补偿后系统分数仍上升。后续可能需要更复杂的 promotion 设计，例如 per-layer quarantine、shadow promotion、layer-level regression gates、Pareto frontier 或分层回滚。

## 决策 7：L3 必须真实，但必须可开关，不阻塞主 demo

**状态：用户确认后的设计决策。**

L3 local SLM 是 proposal 的重要层级，但本地模型可能慢，且本地硬件适配不稳定。设计必须允许 L3 disabled、shadow 或 guarded-enabled，使主 demo 至少不因本地硬件问题被阻塞。

设计含义：

- L3 不能被 mock；启用时必须真实加载 local SLM。
- 开发和硬件不足环境允许 `local_slm.enabled=false`。
- 禁用 L3 时 cascade 变为 `L0 -> L1 -> L2 -> L4`，report 必须明确标注。
- L3 可先以 shadow 模式收集指标，只有 replay 证明有价值才进入 guarded accept。
- 本地硬件适配作为遗留问题记录，不把它变成 MVP 阻塞项。

## 决策 8：Teacher 一致性不依赖 temperature

**状态：用户确认后的设计决策。**

不要把 temperature 当作 teacher 一致性的核心机制。现代 LLM/服务端实现未必严格吃这个参数，且不同 provider 行为可能不同。

设计含义：

- 可以在支持的 API 上设置低随机性参数，但不能把它作为稳定性保证。
- Teacher 稳定性主要来自 cache key、schema version、prompt version、model、validation 和 replay。
- 同一 teacher cache key 命中时不刷新。
- prompt/schema/model 变化进入新的 cache namespace。

## 决策 9：L2 evolve 拆成 coding-agent 结构改造与 Optuna 调参

**状态：用户决策。**

L2 的演化不应让 L4 模型手工猜超参。L4 coding agent 负责需要 generalized intelligence 的工作：修改 L2 代码、设计特征、模型家族、校准方法、accept policy、验证协议和 Optuna search space。Optuna 或同类本地工具负责在已定义 search space 内做局部超参搜索。

设计含义：

- 调参是本地、可复现、可审计的工具调用，不消耗 L4 token 做人工搜索。
- L4 coding agent 可以自由调用 `edge-mvp l2 tune` 或等价 Python API，并读取 tuning report。
- Optuna 不能直接优化最终 e2e test；compiler 中的 tuning 只使用 `teacher_train` 内部切分出的 validation，不读取 promotion holdout、gold eval 或 future stream。
- `L2_TUNING_MODE=optuna` 是显式开关，默认关闭，避免普通 replay 默认变慢。
- 旧的 direct L4 bounded L2 config proposal 只保留为轻量 proposal path；它不能取代 coding-agent 级别的 L2 code evolution。

## 决策 10：L0 第一阶段只要求 exact cache

**状态：设计决策。**

Semantic cache 是有价值的，但第一阶段不强制启用。先让 exact cache、Rust L1、teacher/replay/promotion pipeline 稳定，再加入 embedding/FAISS 和 threshold calibration。

设计含义：

- MVP 第一阶段 L0 exact cache 必须实现。
- Semantic cache 作为第二阶段 artifact。
- Report 中区分 exact cache 贡献和 semantic cache 贡献。

## 决策 11：L2 evolve 使用 Target-dependent inner loop，不能改 Darjeeling core

**状态：用户决策。**

L2 evolve 分为 Outer Darjeeling loop 和 Inner L2 target-evolution loop。Darjeeling core 保持 dataset-independent；具体 target/dataset 的 L2 runtime code 放在隔离 target workspace，由 L4 coding agent 多轮修改、训练和评估。

设计含义：

- Outer Darjeeling loop 负责 replay、teacher-visible split、workspace/provenance、promotion holdout、artifact registry 和 core invariants。
- Inner L2 target loop 在固定 workspace 内运行，不等待新的 stream prefix，也不受 `compile_every` 限制。
- `target/` 是唯一可写 target-dependent code 区域；`system/darjeeling/` 是只读 core/evaluator copy。
- 每轮 mutating command 后、candidate evaluation 前必须做 workspace scope check：候选代码只能改 `target/`；`runs/` 只允许作为 scratch output；`data/`、`tools/`、`system/darjeeling/` 和 `program.md` 是 protected surface。越界写入直接停止该 target-evolve job，不能参与 selection 或 promotion。
- `data/train.jsonl`、`data/inner_validation.jsonl` 和可选的 `data/inner_validation_shadow_*.jsonl` 可以给 agent 读；selection holdout 和 promotion holdout 存在 outer job 的私有目录，不进入 agent workspace。
- Target split 默认保持 chronological；小样本或窄 target patch 诊断可以显式使用 `intent-stratified`，让 visible validation、selection 和 promotion 都覆盖更多 intent family。该策略只改变 fixed target split 的采样方式，不放宽 visible validation、private selection、private promotion 或 outer replay gates。
- `--visible-validation-folds N` 是 agent-visible validation pressure 开关。`N>1` 会创建额外可见 `inner_validation_shadow_*` folds 并用 aggregate visible validation gate 做 candidate selection；它不把 private holdout rows 或 aggregates 暴露给 L4 agent。
- 调参不应占用 L4 coding-agent round。Inner loop 先在固定 target data 上通过 `local-search`/Optuna 做大量 cheap trials，搜索结果只写 `target/config.json`；L4 coding agent 主要负责修改 `target/` 中真正需要设计判断的代码、特征、后处理和 search space。
- Target-specific lexical rules、state machines、feature code 或 model code 可以存在于 `target/`，只由 visible validation、target holdout/promotion 指标和 outer replay 决定是否采用；不能仅因为它是 target-dependent 就拒绝。Darjeeling core 仍必须保持 dataset-independent。
- Inner loop 必须先评估 baseline，再评估 target rounds；可见 round history 只包含 visible validation 聚合，不包含 selection/promotion holdout 聚合。
- L4 agent budget 由 outer harness 控制：`rounds` 是最大 target round 数，不是数据收集次数；默认不会因为 private selection gate 通过而中止 inner loop，selection/promotion 只参与最终 candidate selection/adoption。真实固定 snapshot 探索使用 `fixed-inner` budget profile，其默认 live `codex-cli` cap 高于 smoke/standard；若做 smoke 或节省 live LLM cost，可使用 `smoke` profile、`--max-agent-rounds 0` 或显式打开 selection-gate early stop。
- Candidate selection gate 要求 visible validation gate 和 private selection holdout gate 同时通过；private selection 不能掩盖 visible validation regression。
- Visible validation improvement 不能直接触发采用；通过 candidate selection gate 的 target round 只表示可被选中，只有同时通过 private promotion holdout gate 才能进入 `adoption_decision.adopted=true`。
- 旧的 `L2_AGENT_MODE=codex-cli` patch harness 只能作为 legacy core-patch artifact 生成路径，不是 L2 evolve 主线。
