# Darjeeling 模块设计索引

本目录是 Darjeeling 的模块级设计文档。`../mvp_demo_proposal.md` 是原始 proposal；本目录记录后续设计讨论后的工程决策。若二者冲突，以本目录中标记为“用户决策”的内容为准。

## 用户决策

以下决策来自用户，优先级高于 proposal，也高于当前初始化实现：

1. **L1 从第一版开始使用 Rust native CPU program，不使用 Python 作为真实 L1 backend。**
   Python 只能用于 harness、测试驱动或胶水层，不能作为 L1 latency/coverage 结论的依据。
2. **L1 不再以 DSL 为主路径。**
   DSL 可作为可选的表格/规则辅助格式，但架构主路径是 agent-maintained Rust source tree。
3. **L4 在 evolve L1 时就是 coding agent 形态。**
   不是“一个 L4 模型给另一个 agent 发指令”，而是 L4 层在 L1 compiler mode 下通过 Codex CLI harness 运行，模型搭载在 coding agent 内部，利用多轮对话、局部读写文件、构建、测试和 benchmark 能力修改 L1 Rust 代码。
4. **L2 evolve 拆成 coding-agent 结构改造和本地调参工具。**
   L4 coding agent 负责真正需要 generalized intelligence 的 L2 代码/特征/验证协议改造；Optuna 等本地工具负责在 agent 设计的搜索空间内做超参搜索。旧的 direct L2 bounded config proposal 只保留为轻量 proposal path，不代表最终 L2 evolve 主路径。
5. **L1/L2/L3 evolve 应采用同构的 L4 coding-agent 架构。**
   三层都使用隔离 workspace、round-local L4 agent session、agent
   自主 edit/evaluate/search、outer private gate 和 outer replay promotion。
   不同层的 editable surface 和工具不同：L1 是 Rust/native program，L2
   是 target student code/Optuna/evaluate，L3 是 prompt/context/bench。
6. **Darjeeling core 应进一步收紧为 target-independent core。**
   NLU frame parsing 是一个 target，不是 core 的内建世界模型。Core 只保留
   分层、路由、trace、训练/进化 harness、replay、promotion、artifact 和
   质量 gate 等目标无关机制；`Frame(intent, slots)`、NLU teacher prompt、
   intent/slot diagnostics 和 MASSIVE loader 都属于 NLU target 或其 adapter。
7. **L1/L2/L3 共享外层 round policy，但 target 质量判断留在 target 层。**
   Core 只暴露 `max_rounds`、per-round timeout、patience、round executor、
   round result 和 run summary。Core 不再承载 agent budget/profile/evidence
   这类未执行或只服务历史 artifact 的伪抽象，也不声明 private gate、
   target quality 或 replay cadence 语义。L1/L2/L3 可以用不同 workspace 和
   evaluator，但对外汇报同一种 round/run 结构。
8. **Target-dependent 优化是允许的适配成本，不是 core 贡献。**
   Darjeeling 不承诺零 target knowledge 的 magic。用户可以为 target 提供
   diagnostics、feedback generator、selection helper、search tool 和
   target-specific L1/L2/L3 artifact code 来摸高效果；但实验报告必须区分
   target adaptation 带来的 lift 和 core/system 方法的 reusable evidence。

## 当前状态，2026-06-24

- 当前 Phase 1 benchmark 是 CLINC150 `data_full`。MASSIVE 仍保留为历史
  NLU adapter 和对照材料，但不再是当前机制验证主 benchmark。
- CLINC150 L4 teacher gate 已通过：`clinc150-intent-v2-label-cards` 在
  500-row validation gate 上达到 97.4% overall、98.4% in-scope、0.0%
  parse/schema failure。
- L2 已证明方向有吸收潜力，但还未达到 adoption gate：teacher-distilled
  L2 在 validation 上可达到 99.10% accepted precision / 50.32% coverage，
  locked test 在同一阈值下为 98.77% accepted precision；后续 guard 和
  AutoResearch repair 仍未产生可锁测采用的候选。
- L1 Rust ProgramBank 路线仍保留，但最新 CLINC150 dry-run patch 实验显示
  validation-only phrase rules 不能泛化：validation 为 100.00% accepted
  precision / 60.35% coverage，locked test 降到 92.73% accepted precision。
  下一步应使用真实 `agent-session` evolution 和 train-derived calibration/dev
  pressure，而不是继续 patch-mode 结果。
- Outer evolution policy refactor 已合入。Core 的共享面收敛到普通 round
  policy/summary；target 层继续拥有各自 workspace、diagnostics、private
  gate 和 adoption 逻辑。
- 后续 CLINC150 L1/L2 摸高实验允许在 NLU target 和 isolated candidate
  workspace 中加入 target-specific 优化；这些优化用于衡量 target adaptation
  投入后的上限，不应回流成 core 默认能力。

## 文档结构

- [00 用户决策与优先级](00_decisions.md)
- [01 总体架构](01_architecture.md)
- [02 Target Boundary Handoff](02_target_boundary_handoff.md)
- [03 Target-Independent 架构](03_target_independent_architecture.md)
- [04 Target-Independent 重构计划](04_target_independent_refactor_plan.md)
- [modules/schemas.md](modules/schemas.md)
- [modules/settings.md](modules/settings.md)
- [modules/cli.md](modules/cli.md)
- [modules/data.md](modules/data.md)
- [modules/runtime.md](modules/runtime.md)
- [modules/l0_cache.md](modules/l0_cache.md)
- [modules/l1_rust_programbank.md](modules/l1_rust_programbank.md)
- [modules/l2_student.md](modules/l2_student.md)
- [modules/l3_local_slm.md](modules/l3_local_slm.md)
- [modules/l4_layer.md](modules/l4_layer.md)
- [modules/l4_agent_evolve_harness.md](modules/l4_agent_evolve_harness.md)
- [modules/l4_context.md](modules/l4_context.md)
- [modules/compiler.md](modules/compiler.md)
- [modules/replay_promotion.md](modules/replay_promotion.md)
- [modules/artifacts.md](modules/artifacts.md)
- [modules/eval_reports.md](modules/eval_reports.md)
- [modules/testing.md](modules/testing.md)
- [../experiments/README.md](../experiments/README.md)

## 命名

项目发行包名和 Python import 包名应为 `darjeeling`。CLI 命令名保留 proposal 中的 `edge-mvp`。

```text
[project]
name = "darjeeling"

[project.scripts]
edge-mvp = "darjeeling.cli:app"
edge-mvp-nlu = "darjeeling.targets.nlu.main_cli:app"
```

当前实现已使用 `src/darjeeling`。`edge-mvp` 是 core CLI，通过静态 target
registry 选择 target；`edge-mvp-nlu` 暴露 NLU target 的 dataset/workflow 子命令。
