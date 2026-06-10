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
   三层都使用隔离 workspace、一个 long-running L4 agent session、agent
   自主 edit/evaluate/search、outer private gate 和 outer replay promotion。
   不同层的 editable surface 和工具不同：L1 是 Rust/native program，L2
   是 target student code/Optuna/evaluate，L3 是 prompt/context/bench。

## 文档结构

- [00 用户决策与优先级](00_decisions.md)
- [01 总体架构](01_architecture.md)
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

## 命名

项目发行包名和 Python import 包名应为 `darjeeling`。CLI 命令名保留 proposal 中的 `edge-mvp`。

```text
[project]
name = "darjeeling"

[project.scripts]
edge-mvp = "darjeeling.cli:app"
```

当前实现已使用 `src/darjeeling`。CLI 命令名不需要随 Python 包名变化。
