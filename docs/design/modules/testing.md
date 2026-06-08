# testing 模块

测试策略围绕行为契约，不围绕当前实现细节。

## 单元测试

- frame parser。
- gold leakage boundary。
- teacher cache fail-fast。
- L4 context builder 不泄漏 gold。
- L2 guard threshold。
- L3 JSON parser/repair 标记。
- L3 disabled/shadow/guarded mode transitions。
- objective/promotion gate。
- hard buffer mining priority and gold leakage boundary。

## L1 Rust 测试

Rust side：

- `cargo test`
- hard case fixtures。
- extractor unit tests。
- conflict/abstain tests。

Python integration：

- worker startup。
- JSONL request/response。
- timeout/recover-as-abstain。
- native latency field 存在。

Benchmark：

- native p50/p95 latency。
- batch throughput。
- integration overhead。

## Compiler integration tests

- `edge-mvp run --teacher cache --max-requests 5`。
- L1 coding-agent job 可在 dry-run fixture workspace 中产生 candidate。
- rejected candidate 不更新 current manifest。
- L4 proposal schema invalid 时 candidate 被拒绝。
- artifact-set promotion records per-layer regression flags。
- hard buffer artifact is written and does not replace independent replay coverage。

## No-mock 边界

主 demo 不允许 fake labels。测试可以使用真实 L4 response 形成的 fixture cache；fixture metadata 必须说明来源和 schema version。

## Agent harness 测试

L1 coding-agent harness 应支持 dry-run mode：

- 不调用真实 Codex CLI。
- 使用预置 patch fixture。
- 用于测试 artifact packaging、replay 和 promotion state machine。

真实 L1 evolution 实验必须调用 Codex CLI，不以 dry-run 结果作为 demo 指标。

当前测试覆盖：

- dry-run fixture patch 能在 candidate workspace 中产生 diff。
- agent context files 不包含 `gold_frame` 或 gold slot value。
- disabled mode 不会启动 agent job。
- 单元测试不调用真实 Codex CLI。

## L3 硬件适配测试

- disabled mode 不加载模型且 run 不阻塞。
- shadow mode 模型加载失败时按配置降级并记录。
- guarded mode 模型加载失败时 fail fast，除非 CLI 显式允许 degrade。
- report 记录 actual device 和 mode。
