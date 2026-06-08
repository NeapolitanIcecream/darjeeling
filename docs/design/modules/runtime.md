# runtime 模块

模块根：`darjeeling.runtime`

## `runtime.router`

职责：

- 固定顺序执行 `L0 -> L1 -> L2 -> L3 -> L4`。
- 收集每层 `LayerResult`。
- 根据 `accepted` 决定是否停止。
- 支持 shadow/audit 模式。

Router 不读取 gold label，不训练模型，不调用 compiler。

## L1 调用形态

L1 是 Rust native program。MVP 初期推荐 long-lived subprocess worker：

```text
darjeeling-l1-worker --artifact runs/<id>/artifacts/generations/gen_003/l1
```

Python runtime 通过 JSONL 或 framed stdin/stdout 与 worker 通信。worker 启动后常驻，避免 per-request process spawn 污染 latency。

当前实现状态：

- runtime 默认从 settings 指向的 Rust crate/binary 启动 long-lived worker。
- 若 promoted manifest 中包含 `l1_crate_dir`，runtime 会构建并使用该 promoted crate。
- `compile_every` 后若 manifest 的 L1 crate 发生变化，runtime 会关闭旧 worker 并启动新的 promoted worker，使下一窗口使用 candidate L1。
- Python integration timeout 默认 5s，用于覆盖 worker 冷启动抖动；Rust `native_latency_us` 仍由 worker 内部单独记录。
- `L2_ENABLED=false` 时 runtime 跳过 promoted L2 artifact。
- `L2_GUARD_MODE=always_accept` 时 runtime 将 loaded L2 bundle 的 accept threshold 设为 0，用于 no-guard ablation。

单条 request：

```json
{"request_id":"r1","utterance":"set alarm for seven"}
```

单条 response：

```json
{
  "request_id": "r1",
  "accepted": true,
  "frame": {"intent": "alarm_set", "slots": {"time": "seven"}},
  "program_path": "alarm/set_alarm_v3",
  "native_latency_us": 12
}
```

## `runtime.trace`

职责：

- append-only 写入 `traces.jsonl`。
- 写入 `teacher_cache.jsonl`。
- 生成 compiler-visible `TeacherTrace` view。
- 维护 hard buffer 输入事件。

Trace 写入必须包含 schema version。Teacher cache line 必须记录 prompt version、model、raw response、usage 和 cache key。

## `runtime.cost`

职责：

- 根据配置中的 pricing 参数估算 cost。
- 将 cost 归因到 L4 teacher/fallback、compiler proposal、L1 coding-agent job。

当前实现状态：

- L0/L1/L2/L3 的 per-request cost estimate 来自 settings。
- L4 live call 和 offline replay 优先使用 OpenAI usage token counts 估算成本，支持 cached input token discount。
- 若 replay trace 没有 L4 usage，则使用 `L4_DEFAULT_COST_USD_PER_REQUEST` 作为 fallback estimate。
- `settings.json` 会写入这些非 secret price assumptions；report 的 Settings 段展示它们。

价格不可 hardcode 到逻辑里。Report 必须写明价格假设。

## `runtime.timing`

职责：

- 提供 monotonic timing helper。
- 记录每层 wall-clock latency。
- 对 L1 额外记录 Rust 内部 `native_latency_us`。
