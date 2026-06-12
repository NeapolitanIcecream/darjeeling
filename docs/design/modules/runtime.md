# runtime 模块

模块根：`darjeeling.runtime`

Core runtime 当前只保留 target-independent mechanics：router、exact cache、cost
和 timing。Concrete NLU replay 和 trace JSONL IO 已移到
`darjeeling.targets.nlu.replay` 与 `darjeeling.targets.nlu.trace`。

## `runtime.router`

职责：

- 固定顺序执行 `L0 -> L1 -> L2 -> L3 -> L4`。
- 收集每层 target-neutral `LayerResult`。
- 根据 `accepted` 决定是否停止。
- 返回 target-owned output JSON。

Router 不读取 gold label，不训练模型，不调用 compiler，不解释 target payload。

单条 request：

```json
{"request_id": "r1", "input": {"text": "alpha request"}}
```

单条 layer response：

```json
{
  "layer": "L1",
  "accepted": true,
  "output": {"label": "alpha"},
  "confidence": 0.98,
  "reason": "accepted by target artifact",
  "latency_ms": 1.2,
  "metadata": {"artifact": "l1"}
}
```

## `runtime.exact_cache`

Exact cache 使用 target contract：

```text
target.normalize_request(input) -> teacher_label JSON
```

Core 可以保存 target-owned JSON payload，但不能解释字段名。

## Trace ownership

Core contract 中定义 target-neutral `TraceRecord` 和 `TeacherTrace`。当前 NLU
workflow 仍使用 legacy NLU trace schema，因此 reader/writer 位于 target package。
新增 core workflow 时应使用 `darjeeling.contracts.TraceRecord`。

## `runtime.cost`

职责：

- 根据配置中的 pricing 参数估算 cost。
- 将 cost 归因到 L4 teacher/fallback、compiler proposal、agent job。

价格不可 hardcode 到逻辑里。Report 必须写明价格假设。

## `runtime.timing`

职责：

- 提供 monotonic timing helper。
- 记录每层 wall-clock latency。
- 允许 target layer 在 metadata 中追加 native/backend latency。
