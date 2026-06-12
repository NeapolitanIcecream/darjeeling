# L0 exact cache 模块

Core 模块：`darjeeling.runtime.exact_cache`。NLU legacy layer 位于
`darjeeling.targets.nlu.layers.l0_cache`。

## Core 职责

- 使用 selected target 的 `normalize_request(input)` 生成 cache key。
- 存储 `normalized_request -> output JSON`。
- 返回 target-neutral `LayerResult(output=...)`。
- 不解释 request/output 内部字段。

## Exact cache

Key：

```text
target.normalize_request(input)
```

Value：

```text
teacher_label JSON from prior real teacher call
```

Accept 条件：

```text
normalized request exists in exact cache
```

Exact cache 不使用 gold label。Cache line 必须可追溯到真实 teacher response。

## Target 职责

NLU target 可以把 cache value 解释为 `Frame(intent, slots)`，也可以把
utterance normalization 作为 target-owned 逻辑实现。Core 只调用 target contract，
不能 hardcode utterance、frame、intent 或 slot。

## Metrics

- exact hit rate
- accepted accuracy
- wrong accept rate
- p50/p95 latency
- cache size
- teacher cache provenance coverage
