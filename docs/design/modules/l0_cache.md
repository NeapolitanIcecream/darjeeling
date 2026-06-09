# L0 cache 模块

模块：`darjeeling.layers.l0_cache`

## 职责

- 吸收 exact repeat 和后续可验证的 semantic repeat。
- 使用此前真实 L4 teacher frame 作为 cache value。
- 在命中时以极低 latency 返回 frame。

## 阶段策略

MVP 第一阶段只要求 exact cache。Semantic cache 放到第二阶段，避免在 Rust L1 和 replay pipeline 尚未稳定时引入 embedding/FAISS/threshold calibration 的额外变量。

原因：

- L1 Rust ProgramBank 已经承担 hot-path thesis。
- Semantic cache 和 L1 都会吸收近似重复，早期同时加入会让覆盖率解释变复杂。
- Semantic cache 的 threshold calibration 需要足够 teacher traces 和 replay support。

## Exact cache

Key：

```text
normalize(utterance)
```

Value：

```text
teacher_frame from prior real L4 call
```

Accept 条件：

```text
normalized utterance exists in exact cache
```

Exact cache 不使用 gold label。Cache line 必须可追溯到真实 L4 teacher response。

Promotion objective 中 exact cache 的 artifact complexity 按 `sqrt(cache_entries)`
计入。原因是 exact cache 是可追溯的数据表，维护风险随规模增长但低于线性；
promotion 仍由独立 replay gate 检查 e2e exact match 与 wrong accept rate。

## Semantic cache

第二阶段启用。

组成：

- sentence-transformers embedding model。
- FAISS index。
- cluster/frame artifact。
- threshold sweep。

Accept 条件：

```text
nearest_similarity >= threshold
support >= min_support
teacher_purity >= min_purity
frame_schema_valid
```

L4 可以建议 threshold 或 cluster policy，但最终 threshold 由 deterministic replay/grid search 选择。

## Metrics

- exact hit rate。
- semantic hit rate。
- accepted accuracy。
- wrong accept rate。
- p50/p95 latency。
- cache size。
- teacher cache provenance coverage。
