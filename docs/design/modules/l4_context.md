# L4 context 管理模块

模块：`darjeeling.compiler.l4_context`

## 设计目标

L4 context 管理的目标不是把更多历史塞进 prompt，而是让每次 L4 调用都满足：

- 可复现。
- 可审计。
- 不泄漏 gold label。
- prompt 前缀稳定。
- token budget 可控。
- 对 prompt/KV cache 友好。

## Session 策略

Direct model API：

```text
No long-lived sessions.
Each request is rebuilt deterministically from code-managed prompt templates and context blocks.
```

L1 coding-agent mode：

```text
Scoped multi-turn session inside one Codex CLI job.
Transcript is logged.
Final artifact is externally replayed.
```

## Context 类型

Teacher context：

- task instruction。
- intent schema。
- slot schema。
- output JSON schema。
- current utterance。

不包含历史。

当前实现状态：

- `darjeeling.compiler.l4_context` 已实现 teacher context builder。
- Teacher prompt 采用 stable system prefix + dynamic user tail；dynamic tail 只包含当前 utterance。
- builder 输出 `context_hash`、`prompt_cache_key`、`prompt_cache_retention`、`prompt_version` 和 `context_layout_version`。
- `CloudLLMTeacher` 已使用该 builder；live teacher cache line 会记录 `context_hash` 与 `prompt_cache_key`。

L2/L3/guard proposal context：

- role-specific schema。
- current artifact summary。
- teacher-visible metrics。
- selected teacher traces。
- hard cases。
- objective/gates。

当前实现状态：

- 已实现 proposal context builder，输入类型使用 `TeacherTrace`，并记录 `source_trace_ids`。
- dynamic trace block 按 `request_id` 确定性排序。
- builder 会扫描 forbidden gold/eval/future 字段。
- 已实现 `darjeeling.compiler.l4_proposal.L4ProposalAdapter` direct model call。
- Adapter 会记录 `context_hash`、`prompt_cache_key`、`source_trace_ids`、raw response 和 usage。
- Compiler 已用可开关方式接入 L2 config proposal、L3 prompt candidate proposal 和 guard search proposal：`L4_PROPOSAL_MODE=live` 时启用，默认 disabled。
- 当前 guard proposal 只覆盖 threshold search spec；更复杂的 guard feature-family candidate generation 是后续增强。

L1 coding-agent context：

- current Rust source tree。
- teacher-visible training traces。
- hard cases。
- current L1 metrics。
- target objective。
- forbidden data policy。
- available commands。
- final output requirements。

L1 context 以文件形式提供给 Codex CLI，而不是塞进一条巨大的裸 LLM prompt。

L1 agent context 文件不能包含 `teacher_promotion_holdout`、gold eval、future stream 或外层 evaluator 实现细节。Agent 可以读取当前 L1 Rust workspace 和公开 harness 文档，但不能通过 prompt/context 获得 promotion holdout。

## Prompt layout

Direct L4 API prompt 采用四段式：

```text
A. Stable prefix
   role definition
   output contract
   forbidden actions
   gold leakage warning
   schema
   objective definition

B. Semi-stable run prefix
   dataset locale
   intent list
   slot schema
   artifact schema versions

C. Dynamic context
   selected traces
   hot clusters
   hard cases
   current artifact summaries
   metrics

D. Current task
   label this utterance
   propose L2 config
   propose L3 prompt
   repair invalid JSON
```

A 必须完全稳定。B 尽量稳定。C/D 放最后。

OpenAI prompt caching 依赖 exact prefix match；官方文档建议把静态 instructions/examples 放在 prompt 开头，把动态内容放在末尾，并记录 usage 中的 cached tokens。实现应使用稳定前缀和一致的 `prompt_cache_key` 来提升命中率。参考：[Prompt caching](https://developers.openai.com/api/docs/guides/prompt-caching)、[Latency optimization](https://developers.openai.com/api/docs/guides/latency-optimization)。

## Context block 排序与裁剪

所有 dynamic block 必须确定性排序：

```text
cluster_score desc
wrong_accept_severity desc
trace_id asc
```

Token budget 超限时：

1. 保留 stable prefix。
2. 保留 output schema。
3. 保留 hard constraints。
4. 从低优先级 dynamic block 的尾部裁剪。
5. 不截断 JSON/schema/code fence 中间。

禁止任意截断整个 prompt。

## 记录字段

每次 direct L4 request 记录：

- prompt version。
- context layout version。
- context hash。
- source trace ids。
- artifact set id。
- prompt cache key。
- prompt cache retention 参数。
- input/output token usage。
- cached tokens。
- raw request/response。

Teacher cache 额外记录：

- normalized utterance。
- intent schema version。
- slot schema version。
- teacher prompt version。
- model。
- validated teacher frame。
- raw response。
- validation/repair status。

同一 cache key 不刷新。Temperature 或类似 generation knob 不作为 cache consistency 的核心机制。

每次 L1 coding-agent job 记录：

- agent prompt。
- context files manifest。
- Codex CLI command。
- transcript。
- file diff。
- commands run by agent。
- agent-side test/bench results。
- final candidate artifact path。

## Gold leakage 防线

Context builder 只能接收 `TeacherTrace` 或更窄结构。生成后的 context payload 和 L1 agent workspace manifest 必须通过字段扫描，禁止出现：

- `gold_frame`
- `gold_intent`
- `gold_slots`
- final eval labels
- future stream labels

测试必须覆盖 direct prompt context 和 L1 agent context files。

当前测试覆盖：

- teacher context 在 utterance 改变时 stable prefix 不变、dynamic tail 改变。
- proposal context 只含 teacher-visible trace，不含 `gold_frame` 或 gold slot value。
- forbidden context scanner 遇到 gold 字段会失败。
