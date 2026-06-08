# L4 layer 模块

模块根：`darjeeling.layers.l4`

L4 是一个层次，不是一种单一调用方式。

## 三种 adapter

```text
L4TeacherAdapter
  direct model API
  stateless
  labels utterance / runtime fallback

L4ProposalAdapter
  direct model API
  stateless
  bounded JSON proposal for L2/L3/guard

L4CodingAgentAdapter
  Codex CLI coding-agent harness
  scoped multi-turn session
  evolves L1 Rust source tree
```

## L4TeacherAdapter

职责：

- runtime fallback。
- teacher labeler。
- 写 teacher cache。

Context：

- stable teacher prefix。
- intent schema。
- slot schema。
- output schema。
- current utterance。

不包含历史 trace、artifact metrics、hot clusters 或 gold label。

Teacher consistency：

- Teacher cache key 包含 normalized utterance、intent schema version、slot schema version、teacher prompt version 和 model。
- 同一 cache key 命中时不刷新。
- prompt/schema/model 变化进入新的 cache namespace。
- Parse failure 可以 retry，但不能 fallback fake label。
- 不依赖 temperature 作为一致性保证；若 provider 支持低随机性参数，可以设置，但稳定性来自 cache、schema、prompt version 和 validation。

当前实现状态：

- `CloudLLMTeacher` 已接入 `darjeeling.compiler.l4_context.build_teacher_context`。
- live teacher call 使用 stable system prefix + current-utterance dynamic tail。
- OpenAI request 使用 `prompt_cache_key` 和 `prompt_cache_retention`。
- teacher cache line 记录 `context_hash`、`prompt_cache_key`、prompt/schema/model/cache key、raw response 和 usage。

## L4ProposalAdapter

职责：

- 生成 L2 config candidates。
- 生成 L3 prompt candidates。
- 生成 guard threshold/search candidates。

Context：

- role-specific stable prefix。
- bounded output schema。
- teacher-visible historical summaries。
- current metrics/objective。
- hard cases。

不维护长期 session。

当前实现状态：

- proposal context builder 已实现，能构造 bounded JSON proposal 的 stable prefix 和 teacher-visible dynamic context。
- direct model API adapter `darjeeling.compiler.l4_proposal.L4ProposalAdapter` 已实现。
- adapter 使用 `build_proposal_context`、OpenAI chat completions、JSON response format、`prompt_cache_key`、`prompt_cache_retention` 和 `PROPOSAL_MAX_TOKENS`。
- adapter 返回 validated JSON object、raw response、usage、model、context hash、prompt cache key 和 source trace IDs。
- 当前 compiler 已在 `L4_PROPOSAL_MODE=live` 时调用该 adapter 生成 bounded L2 config proposal。
- L2 proposal 只允许修改白名单训练配置；accept threshold 仍由 deterministic grid search 选择，最终仍由 replay gate 决定。
- 当前 compiler 已在 `L4_PROPOSAL_MODE=live` 时调用该 adapter 生成 bounded L3 prompt candidate proposal。
- L3 prompt proposal 只能引用 teacher-visible trace IDs 作为 few-shot examples；compiler 会展开为 `L3PromptArtifact` 并写入 `l3_prompt_candidate`，但不会在缺少 regenerated/shadow replay 时提升为 runtime `l3_prompt`。
- 当前 compiler 已在 `L4_PROPOSAL_MODE=live` 时调用该 adapter 生成 bounded guard search proposal。
- Guard proposal 只能影响 threshold grid 与 wrong-accept 上限；最终 threshold 仍由 deterministic local search/replay 选择，L4 不能直接决定 runtime accept。
- 更复杂的 guard family/feature-family proposal 仍是后续增强。

## L4CodingAgentAdapter

职责：

- 在 L1 compiler generation 中启动 Codex CLI。
- 向 agent 提供隔离工作区、当前 L1 Rust source tree、teacher-visible context files、objective、constraints、命令说明。
- 允许 agent 多轮编辑 Rust 代码、运行 cargo test/bench/replay。
- 收集 diff、raw transcript、commands、agent report 和结构化 provenance。

Agent 可见范围：

- 当前 L1 Rust source tree。
- `teacher_train` trace/context。
- 公开 hard cases。
- 当前 L1 metrics。
- objective/gates。
- harness 命令说明。

Agent 不可见：

- MASSIVE gold。
- `teacher_promotion_holdout`。
- final eval labels。
- future stream。
- 非 L1 artifact 的私有内部状态，除非明确写入只读 summary。

Agent 权限：

- 只写 L1 candidate workspace。
- 不联网。
- 可运行 `cargo test`、`cargo bench`、train-visible replay。
- 不允许修改外层 evaluator、promotion logic 或 teacher cache。

重要边界：

- coding agent 属于 L4 层的 L1 compiler mode。
- 它不是被 L4 模型另行指派的外部 worker。
- agent 内部可以多轮，但 session 只属于当前 L1 evolution job。
- agent 不能 self-certify。外层 replay evaluator 决定 promotion。

当前实现状态：

- 已实现 `darjeeling.compiler.l1_program_compiler.L4CodingAgentAdapter`。
- adapter 会创建隔离 candidate workspace，写入 teacher-visible context files、prompt、raw transcript、diff、commands、agent report 和 `provenance.json`。
- `provenance.json` 汇总 Codex JSONL event types、外层命令摘要和 diff stats；raw transcript 仍单独保存，供后续更细解析。
- `dry-run` 模式可应用 fixture patch，不调用真实 Codex CLI，用于测试。
- `codex-cli` 模式会调用本机 `codex exec`，模型、sandbox、approval policy、timeout 和命令名由 settings 控制。
- 真实 L1 evolution 实验必须使用 `codex-cli` 模式；`dry-run` 结果不能作为 demo 指标。
- compiler generation 已在 `L1_AGENT_MODE` 非 disabled 时调用该 adapter。
- L1 candidate 已纳入 `L0 -> L1 -> L2 -> teacher fallback` 离线 replay gate。
- 当前默认配置仍是 disabled；主 demo 若要展示 L1 evolution，必须显式开启 `codex-cli`。

## Direct API session 策略

Teacher、L2、L3、guard 的 direct API calls 不使用长期多轮 session。Retry/repair 也是显式 stateless request，包含原始 context hash、invalid output 和 validator errors。

## OpenAI client

```python
OpenAI(
    api_key=settings.openai_api_key,
    base_url=settings.openai_base_url or None,
)
```

不得 hardcode API key。
