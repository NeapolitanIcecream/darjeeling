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
  evolves L2 Python patch candidates
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
- 当前 compiler 已在 `L4_PROPOSAL_MODE=live` 时调用该 adapter 生成 bounded L2 config proposal；这是轻量 proposal path。
- 用户决策后的 L2 主 evolve path 是 L4 coding agent 负责代码/特征/search-space 设计，Optuna 负责局部调参。当前已实现本地 Optuna tuner、`L2_TUNING_MODE=optuna` compiler 接入，以及 `L2_AGENT_MODE` coding-agent patch harness。
- L2 proposal 或 Optuna tuning 都不能直接决定 runtime accept；accept threshold 仍由 deterministic grid search 选择，最终仍由 replay gate 决定。
- 当前 compiler 已在 `L4_PROPOSAL_MODE=live` 时调用该 adapter 生成 legacy bounded L3 prompt candidate proposal。
- L3 prompt proposal 只能引用 teacher-visible trace IDs 作为 few-shot examples；compiler 会展开为 `L3PromptArtifact` 并写入 `l3_prompt_candidate`，但不会在缺少 regenerated/shadow replay 时提升为 runtime `l3_prompt`。真实 L3 prompt evolve 主路径是 `edge-mvp l3 prompt-evolve` 的 agent-session job。
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
- `agent-session` 模式会在隔离 `l1_programbank/` workspace 中启动一个 long-running Codex session；agent 自己决定 edit/test/bench/replay 的顺序和次数。
- `dry-run` 模式可应用 fixture patch，不调用真实 Codex CLI，用于测试。
- `codex-cli` 模式是 legacy one-shot path，模型、sandbox、approval policy、timeout 和命令名由 settings 控制。
- 真实 L1 evolution 实验必须使用 `agent-session` 模式；`dry-run` 结果不能作为 demo 指标。
- compiler generation 已在 `L1_AGENT_MODE` 非 disabled 时调用该 adapter。
- L1 candidate 已纳入 `L0 -> L1 -> L2 -> teacher fallback` 离线 replay gate。
- 当前默认配置仍是 disabled；主 demo 若要展示 L1 evolution，必须显式开启 `agent-session`。

## L2CodingAgentAdapter

用户决策，优先级高于原 proposal：

- L2 evolve 的主路径使用 L4 coding agent，而不是 direct API 直接生成完整设计。
- 动态资料不一股脑塞进 prompt；它们放进 workspace，由 Codex 自己决定读哪些文件。
- L2 dataset/runtime 相关代码放入隔离 target workspace 管理，agent 不直接修改 Darjeeling 宿主仓库。
- Darjeeling core 必须保持 dataset-independent；target workspace 内的 L2 runtime code 可以 target-dependent。
- Codex CLI 使用 GPT-5.5、独立配置和更长 timeout；不能隐式继承宿主机个人配置。当前隔离对象是 config/rules/session persistence，auth 仍由 Codex CLI 的 `CODEX_HOME` 机制提供。

Outer/Inner loop 分工：

- Outer Darjeeling loop 负责 replay、teacher-visible split、workspace creation、provenance、promotion holdout、artifact registry 和 core invariants。
- Inner L2 target-evolution job 在固定 target workspace 内启动一个 long-running L4 coding agent session：agent 自主决定 edit、evaluate、调用 Optuna/search 和停止的次数；本地 evaluator 训练/验证 L2，outer harness 在 session 结束后做 scope check 和 private gates。
- Inner loop 不等待新的 stream prefix，也不把 target-specific code 回写到 Darjeeling core。
- `data/train.jsonl`、`data/inner_validation.jsonl` 和可选的 `data/inner_validation_shadow_*.jsonl` 可以给 agent 使用；selection/promotion holdout 不进入 agent workspace，只保存在 outer job 私有目录并由 outer harness 使用。
- Workspace scope 是硬 gate：agent session 结束后、candidate evaluation 前检查 protected files。候选代码只能改 `target/`；`runs/` 是 scratch output；`data/`、`tools/`、`system/darjeeling/` 和 `program.md` 不能被 agent 修改。越界修改以 `workspace_scope_violation` 停止 job，不进入 private selection/promotion。
- Inner job 先评估 baseline，再启动 agent session。Agent 可见的 `round_state.json` 只包含 visible validation 历史、visible train audit 和可选 visible cross-audit；它不能包含 private gate rows、aggregate feedback，或由 private gate 推导出的 pass/fail 布尔值。`target_diagnostics.json` 只包含 visible validation / train-audit / cross-audit 的 bounded family-level triage、accepted-wrong safety backlogs 和 slot-risk backlogs；outer harness 使用 visible validation gate、visible support gate、visible train-audit safety gate 和 private selection holdout 做 candidate selection，使用 private promotion holdout 做最终验收，但不会把两者的 rows 或 aggregate feedback 写回 workspace。
- Outer summary 可以记录 `private_holdout_evidence`，用于人类和 outer harness 判断 private gate 失败是 zero-accept sparsity、wrong accepts 还是 promotion gate failure；该 evidence 不进入 agent workspace，不能成为下一轮 L4 agent 的可见反馈。
- Target split 默认 `chronological`；`intent-stratified` 是显式诊断选项，用来降低小样本 private selection/promotion 对窄 target family 的 zero-accept 稀疏性。它仍然只在 outer harness 内生成 split，不把 private rows 或 aggregate feedback 写回 agent workspace。
- `fixed-inner` 默认使用 5 个 visible validation folds；`standard`/`smoke` 默认 1 个。多个 folds 只切分 capped 30% visible validation pool，不继续压缩 train。
- `agent-session` 默认只启动一个 live GPT-5.5/Codex session；`rounds` 是 agent-visible 内部迭代预算提示，不是 harness 固定步骤。Optuna/search 是 workspace tool，由 agent 自己调用。Harness budget 限制 wall time、LLM spend、tool/eval/search cost，但不规定 agent 内部顺序。旧 `codex-cli` multi-launch、`local-search` mode 和 `dry-run` mode 只保留为兼容、smoke 和回归测试路径。`budget_policy.profile_intent` 会把这些边界写入 summary、round state 和 agent-visible objective。`--max-agent-rounds 0` 是 no-launch budget check，用于准备 workspace、baseline 和 context 而不调用 Codex；未启动、agent 命令失败或 workspace scope violation 的 run 不能支持质量结论。
- Visible validation improvement 只驱动继续探索；visible validation gate、visible support gate 和 visible train-audit safety gate 都是 selection 的必要条件，但不是充分条件。Adoption 必须看 `adoption_decision`，它只在某个 target round 同时通过 visible validation/support/train-audit、private selection 和 private promotion gates 时接受。

职责：

- 在 L2 compiler generation 中启动 Codex CLI。
- 向 agent 提供 autoresearch-style 隔离 workspace、teacher-visible L2 data files、当前 metrics、objective、constraints 和命令说明。
- L2 context files 包含 `slot_error_summary.json`，用于暴露 teacher-visible L2 wrong accepts 和 slot-level mismatch，使 agent 在扩大 coverage 前优先处理 frame exactness 风险。
- 允许 agent 修改 `candidate/` 中的 L2-owned source、tests 和模块设计文档，并调用 Optuna/local tests。
- 收集 diff、raw transcript、commands、agent report 和结构化 provenance。

Workspace layout：

- `program.md` 是稳定任务说明，承担 prompt 前缀之外的主要 instruction surface。
- `candidate/` 是唯一可写研究代码区，只包含 L2-owned 可 diff 文件。
- `system/darjeeling/` 是固定 Darjeeling system copy，用于 overlay candidate 后跑验证。
- `data/` 存放动态 teacher-visible 资料：`teacher_train.jsonl`、`hard_cases.jsonl`、`l2_context_families.json`、`slot_error_summary.json`、`current_metrics.json`、`objective.json`、`constraints.md` 和 `commands.md`。
- `tools/` 提供本地入口：`inspect_context.py` 查看 data，`run_checks.py` 将 candidate overlay 到 system copy 后运行 focused pytest/ruff。
- `workspace_manifest.json` 记录 workspace schema、candidate/data 路径和标准命令。
- `tools/inspect_context.py` 是无项目依赖的轻量脚本，标准入口为 `python3 tools/inspect_context.py`，避免只查看 context 时受 `uv --project system/darjeeling`、cache 或系统依赖状态影响。
- `tools/run_checks.py` 会优先在当前 Python 环境中调用 pytest/ruff；若当前环境缺少模块，则回退到 `uv run pytest/ruff`，避免有可用 venv 时仍强制嵌套 `uv run`。

Target workspace 工具隔离：

- `workspace/l2_target/tools/inspect_context.py` 同样只依赖 Python 标准库，标准入口为 `python3 tools/inspect_context.py`。
- `workspace_manifest.json.commands` 记录 `inspect_context`、visible validation evaluate、train-audit evaluate 和 local-search 入口。只有 evaluate/search 需要 Darjeeling 依赖；它们默认使用 `uv run --project system/darjeeling ...`，并在 `data/commands.md` 里提供 `PYTHONPATH=system/darjeeling/src python ...` fallback。

Prompt/cache 策略：

- Codex stdin prompt 保持极短且稳定：只要求读取当前 workspace 的 `program.md` 并完成一次 bounded L2 research iteration。
- 动态 trace、hard cases、metrics、objective 和 slot error summary 不进 prompt；它们作为文件放在 `data/`。
- 稳定 prompt 最大化 provider/server-side KV cache 机会，也避免每轮把大量代码或样本直接塞入上下文。
- `data/objective.json` 中的 budget intent 只说明本轮资源边界；它不把 private selection/promotion 结果塞回 prompt，也不让 agent 根据 private holdout 做下一轮调整。
- 是否读取某个 data file 是 coding agent 的局部决策；harness 只提供可见边界和审计 artifact。
- Agent objective 必须以 replay/promotion success 为目标；不能为了 raw L2 coverage 牺牲 frame exactness 或 wrong-accept safety。
- 在 legacy `candidate/` core-patch path 中，dataset-specific intent/slot hardcoding 默认不接受。新的 target-evolution path 不走这个限制：visible-data-derived target-specific code 可以写在 `target/`，且不应仅因为 dataset dependence 被拒绝；拒绝条件是越界进入 Darjeeling core、读取 private holdout、使用 workspace 外部 dataset 知识，或未通过 inner/holdout/outer replay gates。

Agent 可见范围：

- `teacher_train` 或当前 `L2_TRAINING_SCOPE` 选出的 teacher-visible traces。
- `visibility=train_visible` hard cases。
- 当前 L2 config、tuning、guard calibration、promotion-window metrics 的 summary。
- objective/gates 和命令说明。

Agent 不可见：

- MASSIVE gold。
- `teacher_promotion_holdout`。
- final eval labels。
- future stream。

Agent 权限：

- 只写隔离 research workspace 中的 `candidate/`。
- 不联网。
- 可运行 `tools/inspect_context.py`、`tools/run_checks.py`、Optuna/local deterministic tools 和小型 cached experiments。
- 不允许修改外层 replay、promotion logic、teacher cache、data loader 或非 L2-owned orchestration。

重要边界：

- Python L2 patch 不在当前 compiler 进程中热加载。Harness 产出的是 auditable patch candidate，记录 `runtime_patch_applied=false`。
- 若要让 patch 影响真实 L2 runtime，外层开发/实验循环必须应用 patch、纳入 Git、重启实验进程。
- Agent 不能 self-certify；即便 patch 通过自身验证，也必须经过外层 replay/promotion 和后续 experiment comparison。
- Artifact promotion 仍有遗留风险：整组 promotion 可能掩盖单层 regression。后续需要更细的 per-layer regression attribution 或分层 promotion 设计。

当前实现状态：

- 已实现 `darjeeling.compiler.l2_coding_agent.L2CodingAgentAdapter`。
- 当前 workspace schema 为 `l2-research-workspace-v1`，参考 `karpathy/autoresearch` 的 `program.md + editable train/candidate code + fixed evaluator` 思路，但边界改为 Darjeeling 的 L2 candidate overlay。
- 支持 `dry-run` fixture patch 和 `codex-cli` 模式。
- `codex-cli` 默认使用 `L2_AGENT_MODEL=gpt-5.5`、`L2_AGENT_TIMEOUT_S=7200`、`--ignore-user-config`、`--ignore-rules`、`--ephemeral` 和 `--skip-git-repo-check`。`--ignore-user-config` 不加载 `$CODEX_HOME/config.toml`；auth 仍使用 `CODEX_HOME`。
- compiler generation 已在 `L2_AGENT_MODE` 非 disabled 时运行该 harness，并记录 `l2_agent_*` artifact paths 与 metrics。
- `edge-mvp experiment l2-agent` 会开启 `L2_AGENT_MODE=codex-cli` 和 Optuna tuning，用于真实 L2 patch generation 实验。
- 默认仍是 disabled；普通 replay/tuning 不会产生 live LLM cost。
- 已新增 `darjeeling.compiler.l2_target_evolution` 和 `edge-mvp l2 target-evolve --mode agent-session`，用于新的 target-dependent single-session inner job。该路径当前支持 baseline-first evaluation、one-session Codex launch、fixed split evaluator、target workspace scope gate、private selection/promotion gates 和 target snapshot promotion。
- `target-evolve` 仍保留 legacy `dry-run`、`local-search` 和 multi-launch `codex-cli` 模式用于 fixture、protocol probe 和兼容测试。下一步真实 L4 experiment 应优先使用 `agent-session`，而不是 legacy `l2_research/candidate` patch harness 或外层 `local-search` mode。
- `tools/search_config.py` 不消耗 LLM tokens；它只优化可见 train/validation folds，并把 best visible config 写入 `target/config.json`。L4 coding agent 可在同一个 session 内调用它，但 search 结果不能自证 adoption。若 visible support 已经达标，agent 不应仅为了提高 raw accepts 降低 `accept_threshold`；这种 config 覆盖扩张容易绕过 target-local veto 的安全意图，必须由 visible train-audit、cross-audit 和 cue probes 共同证明必要且安全。
- Target workspace 暴露 `accept_prediction(...)` veto hook。L4 agent 可以用它实现 slot-risk、low-support、pattern-mismatch 等 abstain 规则；该 hook 不能 force accept，只能减少 core guard accepts，因此是控制 frame exactness regression 的优先机制。
- Target workspace 也暴露 `postprocess_frame(...)`。当 visible target data 支持稳定解析时，L4 agent 应优先用 postprocess 补全 slot 或修正 frame；这类 target-specific code 只能留在 `target/`，不能进入 Darjeeling core。
- Adopted target workspace 通过 `edge-mvp l2 promote-target` 进入 manifest，写入 `l2_student` 和 `l2_target` artifacts。Runtime replay 与 compiler offline replay 都加载 target wrapper，避免 target-loop evaluator 与系统 replay 语义分叉。
- Target evaluator 在 visible validation 上暴露 `near_miss_examples`，帮助 L4 agent 找到高 guard probability 但被拒绝的 coverage 机会。它同时写入 `target_diagnostics.json`，按 teacher intent family 汇总 rejected-correct、vetoed-correct、accepted-wrong 和 intent-correct-slot-wrong，并把 validation accepted-wrong families 提升成 `latest_safety_backlog`，避免 agent 只凭 8 条样本或 raw coverage 做选择。当 backlog 非空时，agent round 的目标顺序是先修 visible wrong accepts，再考虑 near-miss coverage。若 validation backlog 已清空但 private selection 仍失败，agent 可以读取 `latest_visible_cross_audit_safety_backlog` 和 `latest_train_audit_safety_backlog`，从 visible held-out retraining 和 visible train rows 中寻找更宽的 target safety pattern；slot-risk backlog 还提供 `high_guard_items`，把低频但 guard probability 高的 visible slot/schema mismatch 单独排出来，避免它们被大 family 的数量排序淹没。每个 slot-risk item 同时列出 `missing_slot_keys`、`extra_slot_keys` 和 `changed_slot_keys`，让 L4 agent 能直接按 schema 差异设计 postprocess 或 veto，而不是只靠读少量例子推断。Slot-risk 之后，intent-confusion backlog 用 teacher intent / predicted intent pair 暴露高 guard wrong-intent examples，覆盖 podcast/radio 等不是 slot mismatch 的边界风险。`visible_slot_cue_summary` 则提供跨 intent 的 visible slot cue 索引，例如 room value 对 `house_place` 的支持，或低频 `podcast_name` 对 podcast 语义的支持；其中 `slot_key_terms`、top values 和 examples 是 slotless/missing-slot accepted frame 停止前必须检查的可见依据，program 会把 podcast cue、room value、generic radio-name、missing radio-name cue、radio media-type、calendar date、bare upcoming-events、joke-type 和 spoken amount 风险作为 mandatory cue checks，并提供 `slot_cue_probes` 作为可执行 visible-only probe。这些诊断都是 diagnostic-only，不参与 selection/adoption。Private selection/promotion 的 near-miss、family diagnostics 和 holdout evidence 只能留在 outer artifact 中用于人类/outer harness 分析，不进入 agent workspace。

## Direct API session 策略

Teacher/fallback 和 legacy bounded proposal 的 direct API calls 不使用长期多轮
session。Retry/repair 也是显式 stateless request，包含原始 context hash、
invalid output 和 validator errors。

真实 artifact evolve 不再以 direct proposal 为主路径：L1、L2、L3 使用隔离
workspace + one long-running L4 agent session + outer private gate。当前 direct
L2/L3/guard proposal 代码是兼容和轻量 bounded proposal path，不能替代
agent-session evolve，也不能自证 adoption。

## OpenAI client

```python
OpenAI(
    api_key=settings.openai_api_key,
    base_url=settings.openai_base_url or None,
)
```

不得 hardcode API key。
