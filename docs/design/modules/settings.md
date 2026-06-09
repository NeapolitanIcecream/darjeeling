# settings 模块

模块：`darjeeling.settings`

## 职责

- 读取环境变量、CLI override、可选 settings 文件和默认值。
- 生成不可变 run config。
- 在 run 开始时写入 `runs/<id>/settings.json`。
- 将所有会影响 teacher cache、artifact compatibility、report 解释的参数显式化。

## 配置优先级

```text
CLI option > environment variable / .env > settings.yaml > code default
```

当前实现状态：

- `load_settings()` 会在当前工作目录存在 `settings.yaml` 时自动读取它。
- CLI 支持全局 `--settings <path>` 显式指定 YAML 配置文件；显式路径不存在时 fail fast。
- YAML 文件使用 Python field name，例如 `l1_agent_mode: codex-cli`、`local_slm_mode: shadow`。
- 环境变量和 `.env` 中的变量优先级高于 YAML，例如 `OPENAI_MODEL` 会覆盖 `openai_model`。
- run 开始时写出的 `settings.json` 包含完整非 secret settings snapshot，并用 `openai_api_key_present` 记录 API key 是否存在；不写出 API key 明文。

## 必要配置域

OpenAI / L4：

- `OPENAI_API_KEY`
- `OPENAI_BASE_URL`
- `OPENAI_MODEL`
- `L4_PROPOSAL_MODE`: `disabled | live`
- `PROPOSAL_MAX_TOKENS`
- model pricing
- layer cost estimates: `L0_COST_USD_PER_REQUEST`、`L1_COST_USD_PER_REQUEST`、`L2_COST_USD_PER_REQUEST`、`L3_COST_USD_PER_REQUEST`
- L4 token pricing: `L4_INPUT_USD_PER_MILLION`、`L4_CACHED_INPUT_USD_PER_MILLION`、`L4_OUTPUT_USD_PER_MILLION`
- L4 replay fallback estimate: `L4_DEFAULT_COST_USD_PER_REQUEST`
- teacher prompt version
- compiler prompt versions

L1 Rust：

- Rust workspace path
- cargo profile
- worker mode: batch CLI 或 long-lived worker
- per-request timeout
- build timeout
- agent job timeout
- Codex CLI command/model/effort
- `L1_AGENT_MODE`: `disabled | dry-run | codex-cli`
- `L1_AGENT_CODEX_COMMAND`
- `L1_AGENT_MODEL`
- `L1_AGENT_TIMEOUT_S`
- `L1_AGENT_SANDBOX`
- `L1_AGENT_APPROVAL_POLICY`

L3 local SLM：

- `LOCAL_SLM_MODEL`
- `local_slm.mode`: `disabled | shadow | guarded`
- device policy: `auto | cpu | mps | cuda`
- `LOCAL_SLM_MAX_NEW_TOKENS`
- `LOCAL_SLM_CONFIDENCE_THRESHOLD`
- `LOCAL_SLM_PROMPT_VERSION`
- max local model load time
- max per-request latency budget

Replay/promotion：

- compile cadence
- cold start size
- train/holdout split policy
- `HARD_BUFFER_MAX_CASES`
- `L2_ENABLED`
- `L2_GUARD_MODE`: `learned | always_accept`
- `L2_FRAME_SOURCE`: `retrieval | student`
- `L2_INTENT_MODEL_FAMILY`: `sgd_logreg | mlp`
- `L2_SLOT_MODEL_FAMILY`: `token_sgd | none`
- `L2_MAX_FEATURES`
- `L2_MAX_ITER`
- `L2_MLP_HIDDEN_LAYER_SIZES`
- `L2_MLP_ALPHA`
- `L2_MLP_EARLY_STOPPING`
- `L2_TUNING_MODE`
- `L2_TUNING_TRIALS`
- `L2_TUNING_TIMEOUT_S`
- `L2_TUNING_VALIDATION_FRACTION`
- `L2_TUNING_SPLIT_POLICY`: `chronological | stratified_random`
- `L2_TUNING_SEARCH_SPACE`
- `L2_TUNING_LATENCY_WEIGHT`
- `L2_TRAINING_SCOPE`: `teacher_train | lower_miss`
- `L2_TUNING_MIN_EXAMPLES`
- `L2_MIN_RUNTIME_EXAMPLES`
- objective weights
- wrong accept limits
- promotion accuracy epsilon
- `FORCE_PROMOTE_ARTIFACTS`: 默认 false，只用于隔离诊断实验；打开时 compiler 仍记录原始 promotion decision，但会强制 promotion candidate artifact。

Context/prompt cache：

- context token budgets
- prompt template versions
- prompt cache key policy
- context retention/logging policy

当前 GPT 5.5 provider 要求 `PROMPT_CACHE_RETENTION=24h`；`in_memory` 会导致 live teacher/proposal API 返回 invalid parameter。Cached-only replay 不受该设置影响。

## 设计约束

`settings.py` 不创建 OpenAI client，不加载 local SLM，不启动 Codex CLI，不访问 dataset。它只负责 config normalization 和 validation。

所有配置必须可序列化到 JSON。Report 必须能从 `settings.json` 重建本次实验的关键假设。
