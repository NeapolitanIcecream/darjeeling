# settings 模块

模块：当前实现的 concrete settings 位于
`darjeeling.targets.nlu.settings`。Core 还没有 active `darjeeling.settings`
模块。

## 职责

- 读取环境变量、CLI override、可选 settings 文件和默认值。
- 生成 run config。
- 在 run 开始时写入 `runs/<id>/settings.json`。
- 将会影响 teacher cache、artifact compatibility、report 解释的参数显式化。

## 配置优先级

```text
CLI option > environment variable / .env > settings.yaml > code default
```

当前实现状态：

- `load_settings()` 会在当前工作目录存在 `settings.yaml` 时自动读取它。
- CLI 支持全局 `--settings <path>`；显式路径不存在时 fail fast。
- YAML 文件使用 Python field name，例如 `l1_agent_mode: agent-session`。
- 环境变量和 `.env` 中的变量优先级高于 YAML。
- `settings.json` 包含完整非 secret settings snapshot，并用
  `openai_api_key_present` 记录 API key 是否存在；不写出 API key 明文。

## Core 与 target 分区

Core-owned settings 只应覆盖 provider transport、runtime timing/cost、agent
harness、replay/promotion、artifact store 和 generic cost/latency 假设。

NLU target settings 可以覆盖 NLU L1/L2/L3/L4 workflow，例如 local SLM、
L2 intent/slot model family、guard/tuning/search space、target-evolution agent
配置、NLU diagnostics 和 target report 行为。这些字段保留在
`darjeeling.targets.nlu.settings`，不能重新提升为 core 默认。

## 设计约束

Settings 模块不创建 OpenAI client，不加载 local SLM，不启动 Codex CLI，不访问
dataset。它只负责 config normalization 和 validation。

所有配置必须可序列化到 JSON。Report 必须能从 `settings.json` 重建本次实验的
关键假设。
