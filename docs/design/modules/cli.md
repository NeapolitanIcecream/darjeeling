# CLI 模块

模块：`darjeeling.cli`

## 职责

- 使用 Typer 提供用户入口。
- 做参数解析、配置装配、高层 orchestration。
- 不包含训练逻辑、Rust L1 实现逻辑、prompt 构建细节或 promotion 细节。

## 命令

```bash
edge-mvp prepare --locale en-US --out data/processed/massive_en_us
edge-mvp run --stream zipf-heavy --max-requests 3000 --compile-every 500 --teacher live-or-cache
edge-mvp --settings settings.yaml run --stream zipf-heavy --max-requests 3000 --compile-every 500 --teacher live-or-cache
edge-mvp experiment preflight --run-dir runs/latest --teacher live-or-cache
edge-mvp experiment main-evolution --run-dir runs/main
edge-mvp experiment direct-l4-optimization --run-dir runs/direct-l4-optimization
edge-mvp report --run-dir runs/main
edge-mvp experiment compare --run runs/main --run runs/no-l2 --out-dir runs/experiment-comparison
edge-mvp l1 build --crate-dir native/l1_programbank
edge-mvp l1 bench --crate-dir native/l1_programbank
edge-mvp l1 bench --crate-dir native/l1_programbank --out runs/main/reports/l1_benchmark.json
edge-mvp l2 tune --traces runs/main/traces.jsonl --out runs/main/reports/l2_tuning.json
edge-mvp l3 bench --out runs/main/reports/l3_benchmark.json
edge-mvp l3 replay-prompt --prompt runs/main/artifacts/generations/gen_001/l3/l3_prompt.candidate.json --traces runs/main/traces.jsonl --out runs/main/reports/l3_prompt_replay.json
edge-mvp l3 promote-prompt --run-dir runs/main --prompt runs/main/artifacts/generations/gen_001/l3/l3_prompt.candidate.json --replay runs/main/reports/l3_prompt_replay.json
```

L1 子命令是 harness/debug 入口，不代表 L1 用 Python 实现。`l1 bench` 输出 `l1-benchmark-v1` JSON，可写入 `--out` 供 report 复用。

`--settings <path>` 是全局 option，位置在子命令之前。未显式传入时，CLI 会在当前工作目录存在 `settings.yaml` 的情况下读取它；环境变量和 `.env` 会覆盖 YAML 文件值。写入 run directory 的 `settings.json` 是非 secret 配置快照，不包含 API key 明文。

L3 `bench` 是显式硬件/model preflight，默认用 `shadow` 模式和 settings 中的本地 SLM 配置，输出 `l3-benchmark-v1` JSON。失败时也可写出 error status；只有传 `--fail-on-error` 才把失败变成非零退出码。

L3 `replay-prompt` 是显式 regenerated replay 入口：它读取一个 `L3PromptArtifact` 和带 teacher labels 的 traces，强制以 shadow 语义调用本地 SLM，并写出 `l3-prompt-replay-v1`。该命令会加载本地模型，因此不属于默认 compiler 路径。

L3 `promote-prompt` 只消费已有 replay artifact，不重新调用模型。它检查 replay schema、status、请求数、would-accept 数、accepted accuracy 和 wrong accept rate；通过后创建新的 artifact generation，把 prompt 写为 runtime `l3_prompt`，并保留 `l3_prompt_replay` 证据。

## `run` 流程

1. 解析 CLI 和 settings。
2. 初始化 run directory。
3. 读取或创建 stream。
4. 加载 current artifact manifest。
5. 根据 settings 启动 L1 Rust worker。
6. 根据 settings 决定 L3 disabled/shadow/guarded。
7. 构建 L0/L1/L2/L3/L4 runtime layers。
8. 顺序处理 request。
9. 每条 request 写 trace。
10. 每 `compile_every` 条触发 compiler。
11. 写 generation metrics。

## `experiment` 流程

Experiment 子命令不是 metadata 占位；它们会执行 replay 并生成 report：

1. 读取实验 spec。
2. 应用 spec 的 settings override，例如 `L2_ENABLED=false` 或 `L2_GUARD_MODE=always_accept`。
3. 清理该 experiment run dir 中上一轮 `artifacts`、`reports`、`traces.jsonl`、`settings.json` 和 `experiment.json`，但保留 `teacher_cache.jsonl`。
4. 写 `experiment.json`。
5. 写 `settings.json`。
6. 调用 `run_replay`。
7. 调用 `generate_run_report`，产出 `summary.md`、`metrics.csv`、`artifacts.csv` 和 `curves.html`。

这个默认重置用于避免旧 `manifest.current.json` 中的 L0 exact cache 污染新实验。需要有状态增量演化时使用普通 `edge-mvp run`，不要复用 `experiment` 子命令作为 resume 入口。

当前实验入口：

- `main-evolution`
- `direct-l4-optimization`
- `l2-family`
- `l2-mlp`
- `l2-tuned`
- `l2-tuned-lower-miss`
- `no-guard`
- `no-l2`
- `workload-locality`
- `hard-buffer`

`no-guard` 是诊断性 ablation：它设置 `L2_GUARD_MODE=always_accept` 和 `FORCE_PROMOTE_ARTIFACTS=true`，使无 guard 的 L2 artifact 能进入该隔离 experiment runtime，报告 threshold 移除后的真实错误率和时延。该配置不用于主线 evolution。

`l2-mlp` 是确定性 MLP family 实验：它设置 `L2_INTENT_MODEL_FAMILY=mlp`，不要求 live L4 proposal，用于把 MLP candidate 与默认 `sgd_logreg` 在同一 replay/report 框架下比较。

`l2-tuned` 是 Optuna tuning 实验：它设置 `L2_TUNING_MODE=optuna`，compiler 只用 `teacher_train` 内部切分做调参，写出 `l2/l2_tuning.json`，再用 best config 训练最终 L2 candidate。

`l2-tuned-lower-miss` 是分布对齐诊断实验：它设置 `L2_TRAINING_SCOPE=lower_miss` 和 `L2_TUNING_MODE=optuna`，让 tuning 与训练只看当前 `teacher_train` 中 L0/L1 未接收的样本。该实验用于检验 L2 在真实低层 miss 分布上的吸收改善，不取代默认 `teacher_train` 主线。

`workload-locality` 会在同一个 experiment root 下分别运行 `uniform`、`zipf-mild` 和 `zipf-heavy` 子目录。

`experiment compare` 不重新执行实验，只读取已有 run dir。输入可以是重复传入的 `--run`，也可以是 `--root` 下递归发现的 `traces.jsonl` 所在目录。输出 `comparison.csv` 和 `comparison.html`。

`experiment preflight` 是实验前只读检查入口，输出 `experiment-preflight-v1` JSON。它检查 processed train split、teacher cache/API key 可用性、L1 Rust crate、L1 agent 配置和 L3 benchmark artifact。默认不下载数据、不调用 OpenAI、不加载本地 SLM；`--check-l1-build` 才会构建 L1。

`L1_AGENT_MODE=disabled` 在 preflight 中是 warn，不是 fail。它允许用户跑 smoke/replay 实验，但明确说明这不是完整 L1 evolution；真实 L1 evolution 实验必须显式设置 `L1_AGENT_MODE=codex-cli` 并保证 `codex` 命令可用。

## Fail-fast 行为

- `teacher=live` 且缺少 API key：fail fast。
- `teacher=cache` 且 cache miss：fail fast。
- `experiment preflight` 若有 fail check：fail fast；warn check 只进入 JSON，不导致非零退出。
- L1 promoted artifact 缺失或 Rust worker 启动失败：fail fast。
- L3 enabled 且模型加载失败：如果 mode 是 `guarded`，当前 run fail fast；如果 mode 是 `shadow`，记录错误并自动降级为 disabled。

## 输出要求

CLI 输出只报告关键路径、run dir、artifact generation 和失败原因。详细上下文、prompt、agent transcript、metrics 写入 run directory。
