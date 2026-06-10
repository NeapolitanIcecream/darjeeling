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
edge-mvp l3 prompt-evolve --traces runs/main/traces.jsonl --out-dir runs/main/l3-prompt-evolve
edge-mvp l3 replay-prompt --prompt runs/main/artifacts/generations/gen_001/l3/l3_prompt.candidate.json --traces runs/main/traces.jsonl --out runs/main/reports/l3_prompt_replay.json
edge-mvp l3 promote-prompt --run-dir runs/main --prompt runs/main/artifacts/generations/gen_001/l3/l3_prompt.candidate.json --replay runs/main/reports/l3_prompt_replay.json
```

L1 子命令是 harness/debug 入口，不代表 L1 用 Python 实现。`l1 bench` 输出 `l1-benchmark-v1` JSON，可写入 `--out` 供 report 复用。

`--settings <path>` 是全局 option，位置在子命令之前。未显式传入时，CLI 会在当前工作目录存在 `settings.yaml` 的情况下读取它；环境变量和 `.env` 会覆盖 YAML 文件值。写入 run directory 的 `settings.json` 是非 secret 配置快照，不包含 API key 明文。

L3 `bench` 是显式硬件/model preflight，默认用 `shadow` 模式和 settings 中的本地 SLM 配置，输出 `l3-benchmark-v1` JSON。失败时也可写出 error status；只有传 `--fail-on-error` 才把失败变成非零退出码。

L3 `prompt-evolve` 是真实 L3 evolve 主入口。它创建隔离 `workspace/l3_prompt/`，只允许 agent 修改 `prompt/`，把 `contexts/`、`tools/`、`program.md` 和 `workspace_manifest.json` 作为 protected surface，启动一个 long-running L4 agent session，然后由 outer harness 做 scope check、candidate prompt snapshot、visible validation replay、private selection/promotion replay 和 summary。Agent 可见数据只包含 train、visible validation、task schema、objective 和 local SLM config；private selection/promotion rows 留在 outer private 目录。Workspace tools 包含 prompt validate、visible prompt eval、local SLM bench 和 latency/cost eval，输出只能写入 `runs/`。`--skip-replay` 只用于 smoke/no-model wiring，不支持 L3 质量结论。

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

`edge-mvp l2 tune` 支持 `--split-policy chronological|stratified_random`。默认 `chronological`，用于让 tuning validation 更接近后续 stream；`stratified_random` 仅用于 ablation 或小样本诊断。

`edge-mvp l2 target-evolve` 是新的 Inner L2 target-evolution loop 入口。它从一个 trace JSONL 中切出 `train`、agent-visible validation folds、私有 `selection_holdout` 和私有 `promotion_holdout`，创建 `workspace/l2_target/`，并在同一个 target workspace 内跑多轮快速 train/evaluate。`target/` 是唯一可写 target-dependent code；`runs/` 只是 scratch output；`system/darjeeling/`、`data/`、`tools/` 和 `program.md` 是 protected surface。每轮 mutating command 或 agent session 后、candidate evaluation 前会执行 workspace scope check，越界修改以 `workspace_scope_violation` 停止 job。agent workspace 只包含 train、visible validation 和 visible diagnostics；selection/promotion holdout 留在 outer job 私有目录，只由 outer harness 读取，private gate 派生 pass/fail 也不写入 agent-visible state。该命令用于解耦 L2 evolve 轮数与 outer replay `compile_every` cadence。

`target-evolve` 先评估 unmodified baseline，再根据 `--mode` 执行 target evolution。真实主路径是 `--mode agent-session`：harness 启动一个 long-running L4 agent session，agent 自己决定 edit、evaluate、调用 `tools/search_config.py`/Optuna 和停止的次数；session 结束后，outer harness 做 scope check、visible validation、private selection 和 private promotion。`dry-run` 只用于 fixture patch 测试；`local-search` 只用于 deterministic protocol probe 或作为 agent 可调用工具的外层包装；旧 `codex-cli` multi-round 模式保留为兼容路径。`--split-policy chronological|intent-stratified` 控制 fixed target split 的采样方式：默认 `chronological` 保持 stream-prefix 语义；`intent-stratified` 是小样本/窄 target patch 诊断选项，让 visible validation、selection 和 promotion 覆盖更多 teacher intent family，但不放宽 gates。`--visible-validation-folds N` 控制 agent-visible validation pressure：不显式传入时 `standard`/`smoke` 默认 1 fold，`fixed-inner` 默认 5 folds；`1` 使用 60/20/10/10 split 并只写 `inner_validation.jsonl`；大于 1 时默认使用 capped 50/30/10/10 split，写出 `inner_validation_shadow_*`，并用所有 visible folds 的 aggregate metric 作为 visible gate。继续增加 folds 只切分同一个默认 30% visible pool，不再继续压缩 train；需要更大 visible pool 时显式传 `--visible-validation-ratio`，summary 和 agent-visible state 会记录 requested/effective ratio。`data/target_diagnostics.json` 还会写 `latest_safety_backlog`，把 visible validation accepted-wrong families 单独排成优先队列；当该队列非空时，agent 应先修 wrong accepts，不能先做 broad threshold lowering 或 near-miss coverage 扩张。`latest_train_audit_safety_backlog` 来自 visible train audit，用于 validation backlog 清空但 selection 仍失败时设计更宽的 safety pattern；train audit 的 accepted-wrong count 是 candidate selection 前的 visible safety gate，但 train coverage 不是目标。`tools/evaluate.py --split train_audit` 可本地重跑该诊断。`--visible-cross-audit-folds N` 控制可见 held-out retraining 诊断：`0` 关闭，`N>=2` 在 visible train+validation 上按 intent 分折重训并写 `latest_visible_cross_audit_safety_backlog`；`fixed-inner` 默认 3 折，`standard`/`smoke` 默认关闭。Cross-audit 不读取 private holdout；它不是 private selection/adoption gate。`--budget-profile standard|fixed-inner|smoke` 控制默认预算。`agent-session` 默认只启动一个 live agent session；`--max-agent-rounds 0` 表示只准备 workspace/baseline/context，不启动 Codex。旧 `codex-cli` multi-round 模式仍使用 profile-specific live launch cap。Summary、`data/round_state.json` 和 `data/objective.json` 都会写入 `budget_policy.profile_intent`，标明本次 profile 是 `fixed_snapshot_research`、`cost_capped_default` 还是 `connectivity_smoke`。candidate selection gate 要求 visible validation、visible support、visible train-audit safety 和 private selection holdout 同时通过；visible support 要求每个 visible validation fold 至少 2 个 correct accepts，避免 near-zero coverage candidate 靠 abstain 通过；adoption 还要求 private promotion holdout。`--local-search-space compact|wide` 和 `--local-search-timeout-s` 只控制 local-search wrapper/tool 的 Optuna trial budget；`--timeout-s` 覆盖 live agent session timeout，默认继承 `L2_AGENT_TIMEOUT_S`。

`--target-scope teacher_train|lower_miss` 控制进入 target split 的 teacher-visible traces。默认 `teacher_train` 保持完整 teacher-labeled snapshot；`lower_miss` 只保留 L0/L1 没有接收的 traces，用于把 L2 target-evolution 对齐到真实会到达 L2 的残差分布。Summary、`data/round_state.json` 和 `data/objective.json` 会写 `target_scope`，包括 input count、scoped count 和被 lower layer 接收而排除的数量。该 scope 只改变 visible target data，不让 private selection/promotion holdout 进入 agent workspace。

`target-evolve` 还会在 summary、`data/round_state.json` 和 `data/objective.json` 写入 `evidence_policy`。`standard` 结果标为 `cost_capped_probe`，`smoke` 标为 `connectivity_smoke`，即使失败也不能解释成 L2 evolve 已耗尽。只有足够预算、足够 teacher-labeled snapshot size 且至少完成一次 scoped candidate evaluation 的 `fixed-inner` run 才会标为 `fixed_snapshot_research`，且仍需通过 private selection/promotion gates 和 outer e2e replay 后，才可作为 L2 target-evolution 质量证据。显式把 `fixed-inner` 的 `--rounds` 或 `--max-agent-rounds` 压得过低时，evidence policy 会降级为 short/budget-capped probe；teacher-labeled traces 少于 500 时会降级为 small-snapshot probe；`agent-session` 未启动、命令失败或 workspace scope violation 会降级为 no-launch/incomplete probe。

`edge-mvp l2 promote-target --target-run <target-run> --run-dir <run>` 将一个 `adoption_decision.adopted=true` 的 target-evolve run 提升为 replay runtime artifact。该命令继承 `<run>/artifacts/manifest.current.json` 中已有 artifact paths，重新用 target workspace 的 visible train split 训练 `l2_student.joblib`，复制 selected round 的 `target_snapshot` 到 generation 目录，并写入 `artifact_paths["l2_target"]`。如果历史 summary 没有 `target_snapshot`，才回退到 workspace 当前 `target/`。默认未通过 adoption gate 的 target run 会被拒绝；传 `--allow-non-adopted` 时只把 `best_round` 显式 stage 到 run manifest，用于外层 e2e replay 诊断，manifest 会记录 `l2_target_inner_adopted=false`、`l2_target_staged_for_outer_replay=true`、`l2_target_data_split_policy`、`l2_target_workspace_scope_policy` 和 `l2_target_private_holdout_evidence`。promotion 后普通 `edge-mvp run --compile-every <large>` 会自动加载 target wrapper。

`edge-mvp l2 replay-target --run-dir <run> --traces <traces.jsonl> --out <report.json>` 是 target artifact 的正式 outer replay gate。它默认用 current manifest 作为 candidate、candidate 的 parent manifest 作为 baseline，在同一批 teacher-labeled traces 上跑 compiler offline replay，并输出 `l2-target-outer-replay-v1` JSON：baseline/candidate objective、layer counts、per-layer deltas 和 promotion decision。默认 `accuracy_epsilon=0`，因此 target candidate 不能用 frame exactness regression 换覆盖率；默认包含 settings 中的 L1 Rust worker，`--no-include-default-l1` 只用于轻量测试或隔离诊断。

`no-guard` 是诊断性 ablation：它设置 `L2_GUARD_MODE=always_accept` 和 `FORCE_PROMOTE_ARTIFACTS=true`，使无 guard 的 L2 artifact 能进入该隔离 experiment runtime，报告 threshold 移除后的真实错误率和时延。该配置不用于主线 evolution。

`l2-mlp` 是确定性 MLP family 实验：它设置 `L2_INTENT_MODEL_FAMILY=mlp`，不要求 live L4 proposal，用于把 MLP candidate 与默认 `sgd_logreg` 在同一 replay/report 框架下比较。

`l2-tuned` 是 Optuna tuning 实验：它设置 `L2_TUNING_MODE=optuna`，compiler 只用 `teacher_train` 内部切分做调参，并优先在 chronological residual validation 上打分，写出 `l2/l2_tuning.json`，再用 best config 训练最终 L2 candidate。

`l2-tuned-lower-miss` 是分布对齐诊断实验：它设置 `L2_TRAINING_SCOPE=lower_miss` 和 `L2_TUNING_MODE=optuna`，让 tuning 与训练只看当前 `teacher_train` 中 L0/L1 未接收的样本，并继续使用 residual validation 做目标评估。该实验用于检验 L2 在真实低层 miss 分布上的吸收改善，不取代默认 `teacher_train` 主线。

`l2-agent` 是 legacy L4 coding-agent patch-generation 实验，不是当前 L2 target-evolution 主路径。它设置 `L2_AGENT_MODE=codex-cli` 和 `L2_TUNING_MODE=optuna`。Harness 会创建隔离 `workspace/l2_research/`，用稳定短 prompt 要求 Codex 读取 `program.md`，把动态 teacher-visible context 放入 `data/`，并限制 agent 只改 `candidate/`。当前 harness 只产出可审计 patch artifact，不在同一 Python 进程中热加载；manifest metrics 会写 `l2_agent_harness_role=legacy_patch_generation_not_target_evolve`。要测 patch 的真实效果，需要外层应用 patch、提交 Git、重启实验。主线 target-dependent L2 runtime code 应走 `edge-mvp l2 target-evolve` 和 `promote-target`。

`workload-locality` 会在同一个 experiment root 下分别运行 `uniform`、`zipf-mild` 和 `zipf-heavy` 子目录。

`experiment compare` 不重新执行实验，只读取已有 run dir。输入可以是重复传入的 `--run`，也可以是 `--root` 下递归发现的 `traces.jsonl` 所在目录。输出 `comparison.csv` 和 `comparison.html`。

`experiment preflight` 是实验前只读检查入口，输出 `experiment-preflight-v1` JSON。它检查 processed train split、teacher cache/API key 可用性、L1 Rust crate、L1/L2 agent 配置和 L3 benchmark artifact。默认不下载数据、不调用 OpenAI、不加载本地 SLM；`--check-l1-build` 才会构建 L1。L3 check 会记录 `mode`、model、device policy、benchmark path、readiness、是否 runtime-blocking 和 benchmark latency/device 摘要。`disabled` 是 non-blocking pass；`shadow` 缺成功 benchmark 是 warn；`guarded` 缺成功 benchmark 是 fail。

`L1_AGENT_MODE=disabled` 在 preflight 中是 warn，不是 fail。它允许用户跑 smoke/replay 实验，但明确说明这不是完整 L1 evolution；真实 L1 evolution 实验必须显式设置 `L1_AGENT_MODE=agent-session` 并保证 `codex` 命令可用。

`L2_AGENT_MODE=disabled` 在 preflight 中也是 warn，但它只覆盖 legacy
`l2-agent` patch-generation harness。当前 L2 主路径不靠这个环境变量；真实
target evolution 应显式运行 `edge-mvp l2 target-evolve --mode agent-session`。
Legacy `dry-run` 仍需要 `L2_AGENT_DRY_RUN_PATCH`。

## Fail-fast 行为

- `teacher=live` 且缺少 API key：fail fast。
- `teacher=cache` 且 cache miss：fail fast。
- `experiment preflight` 若有 fail check：fail fast；warn check 只进入 JSON，不导致非零退出。
- L1 promoted artifact 缺失或 Rust worker 启动失败：fail fast。
- L3 enabled 且模型加载失败：如果 mode 是 `guarded`，当前 run fail fast；如果 mode 是 `shadow`，记录错误并自动降级为 disabled。Preflight 不加载模型，但会用最近的 `reports/l3_benchmark.json` 把 guarded 缺失/失败 benchmark 提升为 fail，避免主路由实验在没有硬件证据时继续。

## 输出要求

CLI 输出只报告关键路径、run dir、artifact generation 和失败原因。详细上下文、prompt、agent transcript、metrics 写入 run directory。
