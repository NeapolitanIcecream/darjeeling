# eval 与 reports 模块

模块根：`darjeeling.eval`

## `eval.metrics`

职责：

- intent accuracy。
- slot micro F1。
- frame exact match。
- coverage。
- accepted accuracy。
- wrong accept rate。
- forced global accuracy。
- latency percentiles。
- cost per 100 requests。
- layer share。

Gold label 只在这里与 final output 对比。

## L1 指标

L1 需要单独报告：

- native inner latency。
- Python integration latency。
- accepted path distribution。
- source/binary size。
- benchmark throughput。
- forced global accuracy。

这些指标用于证明 Rust native L1 真的承担 hot path，而不是 Python glue 的表现。

## `eval.experiments`

实验：

- main evolution。
- direct L4 optimization ablation。
- L2 family search。
- no-guard。
- no-L2。
- workload locality。
- hard buffer/disagreement replay。

每个 experiment 输出独立 run dir，不覆盖主 run。

## `eval.reports`

Report 必须包含：

- layer summary。
- evolution summary。
- artifact summary。
- L1 Rust code/path 摘要。
- L1 coding-agent diff 摘要。
- L4 context/cache summary。
- promotion per-layer regression summary。
- L3 mode/hardware summary。
- failed experiment analysis。

失败实验不能伪造曲线。若目标结果未出现，报告 bottleneck。

## L3 report

L3 必须报告：

- mode: disabled/shadow/guarded。
- model name。
- actual device。
- load success/failure。
- model load time。
- generation latency。
- parse failure rate。
- guarded accepted accuracy。
- whether L3 contributed to final route。

若 L3 disabled，不应把它包装成成功贡献；应说明禁用原因和硬件状态。

## Promotion report

每次 promotion 必须报告整组 score delta 和 per-layer delta。若整组提升但某层 regression，报告中必须显式标记该遗留问题，而不是只展示总分。

当前实现状态：

- `edge-mvp experiment ...` 会执行 replay 并生成 report，不再只是初始化 metadata。
- 已有实验入口：main evolution、direct L4 optimization、L2 family、no-guard、no-L2、workload locality、hard-buffer。
- compiler 每个 generation 写 `candidate_metrics.csv` 和 `promotion.json`。
- `edge-mvp report` 会读取 traces、settings、current manifest、generation manifests 和 promotion records。
- report 输出 `summary.md`、`metrics.csv`、`artifacts.csv`、`curves.html` 和 `hard_cases.jsonl`。
- `summary.md` 展示 layer summary、L2 unguarded diagnostics、evolution summary、artifact summary、settings、current manifest、promotion 摘要和 L3 mode/hardware 摘要。
- layer summary 表固定输出 `layer | coverage | accepted_accuracy | wrong_accept_rate | forced_global_accuracy | p50_ms | p95_ms | cost/100`。
- layer summary 的 label 优先使用 gold frame，没有 gold 时退到 teacher frame；`accepted_accuracy` 只统计该层作为最终选择的有 label 样本，`wrong_accept_rate` 按全局有 label 请求数归一化。
- `forced_global_accuracy` 表示强制使用某层当轮产物时的全局准确率；该层缺失产物或 abstain 均按错误计入。
- `p50_ms`/`p95_ms` 统计该层实际观测到的 `LayerResult` latency，沿用 report 模块的 percentile 口径；`cost/100` 表示该层观测成本按全 run 请求数摊销后的每 100 请求美元成本。
- evolution summary 表固定输出 `generation | L4_calls/100 | cost/100 | p95_ms | frame_em | L0_share | L1_share | L2_share | L3_share | L4_share`。
- evolution summary 从 promotion record 的 candidate objective 与 replay layer counts 生成；没有 compiler generation 时显式报告无 generation。
- artifact summary 表固定输出 `artifact_id | type | generation | coverage_delta | accuracy_delta | cost_delta | promoted | reason`，并展示 artifact set 级 objective delta 与 per-layer delta。
- `summary.md` 还展示 L1 Rust ProgramBank 摘要、program path 分布、native latency 和 L1 coding-agent diff/source excerpt。
- 当 run settings 或 promoted manifest 明确给出 L1 crate/binary 时，report 会生成或复用 `reports/l1_benchmark.json`。
- `reports/l1_benchmark.json` 使用 `l1-benchmark-v1` schema，记录 status、benchmark corpus、requests、accepted share、native/integration p50/p95、throughput、program path counts、source size 和 binary size。
- L1 benchmark corpus 优先使用当前 trace 的去重 utterance，最多 128 条；没有 trace 时使用固定 smoke corpus。
- L1 benchmark 若没有配置则跳过；若本地 build/worker 失败，则写入 error status，不阻塞已有 report 生成。
- 若 generation manifest 包含 `l1_benchmark`，report 会在 metrics 和 `curves.html` 中生成 L1 benchmark by generation 表格。
- `metrics.csv` 聚合 run layer share、`layer_summary`、`l2_unguarded`、`evolution_summary`、gold eval frame exact、latency percentile、L1 native latency、L1 program path counts、L1 benchmark metrics、promotion objective 和 per-layer delta。
- `l2_unguarded` 指标来自 trace 中每次 L2 运行的 `metadata.predicted_frame`，按 threshold=0 语义统计 evaluated/labeled/correct/wrong/runtime accepted、frame accuracy、wrong prediction rate、guard probability p50/p95、intent support similarity p50/p95、intent support margin p50/p95 和 L2 latency p50/p95。它用于回答“如果 L2 不被 threshold 拦截，正确率和时延如何”。
- `summary.md` 的 Settings 段展示非 secret price assumptions，包括 layer per-request estimate 和 L4 token pricing；API key 只记录 presence，不写明文。
- `artifacts.csv` 列出 generation manifest 中的 artifact lineage。
- `hard_cases.jsonl` 从最新 generation/current manifest 的 hard buffer 导出，保留 `visibility` 字段，供后续人工检查和实验归因使用。
- `curves.html` 生成基础 cumulative layer share 曲线、L1 program path/native latency table、L1 native benchmark table 和 promotion table。
- `generate_experiment_comparison_report` / `edge-mvp experiment compare` 会从多个 run dir 汇总 `comparison.csv` 和 `comparison.html`，展示 experiment、stream、request count、gold frame exact、端到端 p95、各层 share、promotion 统计、bottleneck codes 和 L1 benchmark 摘要。
- Experiment comparison 已有 report-level `comparison_score` 与 `comparison_rank`，按 frame exact、p95 latency、L4 share 和 bottleneck count 做排序；HTML 还包含 bottleneck summary 表。
- L3 report 从 trace metadata 聚合 actual mode/device、failure/parse failure/repair rate、generation latency、model load time、confidence、would-accept accuracy、guarded accepted accuracy 和 recorded-trace guard threshold recommendation。
- 若存在 `reports/l3_benchmark.json`，report 会展示 L3 explicit hardware/model benchmark，包括 status/error、backend actual device、load/generation latency、parse/repair、would-accept 和 throughput。
- report 生成 failed experiment analysis，基于 traces、settings 和 promotion records 归因到 proposal 定义的 bottleneck：workload locality、L1 rule coverage、L2 guard calibration、local SLM JSON stability、teacher consistency、promotion gate。

非阻塞后续项：

- 更丰富的实验对比可视化，例如趋势图、显著性区间和多因素归因图；当前对比页已有表格、rank 和 bottleneck summary。
- `cargo bench`/Criterion benchmark 曲线；当前已有跨 generation worker smoke benchmark 表格，但不是 Criterion microbench。
- 更细的 failed experiment 自动归因，例如跨实验比较和多因素排序。

## `eval.plots`

使用 matplotlib/plotly 输出：

- `curves.html`
- metrics CSV
- artifact CSV

Plot 输入来自 metrics 文件，不从 trace 中临时推导未记录字段。
