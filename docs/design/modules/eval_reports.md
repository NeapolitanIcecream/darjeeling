# eval 与 reports 模块

Core 模块：`darjeeling.eval.plots`。当前 NLU report generator、experiment registry
和 target-specific metrics 位于 `darjeeling.targets.nlu.{reports,experiments,metrics}`。

## Core report 职责

Core report 只能展示 target-neutral 指标：

- correctness
- coverage
- accepted accuracy
- wrong accept rate
- forced global accuracy
- latency percentiles
- cost per 100 requests
- layer share
- artifact lineage
- promotion decision

Core report 不 import NLU schema，不把 frame exact match、intent accuracy、slot F1
或 intent/slot diagnostics 作为默认 promotion authority。

## NLU target reports

NLU target reports 可以追加：

- frame exact match
- intent accuracy
- slot micro F1
- slot-risk / intent-confusion diagnostics
- L2 unguarded diagnostics
- L1 ProgramBank path summaries
- L3 local SLM parse/repair/guard summaries

这些 sections 属于 target report，不改变 core promotion gate，除非 target contract
明确声明 additional gate。

## 当前实现状态

- `edge-mvp report` 当前指向 NLU target workflow CLI，并调用
  `darjeeling.targets.nlu.reports.generate_run_report`。
- `edge-mvp-nlu experiment ...` 会执行 replay 并生成 NLU target report，不再只是初始化
  metadata。
- Report 输出 `summary.md`、`metrics.csv`、`artifacts.csv`、`curves.html` 和
  `hard_cases.jsonl`。
- `metrics.csv` 可以包含 target-specific NLU metrics，但 shared core code 不能
  import NLU schema 来计算它们。
- `generate_experiment_comparison_report` / `edge-mvp-nlu experiment compare` 当前属于
  NLU target report path。

## `eval.plots`

使用 matplotlib/plotly 输出：

- `curves.html`
- metrics CSV
- artifact CSV

Plot 输入来自 metrics 文件，不从 trace 中临时推导未记录字段。

## 2026-06-13 Field Metrics Update

NLU reports now include a field summary in addition to full-frame metrics. The target
report writes weak accepted field count, weak field coverage, weak field accuracy, wrong
accepted field rate, and per-layer field accepted accuracy/wrong-field rate. These values
come from NLU `FramePatch` traces and remain target-specific diagnostics.

Lower-layer teacher audit metadata is also reflected in hard cases: audit disagreement can
be mined as `teacher_audit_disagreement`, making accepted lower-layer mistakes visible to
compiler focus tasks and reports.
