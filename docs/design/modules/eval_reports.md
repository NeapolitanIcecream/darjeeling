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

## 2026-06-24 Precision/Coverage Plotting

`darjeeling.eval.plots` now also owns target-neutral precision/coverage
reporting helpers:

- normalized JSONL read/write for rows that already contain target-owned
  precision and coverage metrics;
- Pareto frontier annotation over `coverage` and `accepted_precision`;
- Seaborn static evolution curves and single-knob operating-curve plots.

Core plotting still must not parse NLU frames, CLINC150 labels, utterances,
intents, OOS status, request ids, or target-specific policy semantics. Target
packages convert historical artifacts into normalized rows first. The CLINC150
implementation lives in `darjeeling.targets.nlu.precision_coverage` and writes:

- `docs/experiments/precision_coverage/round_metrics.jsonl`
- `docs/experiments/precision_coverage/operating_points.jsonl`
- `docs/experiments/precision_coverage/pareto_frontier.jsonl`

L1 operating curves are target-adapter overlays over recorded L1 accepts.
They are not generated L1 ProgramBank requirements, and the L1 evolve agent
does not need to know about plotting, Seaborn, Pareto frontiers, or operating
sweeps.

The repaired operating-curve contract uses target-neutral fields such as
`curve_id`, `curve_role`, `knob_name`, `knob_value`, `knob_label`,
`knob_order`, `knob_direction`, and `selection_scope`. Standard helpers connect
only one ordered `curve_id` per subplot; locked-test diagnostic curves are
separate from agent-visible curves. CLINC150 chooses L1 `risk_tolerance` and L2
`guard_threshold` in target code, not in core plotting.

## 2026-06-13 Field Metrics Update

NLU reports now include a field summary in addition to full-frame metrics. The target
report writes weak accepted field count, weak field coverage, weak field accuracy, wrong
accepted field rate, and per-layer field accepted accuracy/wrong-field rate. These values
come from NLU `FramePatch` traces and remain target-specific diagnostics.

Lower-layer teacher audit metadata is also reflected in hard cases: audit disagreement can
be mined as `teacher_audit_disagreement`, making accepted lower-layer mistakes visible to
compiler focus tasks and reports.

## 2026-06-14 Residual Metrics Update

Run reports split measured L4 buckets from traces: serving full L4, serving residual L4,
audit L4, and teacher-labeling L4. These rows use live/cache result metadata when tokens,
latency, and cost are present.

Offline replay and promotion still model residual L4 cost and latency from constants. Those
values are labeled as modeled in candidate cost metrics, including
`modeled_residual_l4_cost_usd_per_100_requests`,
`modeled_residual_l4_latency_ms_assumption`, and
`modeled_residual_l4_min_cost_fraction`.

Experiment comparison now puts field and residual metrics before layer share: weak field
coverage/accuracy, wrong accepted field rate, L4 conflict rate, full/residual L4 calls and
tokens per 100 requests, serving/audit cost per 100 requests, correct weak fields avoiding
full L4, and residual L4 verified fields. Layer share remains available, but it is no
longer the main signal for patch-runtime value.
