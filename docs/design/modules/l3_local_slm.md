# L3 local SLM 模块

模块：`darjeeling.layers.l3_local_slm`

## 职责

- 真实加载 local SLM。
- 使用 promoted prompt artifact 生成 frame JSON。
- 用 prompt-level acceptance policy 做 guard。
- 在硬件不足时支持 disabled/shadow，不阻塞主 demo。

默认模型：

```text
Qwen/Qwen2.5-0.5B-Instruct
```

## 子模块

- `l3_model.py`: transformers tokenizer/model lazy loading。
- `l3_prompt.py`: prompt artifact 和 rendering。
- `l3_parse.py`: JSON parse 和 json-repair。
- `l3_guard.py`: output validation / self confidence guard。
- `l3_local_slm.py`: runtime layer。

## Runtime 策略

- 模型 lazy load，避免 CLI help/import 加载大模型。
- parse 原始输出，失败时允许 `json-repair`，但必须记录 `repair_used`。
- `confidence` 来自模型输出字段，只能作为 guard 特征之一，不能单独证明正确。
- generation 参数只作为配置项，不作为稳定性保证。L3 是否进入主路由由 replay 指标和 guard 决定。

## 当前开发切片

已实现：

- `disabled` 模式不加载模型，runtime 中留下可审计的 L3 skipped/abstain 结果。
- `shadow` 模式会调用真实 local SLM backend；若模型加载/生成失败，降级为 disabled 并在 metadata 中记录错误，不阻塞主链路。
- `guarded` 模式会调用真实 local SLM backend；若模型加载/生成失败，run fail fast。
- JSON 输出先严格 parse，失败后用 `json-repair` 修复，并记录 `repair_used`。
- guard 检查 frame schema、allowed intent、allowed slot、`confidence_threshold`。
- runtime 会从 promoted manifest 的 `artifact_paths["l3_prompt"]` 读取 `L3PromptArtifact`；没有 promoted prompt 时使用 settings/default prompt。
- report 输出 L3 configured mode、model、device policy、actual mode/device、failure、parse failure 和 L3 是否成为 final route。
- report 还会从 trace metadata 聚合 L3 generation latency p50/p95、model load time p50/p95、confidence p50/p95、repair rate、would-accept count、shadow/guard would-accept accuracy 和 guarded accepted accuracy。
- `calibrate_l3_confidence_threshold` 已实现 recorded shadow/guard trace 上的第一版 threshold recommendation：在 wrong-accept 上限内最大化 would-accept coverage，并把 recommended threshold、coverage 和 wrong-accept rate 写入 report metrics。
- `edge-mvp l3 bench --out <path>` 已实现显式本地硬件/model benchmark，输出 `l3-benchmark-v1` JSON，记录 config、backend status、actual device、load/generation latency、parse failure、repair、would-accept 和 throughput。
- `edge-mvp report` 会读取已有 `reports/l3_benchmark.json` 并展示在 summary、metrics 和 curves；report 不会自动加载模型。
- offline promotion replay 使用 trace 中 recorded L3 accepted result 计入 L3 correctness、latency 和 wrong accept；compiler 不在 replay 阶段重新加载本地模型。
- `L4_PROPOSAL_MODE=live` 时，compiler 可以生成 `l3_prompt_candidate` artifact。Few-shot examples 只从 teacher-visible trace IDs 展开，不读取 gold labels。
- `edge-mvp l3 replay-prompt --prompt <candidate> --traces <trace.jsonl> --out <replay.json>` 已实现显式重新生成式 replay。该命令强制以 `shadow` 语义运行本地 SLM，输出 `l3-prompt-replay-v1`，记录 prompt hash、would-accept、accepted accuracy、wrong accept、parse/repair、latency 和 backend status。
- `edge-mvp l3 promote-prompt --run-dir <run> --prompt <prompt.json> --replay <replay.json>` 已实现 gated runtime promotion。只有 replay artifact 与 prompt hash 匹配，并通过 accepted accuracy、wrong accept rate 和非空 coverage gate 后，才写入新的 artifact generation，并把 prompt 作为 `artifact_paths["l3_prompt"]`。

非阻塞后续项：

- shadow 统计驱动的 L3 guard threshold 自动应用；当前只给 report 推荐 threshold，不自动修改 runtime prompt artifact 或 settings。
- 多 device policy 的自动搜索和硬件适配决策；当前 benchmark 是显式单配置 preflight，不自动选择或修改 settings。
- compiler 主循环内自动加载本地 SLM、对 `l3_prompt_candidate` 做重新生成式 replay 并自动 promotion；当前必须通过显式 L3 CLI 命令完成。

## 运行模式

```text
disabled
  不加载 local SLM。
  主链路语义上跳过 L3：L0 -> L1 -> L2 -> L4。
  trace 中可以记录一个 L3 disabled/abstain result，用于 report 审计。
  Report 必须标注 L3 disabled。

shadow
  加载并调用 local SLM，但不让 L3 output 成为 final answer。
  用于收集 parse failure、latency、agreement、guard calibration。
  模型加载失败时可以自动降级为 disabled，并写入 report。

guarded
  L3 在 guard 通过时可以 accept。
  模型加载失败或硬件不满足时 run fail fast，除非 CLI 显式允许 degrade。
```

## 硬件适配遗留问题

L3 local SLM 受硬件影响很大。MVP 不把本地硬件适配作为主 demo 阻塞项。

需要记录：

- model name。
- device policy 和实际 device。
- model load time。
- p50/p95 generation latency。
- parse failure rate。
- benchmark status/error。
- disabled/shadow/guarded mode。

Accept 条件：

```text
JSON valid or repaired successfully
Frame schema valid
self_confidence >= threshold
slot schema valid
intent in allowed shortlist
no parser hard failure
```

## L4 参与方式

L4 对 L3 使用 direct model API 提议 prompt artifact。Few-shot labels 必须来自 teacher traces，L4 只能选择 example IDs，不能发明 labels。

L4 生成的 prompt candidate 不是 runtime artifact。它必须先经过显式 `l3 replay-prompt`，再由 `l3 promote-prompt` 写入 manifest。Runtime 是否实际调用 L3 仍由 settings 中的 L3 mode 决定；manifest 中的 `l3_prompt` 只提供 prompt artifact。
