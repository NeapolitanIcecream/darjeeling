# L1 Rust ProgramBank 模块

模块：NLU runtime layer 位于
`darjeeling.targets.nlu.layers.l1_rust_programbank`。Core 只允许保留
target-neutral worker/process mechanics 和中性 fixture。

## 用户决策

L1 从第一版开始使用 Rust native CPU program。L1 evolve 由 L4 coding agent 通过 Codex CLI 修改 Rust 代码完成。DSL 不是主路径。

## 职责

- 将 hot, low-entropy, locally verifiable 子分布编译成 fast CPU path。
- 在不确定时 abstain。
- 输出 target-owned JSON、accept decision、program path 和 native latency。
- 支持 replay evaluator 和 benchmark。

## Rust artifact 结构

Core 默认 crate 是 `native/l1_empty_programbank`，只实现 worker I/O contract
并总是 abstain。它用于保证 Darjeeling core 没有默认应用知识。

具体任务的 ProgramBank 是 target artifact，由 L4 workspace flow 或显式配置的
artifact/source 提供。仓库中只保留中性 Rust accept-path fixture
`tests/fixtures/l1_neutral_programbank`，用于验证 worker contract；它不是 concrete
application target crate。

```text
native/l1_empty_programbank/
  Cargo.toml
  src/
    main.rs              # worker / batch CLI
    lib.rs               # try_answer public API; no target programs configured
    worker.rs            # target-neutral input/output structs

src/darjeeling/targets/nlu/native/l1_empty_programbank/
  Cargo.toml
  src/
    main.rs              # NLU worker / batch CLI
    lib.rs               # NLU try_answer public API
    frame.rs             # NLU utterance/frame ABI

target-specific promoted l1_programbank/
  Cargo.toml
  src/
    main.rs              # worker / batch CLI
    lib.rs               # try_answer public API
    frame.rs             # input/output structs
    normalize.rs
    programs/
      mod.rs
      task_rule.rs
      slot_extractors.rs
    tables/
      hot_phrases.rs
      intent_aliases.rs
  tests/
    hard_cases.rs
  benches/
    l1_latency.rs
```

Promoted artifact 中保存完整 source snapshot、binary 或 build instructions。

## L1 coding-agent harness 当前状态

目标架构与 L2/L3 同构：隔离 L1 workspace、一个 long-running L4 agent
session、agent 自主决定 edit/compile/unit test/bench/replay/stop、outer scope
check、private gate 和 outer replay promotion。L1 的 editable surface 是高性能
Rust/native ProgramBank source；workspace tools 是 compile、unit test、bench 和
replay，而不是 Optuna 或 prompt eval。

已实现 `L4CodingAgentAdapter` / `run_l1_coding_agent_job`：

- 将当前 Rust crate 复制到 generation-scoped `workspace/l1_programbank/`。
- 写入 protected `workspace/program.md`、`workspace/workspace_manifest.json` 和 `workspace/contexts/teacher_train.jsonl`、`hard_cases.jsonl`、`context_families.json`、`current_metrics.json`、`objective.json`、`constraints.md`、`commands.md`。
- `context_families.json` 使用 dataset-independent、schema-aware 聚合：按 teacher intent 与 slot signature 形成 family，记录 support、hard-case support、chosen layer counts、当前 L1 outcome counts、common tokens 和少量例子。Codex prompt 要求优先阅读该 summary，再按需查看 raw JSONL。
- context 输入类型使用 `TeacherTrace`，并扫描 forbidden gold/eval/future 字段。
- 支持 `dry-run` 模式，通过 fixture patch 修改 candidate workspace，用于测试 artifact packaging 和 state machine。
- 支持 `agent-session` 模式，调用一次 Codex over `workspace/`，让 agent 在同一 session 内自主 edit/compile/test/bench/replay/stop；只允许修改 `l1_programbank/` 和写 `runs/` scratch。
- 支持 legacy `codex-cli` 模式，仍以 crate 目录为 cwd 运行一次 Codex，用于兼容旧实验；真实 L1 evolve 应优先使用 `agent-session`。
- `agent-session` 结束后会 scope check protected `contexts/`、`program.md` 和 `workspace_manifest.json`；越界修改会以 scope violation 失败，不进入 validation/promotion。
- `provenance.json` 使用 `l1-agent-provenance-v1` schema，汇总 agent-session policy、workspace scope policy、Codex JSONL event type、外层命令 return code/stdout/stderr tail、diff 文件数和增删行数。
- 可选运行 `cargo test` 作为 agent job validation。
- compiler generation 已在 `L1_AGENT_MODE` 非 disabled 时调用该 harness。
- L1 candidate 成功后写入 artifact manifest，并进入 `L0 -> L1 -> L2 -> teacher fallback` 离线 replay gate。
- runtime 在 promotion 后会重新加载 promoted L1 crate，使下一窗口使用新的 Rust worker。
- `benchmark_worker` 已输出独立 worker benchmark 指标：requests、accepted share、native/integration p50/p95、throughput 和 program path counts。
- `edge-mvp l1 bench --out <path>` 可写出 `l1-benchmark-v1` JSON。
- `edge-mvp report` 在 run settings 或 manifest 提供 L1 crate/binary 时，会生成或复用 `reports/l1_benchmark.json`，并在 summary、metrics 和 curves 中展示。
- L1 coding-agent 成功生成 candidate crate 时，compiler 会写 generation-scoped `l1/l1_benchmark.json`，并把 `l1_benchmark` 放入 artifact manifest。
- `curves.html` 会汇总 generation-scoped L1 benchmark，展示每代 status、native p95、integration p95 和 throughput。

非阻塞后续项：

- `L1_AGENT_MODE` 默认仍是 disabled；真实 L1 evolution 实验必须显式开启 `agent-session`。`experiment preflight` 会把 disabled 记为 warn，避免误以为默认 smoke 配置已经覆盖 L1 coding-agent evolution。
- 更深层的 agent 内部命令重建仍取决于 Codex JSON event schema；当前已稳定记录 raw transcript、event type summary、外层命令摘要和 candidate diff summary。
- Rust crate 尚未接入 Criterion/cargo bench；当前跨代曲线基于 worker smoke benchmark，而不是 Criterion microbench。

## I/O contract

Core worker contract 是 target-neutral JSONL：

```json
{"request_id": "r1", "input": {"text": "alpha request"}}
```

```json
{
  "request_id": "r1",
  "accepted": true,
  "output": {"label": "alpha"},
  "program_path": "target/program_alpha",
  "native_latency_us": 12,
  "reason": "matched target rule"
}
```

NLU target-owned crate/adapter 可以继续使用 NLU ABI：

```rust
pub fn try_answer(utterance: &str) -> L1Result;
```

Result：

```rust
pub struct L1Result {
    pub accepted: bool,
    pub frame: Option<Frame>,
    pub program_path: String,
    pub native_latency_us: u64,
    pub reason: String,
}
```

Worker API 使用 JSONL，以便 Python runtime 和 replay harness 简单集成。

## 内部表示

允许 coding agent 使用任何 native-friendly 结构：

- if/else tree
- tight loop
- trie
- perfect hash
- regex automata
- small state machine
- table-driven matcher
- hand-written extractors

允许存在 dead code 或错误分支。外层不要求理解每个分支，只要求 replay/benchmark 给出可接受结果。

## Accept policy

L1 core 默认 abstain。只有 target-specific ProgramBank 完整通过本地检查才 accept。

L1 native rule 的优先级是 precision，而不是 coverage。对 NLU target，如果一个
Rust path 不能完整输出 teacher frame 所需 slot，必须 abstain，不能用空 slots
或粗糙 span 接收。中性 test fixture 只展示 worker accept/abstain contract；
真实 high-precision native path 必须在 target workspace 或 target artifact 中产生，
不能被解释为 Darjeeling core 默认 schema。

Coverage 扩张应由 L1 coding agent 在 teacher-visible hard cases 上演化 Rust 代码，并经过 replay gate；不能为了吸收更多请求牺牲 frame-level precision。

冲突处理：

- 没有匹配：abstain。
- 多个路径产生同一 frame：accept，并记录 matched paths。
- 多个路径产生冲突 frame：abstain。
- extractor 失败或 slot validator 失败：abstain。

## Metrics

L1 必须报告：

- coverage
- accepted accuracy
- wrong accept rate
- forced global accuracy
- native p50/p95 latency
- integration p50/p95 latency
- binary/source size
- program path coverage

Forced global accuracy 只用于展示 L1 牺牲泛化能力，不用于 runtime accept。

## 禁止项

- L1 不能读取 gold label。
- L1 不能读取 teacher cache 之外的未来 labels。
- L1 runtime 不能调用 L4。
- L1 runtime 不能联网。
- L1 不允许训练统计模型；训练型弱层属于 L2。

## DSL 的位置

DSL 若保留，只能是可选辅助：

- 作为 Rust tables 的输入格式。
- 作为 report 中可读规则摘要。
- 作为 coding agent 生成代码时的中间草稿。

DSL 不能成为 L1 唯一 backend，也不能阻止 agent 写更直接的 Rust 代码。
