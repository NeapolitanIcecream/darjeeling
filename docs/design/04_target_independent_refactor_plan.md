# 04 Target-Independent 重构计划

本文对照 [03 Target-Independent 架构](03_target_independent_architecture.md)
和代码现状，定义把 Darjeeling 收敛到 target-independent core 的重构计划。

## 目标

重构完成后，Darjeeling core 只保留分层、路由、trace、teacher transport、
compiler harness、agent workspace、replay、promotion、artifact、settings
和通用报告机制。Core 可以传递 target-owned JSON payload，但不解释 payload
内部字段。

NLU 是一个 target。`Frame(intent, slots)`、`utterance`、NLU teacher
prompt/parser、MASSIVE adapter、NLU L1/L2/L3 训练逻辑、slot/intent 诊断和
NLU 报告都在 NLU target 侧。

## 当前偏差清单

### Core schema 是 NLU schema

现状：

- `src/darjeeling/schemas.py` 定义 `Frame(intent, slots)`。
- `LayerResult` 使用 `frame`，不是 target-neutral `output`。
- `TraceRecord` 使用 `utterance`、`gold_frame`、`teacher_frame`、
  `final_frame`。
- `TeacherTrace` 仍然暴露 `utterance`、`teacher_frame`、`final_frame`。

目标：

- Core schema 改为 `JsonObject`、`LayerResult.output`、`TraceRecord.input`、
  `gold_label`、`teacher_label`、`final_output`。
- `Frame` 和 NLU `TaskSchema` 进入 NLU target。
- `TeacherTrace` 禁止携带 gold label，并只暴露 teacher-visible JSON view。

### Data 和 adapter 已经是 NLU 形状

现状：

- `src/darjeeling/data/records.py` 的 `DataRecord` 是 `utterance + gold_frame`。
- `src/darjeeling/data/frames.py` 负责 MASSIVE/NLU annotation 解析。
- `src/darjeeling/adapters/massive.py` 和 `massive_cli.py` 在 core package 下。

目标：

- Core data loader 只读取 target record JSONL/parquet，并把 records 交给
  selected target 解释。
- NLU target 定义自己的 record shape 和 frame parser。
- MASSIVE adapter 移到 NLU target adapter 层。

### Runtime path 是 `utterance -> frame`

现状：

- `src/darjeeling/layers/base.py` 的 `RuntimeLayer.try_answer` 接收
  `utterance: str`。
- `src/darjeeling/runtime/router.py` 返回 `Frame`。
- `src/darjeeling/runtime/replay.py` 手写 L0/L1/L2/L3/L4 cascade，并直接读取
  `record.utterance`、`record.gold_frame`。
- `src/darjeeling/layers/l0_cache.py` 使用 `normalize_utterance -> Frame`。
- `src/darjeeling/layers/l1_rust_programbank.py` 和
  `native/l1_empty_programbank/src/frame.rs` 固定 JSON ABI 为
  `utterance -> frame`。

目标：

- Core router 只调用 `RuntimeLayer.try_answer(input: JsonObject)` 并返回
  target-owned output JSON。
- L0 exact cache 使用 `target.normalize_request(input)`。
- Core L1 worker 只负责 build、process lifetime、timeout、JSONL IPC 和
  benchmark mechanics；NLU target 提供 L1 ABI adapter。
- NLU target 提供 L0/L1/L2/L3/L4 runtime builder。
- Target discovery 使用静态 registry，不使用 packaging entry point、插件系统或
  DI container。

### Teacher transport 和 prompt/parser 混在一起

现状：

- `src/darjeeling/layers/l4_cloud_llm.py` 同时负责 L4 transport、NLU
  `TaskSchema`、teacher prompt、response parser 和 teacher cache schema。
- `src/darjeeling/compiler/l4_context.py` 直接渲染 intent/slot prompt。

目标：

- Core 保留 retry、timeout、usage、cost、cache 文件读写和 provider transport。
- Target 提供 `TeacherAdapter.build_messages`、`parse_response` 和
  `cache_key_parts`。
- Teacher cache 使用 target-neutral 字段：`input`、`normalized_request`、
  `target_name`、`target_schema_version`、`prompt_version`、`teacher_label`。

### Compiler harness 直接 import NLU layers

现状：

- `src/darjeeling/compiler/loop.py` 直接调用 NLU L0 cache、L2 distiller、
  L2 tuner、guard optimizer、L3 prompt optimizer 和 L1 Rust program compiler。
- `src/darjeeling/compiler/l0_compile.py` 生成
  `frames_by_normalized_utterance`。
- `src/darjeeling/compiler/replay.py` 用 `frame == expected` 计算 correctness，
  并用 `frame_exact_match` 作为核心 objective。
- `src/darjeeling/compiler/l1_program_compiler.py` 按 intent/slot family 构建
  L1 agent context。
- `src/darjeeling/compiler/l2_distiller.py`、`l2_tuner.py`、`guard_optimizer.py`、
  `l2_target_evolution.py` 都包含 NLU L2 训练、诊断或 evolution 逻辑。
- `src/darjeeling/compiler/l3_prompt_optimizer.py` 是 NLU prompt evolution。

目标：

- Core compiler loop 只做 split、hard buffer、workspace、candidate orchestration、
  replay、promotion 和 manifest update。
- L0 exact cache 可以留在 core，但只存
  `target.normalize_request(input) -> output JSON`。
- L1/L2/L3 artifact generation 通过一个 `TargetCompiler.propose_artifacts`
  入口调用。
- Correctness 只通过 `target.labels_equal(output, expected, task_schema)` 判断。
- NLU L1 context、L2 train/tune/guard/evolution 和 L3 prompt evolution 进入
  NLU target compiler。

### Reports、metrics、CLI 和 settings 暴露 NLU vocabulary

现状：

- `src/darjeeling/eval/metrics.py` 定义 `frame_exact_match` 和
  `intent_matches`。
- `src/darjeeling/eval/reports.py` 把 `frame_exact_match` 作为 experiment
  comparison 字段，并展示 NLU L2/L3 细节。
- `src/darjeeling/cli.py` 暴露 L2/L3 NLU train/evolve/bench 命令和
  `teacher_frame`、`intent-stratified` 选项。
- `src/darjeeling/settings.py` 混合 core settings 和 NLU target settings，
  例如 `l2_intent_model_family`、`l2_slot_model_family`、L2 target agent
  settings。

目标：

- Core reports 只展示 correctness、accepted accuracy、wrong accept、
  layer share、latency、cost、artifact lineage 和 promotion decision。
- Target reports 追加 target-specific sections 和 metrics。
- Core CLI 选择 target 并运行通用 flow；target CLI 暴露 NLU prepare/train/evolve。
- Settings 分为 core settings 与 target settings，target settings 由 target
  package 解析。

### Tests 主要锁定 NLU-as-core 行为

现状：

- 多数 tests 直接 import `darjeeling.schemas.Frame`。
- Core tests 使用 `utterance`、`teacher_frame`、`gold_frame` fixture。
- NLU 行为测试和 core orchestration 测试混在同一个测试层。

目标：

- Core tests 使用 neutral target fixture，例如 `input={"text": ...}`、
  `output={"label": ...}` 或更简单的 JSON payload。
- NLU tests 迁移到 target test scope，并继续覆盖 MASSIVE、Frame parser、
  NLU teacher、NLU L2/L3、NLU diagnostics。
- Boundary test 扫描 core source/shared core tests，禁止 NLU vocabulary。

## 重构阶段

### Phase 0：锁定边界与测试入口

产出：

- 新增 core boundary test，扫描 core source 和 shared core tests。
- 定义允许出现 NLU vocabulary 的路径白名单：NLU target、adapter、target
  fixtures、experiment evidence。
- 新增 neutral target test fixture，用来跑 core router/replay/promotion。

约束：

- 这一阶段不移动大量代码，只先让后续改动有红线。
- Boundary test 可以先以 xfail 或 allowlist 记录现状，但每个阶段都必须减少
  allowlist。

完成标准：

- 有一条自动化测试能回答“core 里还剩哪些 NLU 词”。
- Core orchestration 至少有一个不依赖 Frame/utterance 的测试样例。

### Phase 1：引入 target-neutral core contracts

产出：

- 新增 core contract 模块，包含：
  - `JsonValue` / `JsonObject`
  - `LayerResult(output=...)`
  - `TraceRecord(input, gold_label, teacher_label, final_output, ...)`
  - `TeacherTrace`
  - `TargetSpec`
  - `TeacherAdapter`
  - `RuntimeLayer`
  - `TargetRuntime`
  - `TargetCompiler`
- Core replay/promotion API 增加 target 参数，但可以暂时由 NLU adapter 包装旧
  `Frame` 模型。

约束：

- 不引入动态插件系统；使用普通 Python object 和 repo 内静态 target registry。
- `Protocol` 只是 typing view；第一版允许一个 `NluTarget` object 同时实现
  teacher、runtime、compiler 和 report 方法。
- 不在这一阶段重写 NLU algorithms。

完成标准：

- 新 contract 有直接单元测试。
- 旧 NLU path 可以通过 adapter 跑通现有 smoke tests。
- 新 trace model 明确 `extra="forbid"`，teacher-visible view 不含 gold label。

### Phase 2：建立 NLU target package

产出：

- 新建 `src/darjeeling/targets/nlu/`。
- 移入或复制后收敛：
  - `Frame`
  - NLU `TaskSchema`
  - NLU `DataRecord`
  - NLU frame parser and utterance normalization
  - NLU teacher prompt/parser
  - MASSIVE adapter and CLI
- 提供 `NluTargetSpec`、`NluTeacherAdapter` 和 `NluTargetRuntime`。

约束：

- Core 不 import `darjeeling.targets.nlu.*`。CLI/bootstrap 通过静态 registry
  将 target name 解析为 target object。
- MASSIVE 只作为 NLU adapter 存在，不作为 core data module。

完成标准：

- `edge-mvp ... --target nlu` 能加载 NLU target object。
- MASSIVE prepare 命令迁移到 target CLI，例如 `edge-mvp-nlu massive prepare`。
- NLU target tests 覆盖 Frame parser、MASSIVE mapping、teacher parsing。

### Phase 3：重写 runtime layer 边界

产出：

- `RuntimeLayer.try_answer(input: JsonObject) -> LayerResult`。
- `CascadeRouter.route(input)` 返回 target-owned output JSON。
- L0 exact cache 改为 `normalized_request -> output JSON`。
- Core L1 worker 改为 generic JSONL worker：
  - core 写入 `{"request_id": ..., "input": ...}` 或 target ABI adapter 产物；
  - core 读取 accepted/output/metadata；
  - target 负责把 input/output 映射到 NLU ABI。
- NLU L2/L3 layers 迁移到 target package，并实现 target-neutral runtime
  protocol。

约束：

- Core L1 benchmark 只能 benchmark generic worker mechanics；NLU benchmark
  corpus 由 NLU target 提供。
- `native/l1_empty_programbank` 要么移动到 NLU target artifact template，要么
  改成 target-neutral empty worker。

完成标准：

- `src/darjeeling/layers/base.py` 不出现 `utterance`。
- Core router/replay tests 不依赖 `Frame`。
- NLU target runtime tests 覆盖 `utterance -> Frame` 行为。

### Phase 4：拆 compiler harness 与 target compiler

产出：

- Core compiler loop 只保留：
  - teacher-visible split
  - private selection/promotion holdouts
  - hard buffer lifecycle
  - agent workspace lifecycle
  - protected root checks
  - candidate replay
  - promotion decision
  - manifest update
- NLU target compiler 在一个 `propose_artifacts(context)` 入口内提供：
  - NLU L1 coding-agent context and artifact serialization
  - NLU L2 train/tune/guard
  - NLU L2 target evolution
  - NLU L3 prompt evolution
  - NLU diagnostics and probes
- `compiler/l0_compile.py` 改为 generic exact cache builder，依赖
  `target.normalize_request`。
- `compiler/replay.py` correctness 改用 `target.labels_equal`。

约束：

- 先移动/封装，再优化算法。`l2_target_evolution.py` 体量大，应先整体迁入
  NLU target，再在 target 内继续拆小模块。
- Core 不知道 intent-stratified split。类似策略由 target compiler 声明，core
  只执行 target 给出的 group key 或 split hints。

完成标准：

- `src/darjeeling/compiler/loop.py` 不 import NLU concrete layers。
- `src/darjeeling/compiler/replay.py` 不 import `Frame`。
- NLU compiler tests 覆盖 L1/L2/L3 artifact generation parity。
- Core compiler tests 使用 neutral target compiler object。

### Phase 5：改 replay、objective 和 reports

产出：

- Core objective 字段从 `frame_exact_match` 收敛为 generic correctness 指标。
- Wrong accept、accepted accuracy、coverage、latency、cost 和 layer deltas 保持
  core-owned。
- Target reports 可以追加 NLU metrics，例如 frame exact match、intent match、
  slot risk、intent confusion。
- Experiment comparison 使用 core generic score；target-specific comparison
  字段由 target 的普通 `report_sections(...)` 方法追加。

约束：

- Target-specific metric 不能默认成为 promotion authority，除非 target
  contract 明确声明 additional gate。

完成标准：

- Core report generation 不 import NLU schema。
- NLU report tests 覆盖原有 NLU summary 信息。
- Promotion decisions 在 core tests 中只依赖 `target.labels_equal`。

### Phase 6：拆 CLI 与 settings

产出：

- Core CLI：
  - `edge-mvp run --target nlu ...`
  - `edge-mvp report --target nlu ...`
  - `edge-mvp experiment ... --target nlu ...`
- Target CLI：
  - `edge-mvp-nlu massive prepare ...`
  - `edge-mvp-nlu l2 train ...`
  - `edge-mvp-nlu l2 target-evolve ...`
  - `edge-mvp-nlu l3 prompt-evolve ...`
- Settings 拆为 core settings 和 target settings：
  - core settings 保留 provider transport、runtime、agent harness、replay、
    promotion、artifact、generic cost/latency；
  - NLU settings 保留 L2 intent/slot model、NLU L3 prompt/model、NLU target
    evolution budget 和 NLU diagnostics。

约束：

- Core CLI 不出现 NLU-only option name。
- Target CLI 可以使用 intent/slot/frame vocabulary。

完成标准：

- Core CLI help 不包含 `teacher_frame`、`intent-stratified`、`slot` 等 NLU 术语。
- NLU CLI help 保留必要的 NLU vocabulary。
- Settings tests 覆盖 core-only 与 NLU target settings 分别加载。

### Phase 7：测试迁移与 boundary 收口

产出：

- Tests 分层：
  - core tests：schema contracts、router、teacher transport、artifact store、
    workspace, replay/promotion, settings, generic reports；
  - target tests：NLU data, MASSIVE, teacher, L1 ABI, L2/L3, diagnostics,
    target reports；
  - integration tests：`--target nlu` end-to-end smoke。
- Boundary allowlist 清零到只剩 target/adapters/target fixtures/experiment
  evidence。

完成标准：

- `rg` boundary test 在 core source 和 shared core tests 中找不到 NLU vocabulary。
- Existing NLU behavior 由 target tests 保护。
- Full test suite passes。

### Phase 8：删除旧入口和文档收敛

产出：

- 删除或重定向旧 core NLU modules：
  - `darjeeling.data.frames`
  - `darjeeling.adapters.massive*`
  - core `Frame` aliases
  - core NLU L2/L3 modules
  - core NLU CLI commands
- 更新模块设计文档，使 `modules/*` 不再把 NLU 形状当 core 默认。
- 保留 `02_target_boundary_handoff.md` 作为审计记录，`03` 作为目标架构，
  `04` 作为执行计划。

完成标准：

- 新用户只看 `03` 和 `04` 就能理解目标架构与实施顺序。
- Core source 的 public API 与文档中的 target-neutral contract 一致。

## 推荐执行顺序

1. Phase 0 和 Phase 1 一起做，先拿到 contract 和 boundary test。
2. Phase 2 建 NLU target package，但先保留行为 parity。
3. Phase 3 改 runtime，确保 e2e replay 还能通过 NLU target 跑。
4. Phase 4 迁 compiler，是最大风险阶段，优先整体迁移 NLU compiler entry。
5. Phase 5 改 reports/objective，避免 promotion 继续用 frame vocabulary。
6. Phase 6 拆 CLI/settings。
7. Phase 7/8 收尾，减少 allowlist，删除旧入口。

## 执行记录

### 2026-06-12 Phase 0/1 initial slice

- 新增严格 core boundary 扫描测试，直接跟踪 core source 和 shared core tests
  中剩余的 `Frame`、`utterance`、`intent`、`slot`、`teacher_frame` 等 NLU 词汇。
  当前测试使用显式 allowlist 记录历史耦合路径；后续每个迁移阶段必须减少该
  allowlist，最终只允许 target、adapter、target fixture 和 experiment
  evidence。
- 新增 `darjeeling.contracts`，提供 target-neutral `JsonObject`、
  `LayerResult(output=...)`、`TraceRecord(input/gold_label/teacher_label/
  final_output)`、`TeacherTrace(extra="forbid")`、`TargetSpec`、
  `TeacherAdapter`、`RuntimeLayer`、`TargetRuntime` 和 `TargetCompiler`。
- 在 `runtime.router` 中新增 JSON payload cascade router，保留旧 `Frame`
  router 作为兼容路径；新增中立 fixture 测试证明 core orchestration 可以在
  `input={"text": ...}` / `output={"label": ...}` 形状上运行。

### 2026-06-12 Phase 2 package scaffold

- 新增 `darjeeling.targets.nlu` package，先放入 NLU-owned `Frame`、
  `TaskSchema`、`DataRecord`、utterance normalization、annotation frame parser、
  teacher adapter、target spec、runtime placeholder 和 MASSIVE adapter。
- 新增 repo-local static registry，当前显式注册 `nlu -> NluTarget`；该 registry
  是普通 mapping，不使用 entry point、plugin system 或 DI container。
- 新增 `edge-mvp-nlu massive prepare` 入口；旧 `edge-mvp-massive` 仍保留并通过
  compatibility wrapper 调用 target-owned MASSIVE adapter。
- 新增 NLU target test scope `tests/targets/nlu/`，覆盖 target registry、
  frame parser、teacher adapter、target spec equality/validation 和 MASSIVE mapping。

### 2026-06-12 Phase 3 exact-cache slice

- 新增 target-neutral `runtime.exact_cache.ExactJsonCacheLayer`，核心存储形状为
  `target.normalize_request(input) -> output JSON`，返回
  `contracts.LayerResult(output=...)`。
- 新增 `exact_cache_from_teacher_traces(...)`，只读取 teacher-visible
  `TeacherTrace.input` 和 `TeacherTrace.teacher_label`，可选调用
  `target.validate_output(...)`，不解释 target payload。
- 旧 `layers/l0_cache.py` NLU 兼容路径暂不删除，后续 runtime/replay 切换时再把
  manifest 装载格式从 `frames_by_normalized_utterance` 收敛到 generic exact cache。

### 2026-06-12 Artifact manifest target identity

- `ArtifactManifest` 新增可选 `target_name` 和 `target_schema_version` 字段，为后续
  replay/promotion 装载 target-owned artifacts 提供通用身份信息。
- 字段保持可选，旧 manifest 可以继续加载；core 不写入 NLU 默认值，后续由
  target-aware CLI/runtime builder 在选择 target 后显式填充。

### 2026-06-12 CLI target selection compatibility

- `edge-mvp run` 和 `edge-mvp report` 新增 `--target` 参数，默认 `nlu`，通过
  repo-local static registry 校验 target 名称。
- `run` 暂时仍调用旧 NLU replay path，但会把 `target_name` 和
  `target_schema_version` 写入 run settings，给后续 target-aware runtime builder
  和 manifest 写入打基础。
- 现有 experiment helpers 暂时显式使用 `target="nlu"`，等 core experiment CLI
  拆分 target settings 后再暴露 target option。

### 2026-06-12 NLU data compatibility wrappers

- 旧 `darjeeling.data.records` 和 `darjeeling.data.frames` 不再承载
  `DataRecord`、utterance normalization 或 frame annotation parser 的实现；
  它们只兼容 re-export `darjeeling.targets.nlu.data` 中的 target-owned 实现。
- 旧 `darjeeling.schemas.Frame` 不再定义独立 core model，而是兼容 alias 到
  `darjeeling.targets.nlu.schemas.Frame`，避免 core 和 target 出现两个不同的
  NLU Frame 类型。
- 删除 shared core `tests/test_frame_parser.py`，frame parser 行为由
  `tests/targets/nlu/test_nlu_target.py` 覆盖；shared boundary allowlist 去掉
  `tests/test_frame_parser.py` 和已清空实现的 `src/darjeeling/data/records.py`。

### 2026-06-12 L0 compiler compatibility wrapper

- 旧 `compiler.l0_compile.exact_cache_from_teacher_traces(...)` 保持
  `dict[str, Frame]` 返回值以兼容现有 compiler loop，但内部先把 legacy
  `TeacherTrace(utterance/teacher_frame)` 转为 target-neutral
  `contracts.TeacherTrace(input/teacher_label)`，再调用
  `runtime.exact_cache.exact_cache_from_teacher_traces(...)`。
- NLU normalization/equality 来源改为 `NluTargetSpec` 和 target-owned `Frame`；
  后续切换 compiler loop 时可以直接传 target-neutral traces 和 target object。

### 2026-06-12 NLU teacher schema/parser aliases

- 旧 `layers.l4_cloud_llm.TaskSchema` 改为 `darjeeling.targets.nlu.schemas.TaskSchema`
  的兼容 alias；旧 `parse_teacher_frame(...)` 委托给
  `darjeeling.targets.nlu.teacher.parse_teacher_frame(...)`，但保持旧
  `TeacherParseError` 对外错误类型。
- L4 transport 仍在旧 module 中，prompt context 渲染仍待拆出；本切片只移走
  schema/parser ownership，降低 core 中重复 NLU 类型定义。

### 2026-06-12 Runtime layer base contract alias

- 旧 `layers.base.RuntimeLayer` 不再定义 `try_answer(utterance: str)`，改为兼容
  re-export `contracts.RuntimeLayer` 的 target-neutral
  `try_answer(input: JsonObject) -> LayerResult` protocol。
- 具体 NLU runtime layers 仍保留旧 `utterance -> frame` 兼容实现，后续迁到
  `targets.nlu.layers` 并通过 NLU runtime builder 包装。

### 2026-06-12 Runtime router API flip

- `runtime.router.CascadeRouter` 改为 target-neutral 主 API：
  `route(input: JsonObject) -> (output JSON, layer_results)`。
- 旧 `utterance -> Frame` cascade 保留为显式 `FrameCascadeRouter`，`JsonCascadeRouter`
  暂时作为 `CascadeRouter` alias 保持兼容。

### 2026-06-12 NLU L0 layer target ownership

- NLU `ExactCacheLayer(utterance -> Frame)` moved to
  `darjeeling.targets.nlu.layers.l0_cache`; old `darjeeling.layers.l0_cache`
  is now a compatibility re-export.
- Removed `src/darjeeling/layers/l0_cache.py` from the strict boundary allowlist.

### 2026-06-12 NLU L1 Python DSL target ownership

- NLU Python `ProgramRule` / `ProgramBankLayer` DSL moved to
  `darjeeling.targets.nlu.layers.l1_program_bank`; old
  `darjeeling.layers.l1_program_bank` is now a compatibility re-export.
- Moved shared `tests/test_l1_dsl.py` coverage into the NLU target test scope and
  removed the old shared test path plus `src/darjeeling/layers/l1_program_bank.py`
  from the strict boundary allowlist.

### 2026-06-12 NLU L2 target runtime ownership

- NLU `TargetL2Layer` postprocess/veto runtime wrapper moved to
  `darjeeling.targets.nlu.layers.l2_target`; old `darjeeling.layers.l2_target`
  is now a compatibility re-export.
- Moved `tests/test_l2_target_runtime.py` into the NLU target test scope and
  removed the old shared test path plus `src/darjeeling/layers/l2_target.py`
  from the strict boundary allowlist.

### 2026-06-12 NLU native L1 ABI target ownership

- NLU `RustL1Worker` / `RustProgramBankLayer` moved to
  `darjeeling.targets.nlu.layers.l1_rust_programbank`; old
  `darjeeling.layers.l1_rust_programbank` is now a compatibility re-export.
- Moved `tests/test_l1_rust_worker.py` into the NLU target test scope and
  removed the old shared test path plus `src/darjeeling/layers/l1_rust_programbank.py`
  from the strict boundary allowlist.

### 2026-06-12 NLU L3 local SLM target ownership

- NLU `L3LocalSLMLayer`, prompt artifact parsing, validation and benchmarking moved
  to `darjeeling.targets.nlu.layers.l3_local_slm`; old
  `darjeeling.layers.l3_local_slm` is now a compatibility re-export.
- Moved `tests/test_l3_local_slm.py` into the NLU target test scope and removed
  the old shared test path plus `src/darjeeling/layers/l3_local_slm.py` from the
  strict boundary allowlist.

### 2026-06-12 NLU L4 teacher runtime target ownership

- NLU cached/live teacher runtime moved to
  `darjeeling.targets.nlu.layers.l4_cloud_llm`; old `darjeeling.layers.l4_cloud_llm`
  is now a compatibility re-export for remaining compiler/CLI callers.
- Moved `tests/test_l4_teacher.py` into the NLU target test scope and removed the
  old shared test path from the strict boundary allowlist. The old source path
  remains allowlisted until legacy `TaskSchema` imports are migrated.

### 2026-06-12 NLU L2 student target ownership

- NLU `L2StudentConfig`, `L2StudentBundle`, `L2StudentLayer`, guard features,
  intent/slot models, slot BIO helpers and retrieval logic moved to
  `darjeeling.targets.nlu.layers.l2_student`; old `darjeeling.layers.l2_student`
  is now a compatibility re-export for remaining compiler/CLI callers.
- Moved `tests/test_l2_student_training.py` into the NLU target test scope and
  removed the old shared test path from the strict boundary allowlist. The old
  source path remains allowlisted until compiler and CLI imports are migrated off
  the legacy module.

### 2026-06-12 NLU L2 compiler helper target ownership

- NLU L2 guard threshold search, config distillation and tuner logic moved to
  `darjeeling.targets.nlu.compiler.{guard_optimizer,l2_distiller,l2_tuner}`; old
  `darjeeling.compiler.*` modules are compatibility re-exports for existing
  compiler/CLI imports.
- Moved `tests/test_l2_guard.py` and `tests/test_l2_tuner.py` into the NLU target
  test scope and removed the old shared test paths plus the old compiler helper
  source paths from the strict boundary allowlist.

### 2026-06-12 NLU L4 compiler helper target ownership

- NLU L4 teacher/proposal context rendering and proposal parsing/call adapter
  moved to `darjeeling.targets.nlu.compiler.{l4_context,l4_proposal}`; old
  `darjeeling.compiler.l4_*` modules are compatibility re-exports for existing
  compiler callers.
- Moved `tests/test_l4_context.py` and `tests/test_l4_proposal.py` into the NLU
  target test scope and removed the old shared test paths plus the old compiler
  helper source paths from the strict boundary allowlist.

### 2026-06-12 NLU L3 prompt optimizer target ownership

- NLU L3 prompt artifact proposal parsing, workspace preparation, replay gates,
  calibration and agent-session evolution moved to
  `darjeeling.targets.nlu.compiler.l3_prompt_optimizer`; old
  `darjeeling.compiler.l3_prompt_optimizer` is a compatibility re-export.
- Moved `tests/test_l3_prompt_optimizer.py` into the NLU target test scope and
  removed the old shared test path plus old compiler source path from the strict
  boundary allowlist.

### 2026-06-12 NLU L2 target evolution ownership

- NLU L2 target module evolution, local search, diagnostics, visible split/cross-audit
  gates and workspace tools moved whole to
  `darjeeling.targets.nlu.compiler.l2_target_evolution`; old
  `darjeeling.compiler.l2_target_evolution` is a compatibility re-export for CLI
  and generated tool entrypoints.
- Moved `tests/test_l2_target_evolution.py` into the NLU target test scope and
  removed the old shared test path plus old compiler source path from the strict
  boundary allowlist.

## 风险和处理

- **大文件迁移风险**：`l2_target_evolution.py` 很大。先整体迁移到 NLU target，
  再在 target 内拆 diagnostics、workspace、evaluation、program text。
- **测试大面积变更风险**：先建立 neutral core fixtures，再批量迁移 NLU tests。
- **artifact 兼容风险**：manifest 应显式记录 `target_name`、
  `target_schema_version` 和 artifact kind。旧 artifacts 可以用一次性转换脚本
  或仅在实验目录中保留。
- **CLI 用户体验风险**：保留短期 deprecation message 可以降低切换成本，但
  core 新命令必须以 `--target` 为主路径。
- **抽象税风险**：target contract 保持普通 Python Protocol，不引入复杂插件、
  DI container 或动态 schema DSL。先移动 NLU 代码到 target 边界内，再提炼共享
  helper；没有第二个 target 证明需求前，不把 NLU 内部算法抽成 core 通用框架。

## Definition Of Done

- `src/darjeeling` 中除 `targets/` 外的 core packages 不 import NLU target
  concrete modules。
- Core source/shared core tests 不包含 NLU vocabulary，boundary test 自动检查。
- Core trace、runtime、teacher cache、replay 和 promotion 使用 target-neutral
  JSON payload。
- NLU target 提供 Frame、TaskSchema、MASSIVE adapter、teacher adapter、layer
  runtime builder、compiler entry、diagnostics 和 reports。
- `edge-mvp run --target nlu` 和 `edge-mvp-nlu massive prepare` 是主路径。
- Core tests、NLU target tests 和 NLU integration smoke 全部通过。
