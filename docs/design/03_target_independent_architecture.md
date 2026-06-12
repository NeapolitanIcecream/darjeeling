# 03 Target-Independent 架构

Darjeeling 由一个 target-independent core 和一组 target package 组成。
Core 负责分层执行、trace、compiler harness、replay、promotion、artifact
和质量 gate。Target 负责解释输入、输出、schema、teacher prompt、训练逻辑、
诊断和报告中的任务语义。

NLU 是一个 target。MASSIVE 是 NLU target 的一个 dataset adapter。

## 架构分区

```text
darjeeling core
  contracts.py
  runtime/
  compiler/
  artifacts/
  eval/
  layers/base.py

targets.nlu
  schemas.py
  settings.py
  main_cli.py
  cli.py
  teacher.py
  data.py
  layers/
    l1_native.py
    l2_student.py
    l3_prompt.py
  compiler.py
  diagnostics.py
  reports.py
  adapters/
    massive.py
```

Core packages outside `targets/` 不 import `targets.nlu` 的具体模块。Core 通过 target contract 调用
目标任务能力。CLI 或配置负责选择 target，并把 target object 传给 core。
Target discovery 使用 repo 内静态 registry，例如 target name 到 constructor
的显式 mapping；不引入 packaging entry point、动态插件系统或 DI container。

Target package 可以 import core 的协议、trace、artifact、workspace 和 replay
工具。Target package 不能修改 core 的 promotion authority，也不能绕过 core
的 visible/private 数据边界。

## Core 职责

Core 保留以下职责：

- 固定层级：`L0 -> L1 -> L2 -> L3 -> L4`。
- 路由：逐层调用 runtime layer，遇到 accepted output 停止。
- Trace：记录 request、teacher label、final output、layer results、usage、
  latency、cost 和 metadata。
- Teacher transport：调用 L4、retry、timeout、usage 提取、cache 读写。
- Artifact：manifest、generation 目录、promotion record、artifact set 装载。
- Compiler harness：visible/private split、hard buffer、candidate generation
  编排、agent workspace 生命周期、transcript/provenance、scope check。
- Replay：在 holdout、regression sample 和 hard buffer 上重放 artifact set。
- Promotion：correctness、wrong accept、latency、cost、complexity 和 per-layer
  regression gate。
- Settings：target-independent 的 runtime、teacher transport、agent harness、
  replay、promotion、cost/latency 假设。
- Core reports：通用 coverage、correctness、wrong accept、latency、cost、
  layer share、promotion decision、artifact lineage。

Core 可以保存 target-owned JSON payload，但不解释 payload 内部字段。

Core 文件中不出现任务语义名词，例如 NLU 的 `Frame`、`intent`、`slot`、
`utterance`、`frame_exact_match`、`intent_confusion` 或 `slot_risk`。这些词只
能出现在 target package、adapter、target-owned workspace、target tests、
experiment evidence 中。

## Target 职责

Target package 保留以下职责：

- 定义 request schema、label schema、output schema 和 task schema。
- 将 dataset adapter 输出转换成 target records。
- 标准化 request，用于 exact cache、重复检测和 workload locality。
- 构建 L4 teacher prompt，解析 teacher response，校验 teacher label。
- 判断 output 与 teacher label 是否正确。
- 定义 target-specific runtime layers 的 artifact 格式和加载方式。
- 训练或生成 target-specific L1/L2/L3 artifacts。
- 提供 target-specific diagnostics、probe contract 和 report sections。
- 提供 target-specific CLI 子命令。

Target 可以理解自己的业务字段。NLU target 可以自由使用 `utterance`、
`Frame(intent, slots)`、slot cue、intent confusion 和 frame exact match。

## 低抽象税约束

Target boundary 是代码所有权边界，不是新的 framework 层。实现时遵守：

- 一个 target 可以先由一个普通 Python object 实现全部 target contract；只有
  当文件变大或职责真正分裂时，才拆成 teacher/runtime/compiler/report 子对象。
- 文档里的 `Protocol` 是 typing view，不是必须继承的 base class，也不是运行时
  注册机制。
- Target 选择使用静态 registry，不使用 entry point、插件 marketplace、
  service locator 或 DI container。
- Core/target 之间只传 JSON payload、artifact manifest 和少量显式 context；
  不设计通用 schema DSL。
- 先把 NLU 代码移动到 target 边界内，再提炼共享 helper。没有第二个 target
  证明需求前，不把 NLU 内部算法抽成 core 通用框架。

## Core Data Model

Core trace 使用 target-neutral 名称：

```python
JsonValue = None | bool | int | float | str | list["JsonValue"] | dict[str, "JsonValue"]
JsonObject = dict[str, JsonValue]

LayerName = Literal["L0", "L1", "L2", "L3", "L4"]

class LayerResult(BaseModel):
    layer: LayerName
    accepted: bool
    output: JsonObject | None = None
    confidence: float | None = None
    reason: str = ""
    latency_ms: float
    cost_usd: float = 0.0
    metadata: JsonObject = Field(default_factory=dict)

class TraceRecord(BaseModel):
    request_id: str
    input: JsonObject
    gold_label: JsonObject | None = None
    teacher_label: JsonObject | None = None
    chosen_layer: LayerName
    final_output: JsonObject
    layer_results: list[LayerResult]
    l4_usage: JsonObject = Field(default_factory=dict)
    timestamp: str

class TeacherTrace(BaseModel):
    request_id: str
    input: JsonObject
    teacher_label: JsonObject | None = None
    chosen_layer: LayerName
    final_output: JsonObject
    layer_results: list[LayerResult]
    l4_usage: JsonObject = Field(default_factory=dict)
    timestamp: str
```

`TeacherTrace` 不包含 `gold_label`，并设置 `extra="forbid"`。Compiler、
agent context、target training、guard training 和 target diagnostics 使用
`TeacherTrace` 或更窄的 teacher-visible view。

## Target Contract

Target contract 是普通 Python object，不要求插件系统。

```python
class TargetSpec(Protocol):
    name: str
    schema_version: str

    def load_task_schema(self, records: Sequence[JsonObject]) -> JsonObject: ...

    def normalize_request(self, input: JsonObject) -> str: ...

    def validate_output(self, output: JsonObject, task_schema: JsonObject) -> None: ...

    def labels_equal(
        self,
        output: JsonObject,
        expected: JsonObject,
        *,
        task_schema: JsonObject,
    ) -> bool: ...

    def summarize_for_context(
        self,
        traces: Sequence[TeacherTrace],
        *,
        budget: int,
    ) -> JsonObject: ...
```

`labels_equal` 是 promotion correctness 的唯一 target-specific 判断入口。
Core 不直接比较 target payload。

## Teacher Contract

Core 拥有 L4 transport，target 拥有 prompt 和 parser。

```python
class TeacherAdapter(Protocol):
    prompt_version: str

    def build_messages(
        self,
        *,
        input: JsonObject,
        task_schema: JsonObject,
    ) -> list[dict[str, str]]: ...

    def parse_response(
        self,
        raw_response: str,
        *,
        task_schema: JsonObject,
    ) -> JsonObject: ...

    def cache_key_parts(self, *, task_schema: JsonObject) -> JsonObject: ...
```

Teacher cache lines store target-neutral fields plus target-provided key parts:

```json
{
  "cache_key": "...",
  "input": {},
  "normalized_request": "...",
  "target_name": "nlu",
  "target_schema_version": "...",
  "prompt_version": "...",
  "teacher_label": {},
  "raw_response": "...",
  "usage": {},
  "created_at": "..."
}
```

## Runtime Layer Contract

Core runtime layers all implement one target-neutral shape:

```python
class RuntimeLayer(Protocol):
    layer_name: LayerName

    def try_answer(self, input: JsonObject) -> LayerResult: ...
```

Target runtime loads target artifacts and returns layers as a mapping. Core owns
the fixed route order and skips missing layers:

```python
class TargetRuntime(Protocol):
    def build_layers(
        self,
        *,
        manifest: ArtifactManifest,
        teacher: TeacherRuntime,
        settings: JsonObject,
    ) -> Mapping[LayerName, RuntimeLayer | None]: ...
```

L0 exact cache can be implemented in core when it stores:

```text
target.normalize_request(input) -> target-owned output JSON
```

L1 native subprocess management can be implemented in core when the target
provides the request/response ABI adapter. Core owns build, process lifetime,
timeout and benchmark mechanics; target owns the JSON fields and output schema.

## Compiler Contract

Core compiler loop calls one target compiler entry point and remains the
promotion authority. The target decides internally which L1/L2/L3 work to run
for that generation.

```python
class TargetCompiler(Protocol):
    def propose_artifacts(
        self,
        context: CompileContext,
    ) -> Sequence[ArtifactCandidate]: ...
```

Core handles:

```text
teacher-visible split
private selection and promotion holdouts
hard-buffer visibility
workspace creation
protected root checks
candidate replay
promotion decision
manifest update
```

Target handles:

```text
target-specific context packing
target-specific training
target-specific agent instructions inside editable target workspace
target-specific diagnostics and probes
target artifact serialization
```

## Replay And Promotion

Replay routes each trace through candidate layers. When a layer accepts, core
asks target whether the output is correct:

```python
correct = target.labels_equal(
    output=result.output,
    expected=trace.teacher_label,
    task_schema=task_schema,
)
```

Core computes generic metrics:

- correctness rate
- accepted accuracy
- wrong accept rate
- layer coverage
- p50/p95 latency
- cost per 100 requests
- artifact complexity
- per-layer deltas

Target may add named metrics, but target metrics do not replace core promotion
authority unless the target contract declares them as additional gates.

## Agent Workspaces

Core owns workspace mechanics:

```text
workspace/
  program.md
  workspace_manifest.json
  contexts/
  tools/
  runs/
  target-editable-root/
```

Core guarantees:

- private holdout rows are outside the workspace;
- protected roots are hashed before and after agent execution;
- transcript, commands, diff and provenance are recorded;
- candidate artifacts are snapshotted before replay;
- agent-visible outputs cannot self-promote.

Target owns:

- editable root contents;
- target-specific `program.md` sections;
- target-specific context summaries;
- target-specific validation/evaluation tools;
- target-specific probe files.

## NLU Target

The NLU target defines:

```python
class Frame(BaseModel):
    intent: str
    slots: dict[str, str] = Field(default_factory=dict)
    is_abstain: bool = False

class TaskSchema(BaseModel):
    intent_names: list[str]
    slot_names: list[str]
    schema_version: str
```

NLU request payload:

```json
{"utterance": "target example text"}
```

NLU output/label payload:

```json
{
  "intent": "intent_name",
  "slots": {
    "slot_name": "slot value"
  },
  "is_abstain": false
}
```

NLU owns:

- utterance normalization;
- task schema discovery from NLU records;
- teacher prompt that asks for strict intent/slots JSON;
- teacher response parser into `Frame`;
- frame exact equality;
- L1 native ABI for utterance-to-frame program banks;
- L2 intent classifier, slot tagger, retrieval, learned guard features;
- L2 target extension functions such as `postprocess_frame` and
  `accept_prediction`;
- L3 local SLM prompt artifact, parser and intent/slot validator;
- slot cue, slot-risk and intent-confusion diagnostics;
- NLU report sections.

## MASSIVE Adapter

MASSIVE belongs under the NLU target adapter layer:

```text
targets.nlu.adapters.massive
  AmazonScience/massive row
  -> NLU request payload
  -> NLU Frame gold label
  -> target record JSONL/parquet
```

The MASSIVE adapter may know:

- Hugging Face dataset name and config;
- MASSIVE split names and locale names;
- MASSIVE row fields such as raw utterance and annotated utterance;
- how annotated spans map to NLU slot values;
- how MASSIVE intent ids map to NLU intent names.

Core does not import the MASSIVE adapter. Core only reads target records through
the selected target's data contract.

Other NLU datasets use sibling adapters:

```text
targets.nlu.adapters.massive
targets.nlu.adapters.atis
targets.nlu.adapters.snips
targets.nlu.adapters.internal_assistant
```

All of them produce the same NLU target payload shape.

## CLI Shape

Core CLI selects a target and delegates target-specific subcommands:

```text
edge-mvp run --target nlu --data-dir data/processed/default
edge-mvp report --target nlu --run-dir runs/latest
edge-mvp experiment preflight --target nlu --run-dir runs/latest
```

Target CLIs own preparation and specialized training/evolution commands:

```text
edge-mvp-nlu massive prepare --locale en-US --out data/processed/massive-en-us
edge-mvp-nlu l2 train --traces runs/latest/traces.jsonl --out ...
edge-mvp-nlu l2 target-evolve --traces runs/latest/traces.jsonl --out-dir ...
edge-mvp-nlu l3 prompt-evolve --traces runs/latest/traces.jsonl --out-dir ...
```

Core CLI should not expose NLU-only option names. Target CLI may expose
intent/slot/frame vocabulary.

Current implementation note: `edge-mvp` is packaged as the core CLI
(`darjeeling.cli`) and dispatches to a target object from the static registry.
The NLU target implements runtime building through `NluTargetRuntime` and the
compiler entry through `NluTargetCompiler`. `edge-mvp-nlu` owns NLU preparation
and specialized workflow commands such as MASSIVE prepare, L2 target evolve and
L3 prompt evolve.

## Boundary Tests

Core boundary tests scan core source and shared core tests. Outside target
packages, adapters, target fixtures and experiment evidence, core source must
not contain:

```text
Frame
intent
slot
utterance
TaskSchema(intent_names, slot_names)
frame_exact_match
intent_confusion
slot_risk
```

NLU target tests cover the NLU behavior directly. Adapter tests cover concrete
dataset mapping. Experiment docs may include dataset-specific evidence, but it
does not become a core default or reusable core rule.
