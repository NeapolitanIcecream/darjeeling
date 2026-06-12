# schemas / contracts 模块

模块：core 使用 `darjeeling.contracts`；NLU schema 位于
`darjeeling.targets.nlu.schemas`。

## Core 职责

- 定义 target-neutral JSON payload 类型。
- 定义 runtime trace、layer result 和 target contract 的稳定形状。
- 只传递 target-owned request、label、output 和 metadata，不解释其中字段。
- 保持模型可 JSON 序列化，便于 trace、manifest、context 和测试使用。

## Core contract

```python
JsonValue = None | bool | int | float | str | list["JsonValue"] | dict[str, "JsonValue"]
JsonObject = dict[str, JsonValue]

class LayerResult(BaseModel):
    layer: Literal["L0", "L1", "L2", "L3", "L4"]
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

`TeacherTrace` 设置 `extra="forbid"`，并由 `TraceRecord` 去除 `gold_label`
后生成。Compiler、context builder、target training、guard training 和 target
diagnostics 必须使用 `TeacherTrace` 或更窄的数据结构。

## Target schema

NLU 的 `Frame(intent, slots)`、`TaskSchema`、legacy NLU `TraceRecord` 和
teacher-view helpers 属于 `darjeeling.targets.nlu.schemas`。这些名字不能作为 core
默认 contract，也不能出现在 core source 或 shared core tests 中。

## 版本

以下内容必须显式带 version：

- target name 和 target schema version
- prompt template version
- context layout version
- artifact manifest schema version
- teacher cache key parts

Target 可以在自己的 schema 中继续记录 frame、intent、slot 或 dataset adapter
版本，但 core 只把它们当作 JSON payload 或 target-provided cache key parts。
