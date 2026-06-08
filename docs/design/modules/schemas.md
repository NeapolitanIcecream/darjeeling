# schemas 模块

模块：`darjeeling.schemas`

## 职责

- 定义跨模块共享的数据契约。
- 避免包含重依赖、I/O、训练逻辑、prompt 文本或 artifact 读写。
- 保持所有模型可 JSON 序列化，便于 trace、manifest、context 和测试使用。

## 核心 schema

```python
class Frame(BaseModel):
    intent: str
    slots: dict[str, str] = Field(default_factory=dict)
    is_abstain: bool = False

class LayerResult(BaseModel):
    layer: Literal["L0", "L1", "L2", "L3", "L4"]
    accepted: bool
    frame: Frame | None
    confidence: float | None
    reason: str
    latency_ms: float
    cost_usd: float
    metadata: dict[str, Any] = Field(default_factory=dict)

class TraceRecord(BaseModel):
    request_id: str
    utterance: str
    gold_frame: Frame | None
    teacher_frame: Frame | None
    chosen_layer: str
    final_frame: Frame
    layer_results: list[LayerResult]
    l4_usage: dict[str, Any]
    timestamp: str

class TeacherTrace(BaseModel):
    request_id: str
    utterance: str
    teacher_frame: Frame | None
    chosen_layer: str
    final_frame: Frame
    layer_results: list[LayerResult]
    l4_usage: dict[str, Any]
    timestamp: str
```

## Gold label 隔离

`TeacherTrace` 不包含 `gold_frame`，并应设置 `extra="forbid"`。所有 compiler、context builder、L1 coding-agent prompt pack、L2 distillation、L3 prompt optimization 和 guard training 的入口都必须使用 `TeacherTrace` 或更窄的数据结构。

测试要求：

- compiler-visible payload 序列化后不含 `gold_frame`。
- L4 context payload 序列化后不含 `gold_frame`、`gold_intent`、`gold_slots` 等字段。
- L1 agent workspace 输入目录不包含 gold 文件或 final eval 文件。

## 版本

以下内容必须显式带 version：

- frame schema
- intent schema
- slot schema
- prompt template
- context layout
- artifact manifest
- teacher cache key

不能依赖代码版本隐式推断 schema。
