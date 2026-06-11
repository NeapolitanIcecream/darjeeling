# data 模块

模块根：`darjeeling.data`

## `data.records`

职责：

- 定义 dataset-independent `DataRecord`。
- Core runtime/replay 只依赖该记录格式，不依赖具体数据集 loader。
- 必需字段只表达通用 request、utterance 和 label；adapter-specific annotation
  或 template 字段只能作为可选兼容字段或 metadata。

## `adapters.massive`

职责：

- 使用 `datasets.load_dataset("AmazonScience/massive", locale, trust_remote_code=True)` 下载 MASSIVE。
- 依赖约束固定为 `datasets>=2.20.0,<4.0.0`，因为 `AmazonScience/massive` 当前是脚本型 dataset repo；`datasets` 4/5 系列不再支持该 script loading 路径。
- `trust_remote_code=True` 是 adapter CLI 非交互初始化的一部分，避免 `edge-mvp-massive prepare` 在脚本型 dataset 安全提示处阻塞。
- 将 train/dev/test 处理为稳定的本地 parquet/jsonl。
- 生成通用 `DataRecord`，包含 utterance、gold frame 和可选 workload group key。
  MASSIVE 的 annotated utterance / normalized template 可写入兼容字段，但不是
  core 必需 contract。

约束：

- `gold_frame` 只用于 eval/report。
- Runtime/compiler/L1 agent 只接收 request id、utterance 和 teacher-visible trace。
- `prepare` 阶段可以构造 workload group key 和 gold frame，但这些字段不能进入
  compiler context。

## `data.frames`

职责：

- bracket annotation parser。
- frame equality 和 slot normalization。
- teacher slot 到 token span 的 best-effort alignment。

Slot alignment：

- L2 slot training 需要 BIO tag，但 teacher frame 只有 slot string。
- 用 normalized substring matching 将 teacher slot value 对齐到 utterance token span。
- 同一 slot value 多次出现时选择最短、最左、未占用 span。
- 无法对齐的 slot 记录 `alignment_failure`。
- 不使用 gold annotation 补齐 teacher slot alignment。

## `data.streams`

职责：

- 从真实 utterance 构造 replay stream。
- 支持 `uniform`、`zipf-mild`、`zipf-heavy`。
- 将抽样结果写入 `runs/<id>/stream.json`。

Zipf stream 可以使用 `workload_group_key` 形成 workload locality；没有该字段时
回退到 intent + normalized utterance。Group key 不进入 runtime/compiler 输入。
