# data 模块

模块根：当前 NLU 数据形状在 `darjeeling.targets.nlu.data`。

## `targets.nlu.data` records

职责：

- 定义当前 NLU target 的 `DataRecord`。
- 当前 replay compatibility path 读取该记录格式；target-neutral core replay
  应改为读取 target JSON payload，并交给 selected target 解释。
- NLU 记录可以包含 utterance、gold frame、workload group key、annotation 和
  template；这些字段不能成为 core 默认 contract。

## `targets.nlu.adapters.massive`

职责：

- 使用 `datasets.load_dataset("AmazonScience/massive", locale, trust_remote_code=True)` 下载 MASSIVE。
- 依赖约束固定为 `datasets>=2.20.0,<4.0.0`，因为 `AmazonScience/massive` 当前是脚本型 dataset repo；`datasets` 4/5 系列不再支持该 script loading 路径。
- `trust_remote_code=True` 是 target CLI 非交互初始化的一部分，避免
  `edge-mvp-nlu massive prepare` 在脚本型 dataset 安全提示处阻塞。
- 将 train/dev/test 处理为稳定的本地 parquet/jsonl。
- 生成 NLU `DataRecord`，包含 utterance、gold frame 和可选 workload group key。
  MASSIVE 的 annotated utterance / normalized template 可写入兼容字段，但不是
  core 必需 contract。

约束：

- `gold_frame` 只用于 eval/report。
- Runtime/compiler/L1 agent 只接收 request id、utterance 和 teacher-visible trace。
- `prepare` 阶段可以构造 workload group key 和 gold frame，但这些字段不能进入
  compiler context。

## `targets.nlu.data` frame helpers

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
