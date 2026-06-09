# L2 student 模块

模块：`darjeeling.layers.l2_student`

## 职责

- 训练 tiny student。
- 训练 learned guard。
- runtime 中先 student predict，再 guard decide。

## 子模块

- `l2_features.py`: utterance vectorizer 和 token features。
- `l2_intent.py`: intent classifier families。
- `l2_slots.py`: token-level slot model。
- `l2_guard.py`: guard feature extraction 和 classifier。
- `l2_bundle.py`: joblib serialization。
- `l2_student.py`: runtime layer。

## Candidate families

Intent：

- `sgd_logreg`
- `mlp`
- `small_decision_tree`

Slot：

- `token_sgd`
- `mlp_token_classifier`
- `none`

Decision tree 只允许在 L2，不允许在 L1。

## 阶段边界

当前开发切片已经实现 `token_sgd` 作为第一版 slot model，并将它纳入 joblib artifact、runtime routing 和 guard feature。`slot_model=none` 只保留为显式配置开关，用于 ablation 或调试，不作为 proposal 主实验的默认完成态。

第一版 `token_sgd` 的设计目标是让 L2 的吸收评估从 intent-only 推进到 frame-level：slot 训练标签来自 teacher frame 中 slot string 在 utterance token 序列里的精确连续对齐，预测时用 BIO tag 重构 slot span，并把 slot 平均概率与 invalid BIO flag 输入 guard。

当前 runtime intent classifier 使用 word + char n-gram TF-IDF 特征；`word_ngram_range` 与 `char_ngram_range` 都是实际训练参数。Guard 校准仍使用 train/guard split，但最终 runtime intent classifier 与 slot tagger 会在全部 teacher-visible examples 上重训，以减少小窗口下的数据浪费。

L2 artifact 同时保存 intent prototype index：用同一套 TF-IDF feature space 存储 teacher-visible utterance prototype 及其 teacher intent。Runtime 预测时计算：

- 最近 teacher-visible prototype 相似度。
- predicted intent 下最近 prototype 相似度。
- predicted intent 支持度与其他 intent 最近支持度的 margin。

这些值进入 guard feature 和 trace metadata。它们不 hardcode 数据集规则，但让 guard 能识别“分类器很自信、但最近邻其实更支持别的 intent”的高风险错误。为兼容旧 artifact，guard feature 的前 5 维保持原顺序；旧 5 维 guard model 加载后会自动截断新增特征。

Slot 输出还经过两层 schema-aware 后处理：

- 按 predicted intent 的 teacher-visible slot 白名单过滤，避免给该 intent 未见过的 slot。
- 从 teacher examples 自动抽取 slot 左右上下文 pattern，作为 token tagger 漏召回时的 fallback span extractor，例如从 `how old is <person>` 或 `what does <definition_word> mean` 这类 teacher-visible pattern 泛化。该机制不写死数据集 intent、slot 或模板，只依赖当前 examples。

仍需作为实验限制记录：

- 对齐只支持 teacher slot value 与 utterance token 的规范化后连续匹配。
- slot value 输出为 token 级规范化重构，不保留原始大小写、标点或复杂规范化形式。
- 该模型不是神经序列标注器，不保证跨域泛化；若 slot 分布明显复杂，需要在 L2 内替换为更强 family，但 L1 仍只吸收 profile 里稳定的 native path。

## Distillation data

训练数据是：

```text
utterance -> L4 teacher_frame
```

不使用 MASSIVE gold。

## Guard training

```text
student_frame = student.predict(x)
teacher_frame = teacher(x)
correct = frame_equal(student_frame, teacher_frame)
guard.train(features -> correct)
```

Guard features：

- top1 intent probability（已实现）
- margin（已实现）
- entropy（已实现）
- slot average probability（已实现）
- invalid BIO flag（已实现）
- nearest teacher trace similarity（已实现）
- predicted intent nearest similarity（已实现）
- predicted-vs-other intent support margin（已实现）
- predicted intent recent error rate
- slot alignment failure signals
- utterance length bucket

Runtime accept：

```text
artifact.runtime_enabled and guard_probability >= artifact.accept_threshold
```

当前 compiler 已实现 deterministic threshold search：

- grid 默认覆盖 `0.70..0.98`。
- 搜索会额外加入 teacher_train 上观测到的 guard probability 及其相邻阈值，避免安全阈值落在粗 grid 间隙时被跳过。
- 搜索输入只来自 `teacher_train`，不读取 promotion holdout 或 MASSIVE gold。
- 先过滤 `wrong_accept_rate <= l2_max_wrong_accept_rate` 且 `accepted_accuracy >= l2_min_guarded_accuracy` 的候选。
- 若存在非零覆盖且 train-window zero-observed-wrong 的候选，优先在这组里选 coverage 最高的阈值；否则再在 eligible 候选里按 coverage、accepted accuracy、wrong accept、threshold 排序。
- 选中的 threshold 写入 `L2StudentConfig.accept_threshold`，并记录到 `candidate_metrics["l2_guard_search"]`。
- 若 teacher-visible examples 少于 `L2_MIN_RUNTIME_EXAMPLES`，compiler 仍训练 L2、记录 unguarded/guard diagnostics，但将 `L2StudentConfig.runtime_enabled=false`，runtime 只记录 prediction metadata，不接收。
- Compiler 同时记录 `candidate_metrics["l2_unguarded_train"]`，即 threshold=0 时的 train-window frame accuracy、wrong accept 和 coverage，用于区分 student 本体质量问题与 guard 过严问题。
- `L4_PROPOSAL_MODE=live` 时，L4 可以提议 threshold grid 和 max wrong-accept 上限；该 proposal 写入 `guard_candidate`，但最终 threshold 仍由 deterministic search 选择。

## L4 参与方式

L4 对 L2 使用 direct model API，输出 bounded config JSON。L4 不生成训练代码，不参与多轮 coding-agent session。

当前接入状态：

- `L4_PROPOSAL_MODE=disabled` 为默认值，不调用 L4 proposal。
- `L4_PROPOSAL_MODE=live` 时，compiler 请求 L2 config proposal。
- Proposal 只允许影响白名单字段：`slot_model_family`、`min_examples`、`max_features`、`max_iter`、`word_ngram_range`、`char_ngram_range`。
- Proposal 不直接决定 accept threshold；threshold 仍由 deterministic grid search 选择。
