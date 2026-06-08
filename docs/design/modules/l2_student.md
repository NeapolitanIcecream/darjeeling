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
- nearest teacher trace similarity
- predicted intent recent error rate
- slot alignment failure signals
- utterance length bucket

Runtime accept：

```text
guard_probability >= artifact.accept_threshold
```

当前 compiler 已实现第一版 deterministic threshold grid search：

- grid 默认覆盖 `0.70..0.98`。
- 搜索输入只来自 `teacher_train`，不读取 promotion holdout 或 MASSIVE gold。
- 先过滤 `wrong_accept_rate <= l2_max_wrong_accept_rate` 的候选，再优先选择 coverage 高、accepted accuracy 高、wrong accept 低的 threshold。
- 选中的 threshold 写入 `L2StudentConfig.accept_threshold`，并记录到 `candidate_metrics["l2_guard_search"]`。
- `L4_PROPOSAL_MODE=live` 时，L4 可以提议 threshold grid 和 max wrong-accept 上限；该 proposal 写入 `guard_candidate`，但最终 threshold 仍由 deterministic search 选择。

## L4 参与方式

L4 对 L2 使用 direct model API，输出 bounded config JSON。L4 不生成训练代码，不参与多轮 coding-agent session。

当前接入状态：

- `L4_PROPOSAL_MODE=disabled` 为默认值，不调用 L4 proposal。
- `L4_PROPOSAL_MODE=live` 时，compiler 请求 L2 config proposal。
- Proposal 只允许影响白名单字段：`slot_model_family`、`min_examples`、`max_features`、`max_iter`、`word_ngram_range`、`char_ngram_range`。
- Proposal 不直接决定 accept threshold；threshold 仍由 deterministic grid search 选择。
