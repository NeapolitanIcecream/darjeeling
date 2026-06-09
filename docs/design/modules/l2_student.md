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

当前 runtime intent classifier 使用 word + char n-gram TF-IDF 特征；`word_ngram_range` 与 `char_ngram_range` 都是实际训练参数。Intent family 当前已实现 `sgd_logreg` 和 `mlp` 两种：`sgd_logreg` 使用 `SGDClassifier(loss="log_loss")`，`mlp` 使用 sklearn `MLPClassifier`，隐藏层、alpha、early stopping 和 max iter 由 bounded config 控制。Guard 校准仍使用 train/guard split，但最终 runtime intent classifier 与 slot tagger 会在全部 teacher-visible examples 上重训，以减少小窗口下的数据浪费。

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

用户决策：L2 evolve 拆成两类工作。

- 调参交给 Optuna 或同类本地 optimizer。
- 真正需要 generalized intelligence 的设计工作由 L4 coding agent 承担，包括修改 L2 代码、设计特征管线、模型家族、calibration、accept policy、验证协议和 Optuna search space。

旧的 direct L4 bounded config proposal 仍保留为轻量 proposal path，但它不是最终 L2 evolve 主路径。

当前接入状态：

- `L4_PROPOSAL_MODE=disabled` 为默认值，不调用 L4 proposal。
- `L4_PROPOSAL_MODE=live` 时，compiler 请求 L2 config proposal。
- Proposal 只允许影响白名单字段：`intent_model_family`、`slot_model_family`、`min_examples`、`max_features`、`max_iter`、`mlp_hidden_layer_sizes`、`mlp_alpha`、`mlp_early_stopping`、`word_ngram_range`、`char_ngram_range`。
- Proposal 不直接决定 accept threshold；threshold 仍由 deterministic grid search 选择。

Optuna tuning path：

- `edge-mvp l2 tune --traces <trace.jsonl> --out <report.json>` 是 L4 coding agent 可调用的本地工具接口。
- `L2_TUNING_MODE=optuna` 时，compiler 在每个 generation 中先对 `teacher_train` 做内部 train/validation split，再运行 Optuna search，最后用 best config 训练 runtime candidate。
- Tuning report 使用 `l2-tune-v1`，记录每个 trial 的 params、最终 `L2StudentConfig`、validation unguarded/guarded metrics 和 p95 latency。
- Optuna 不能读取 promotion holdout、MASSIVE gold、final eval 或 future stream；它只优化 teacher-visible train window 的内部 validation。
- `candidate_metrics["l2_tuning"]` 记录 trial 数、best value、best metrics；`artifact_paths["l2_tuning"]` 指向完整 tuning report。
- Compact search 和小样本窗口默认不启用 MLP `early_stopping`，因为 sklearn 会在过小 validation split 上拒绝训练，浪费 trial budget。

## MLP evolve path

MLP 不是替换默认 L2 的硬编码选择，而是一个可复现实验 family：

- `L2_INTENT_MODEL_FAMILY=mlp` 可以直接启用 deterministic MLP candidate。
- `edge-mvp experiment l2-mlp` 固定开启 MLP intent family，用于与 baseline 同场 replay。
- `edge-mvp experiment l2-tuned` 开启 Optuna tuning，用于验证本地搜索是否优于 baseline/固定 MLP。
- `L4_PROPOSAL_MODE=live` 时，L4 可以在 bounded config 中提议 `intent_model_family=mlp` 及相关参数；更强路径是 L4 coding agent 修改 L2 代码或 search space 后再调用 Optuna。
- `candidate_metrics["l2_config"]` 记录最终训练配置，避免实验结果只留下自然语言描述。
