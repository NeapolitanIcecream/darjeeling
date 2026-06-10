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

Frame source：

- `retrieval`（默认）：从 teacher-visible prototype 中取最近邻 teacher frame，作为高精度 semantic-cache 式吸收路径。
- `student`：使用 intent classifier + slot tagger 直接生成 frame，保留为 ablation 和后续模型家族演进路径。

Decision tree 只允许在 L2，不允许在 L1。

## 阶段边界

当前开发切片已经实现 `token_sgd` 作为第一版 slot model，并将它纳入 joblib artifact、runtime routing 和 guard feature。`slot_model=none` 只保留为显式配置开关，用于 ablation 或调试，不作为 proposal 主实验的默认完成态。

第一版 `token_sgd` 的设计目标是让 L2 的吸收评估从 intent-only 推进到 frame-level：slot 训练标签来自 teacher frame 中 slot string 在 utterance token 序列里的精确连续对齐，预测时用 BIO tag 重构 slot span，并把 slot 平均概率与 invalid BIO flag 输入 guard。

当前 runtime intent classifier 使用 word + char n-gram TF-IDF 特征；`word_ngram_range` 与 `char_ngram_range` 都是实际训练参数。Intent family 当前已实现 `sgd_logreg` 和 `mlp` 两种：`sgd_logreg` 使用 `SGDClassifier(loss="log_loss")`，`mlp` 使用 sklearn `MLPClassifier`，隐藏层、alpha、early stopping 和 max iter 由 bounded config 控制。Guard 校准仍使用 train/guard split，但最终 runtime intent classifier 与 slot tagger 会在全部 teacher-visible examples 上重训，以减少小窗口下的数据浪费。

当前主线 frame source 是 `retrieval`。原因是实验显示直接 student 生成开放 slot frame 的 forced accuracy 很低，错误主要集中在 slot exact match；retrieval 路径把 L2 收缩为高精度 semantic-cache 式层，只在 replay 证明安全时吸收请求。`student` 路径继续保留，因为后续 L4 coding agent 可以演进更强的 slot extractor 或模型 family。

Retrieval prototype index 使用同一套 TF-IDF feature space 保存 teacher-visible utterance 及其 teacher frame。查询时会排除 normalized utterance 完全相同的 prototype：exact repeat 属于 L0 的职责，L2 retrieval 只处理近似重复或语义相似请求。这也避免 final threshold search 在 teacher_train 上检索到同一条训练样本而产生虚假的 100% coverage/accuracy。

Guard/calibration 训练时，prototype index 只来自 internal train split，guard split 不会进入自己的 retrieval index；runtime artifact 才使用完整 teacher-visible train window。Promotion holdout、gold eval 和 future stream 仍不可见。

L2 artifact 同时保存 intent prototype index：用同一套 TF-IDF feature space 存储 teacher-visible utterance prototype 及其 teacher intent。Runtime 预测时计算：

- 最近 teacher-visible prototype 相似度。
- predicted intent 下最近 prototype 相似度。
- predicted intent 支持度与其他 intent 最近支持度的 margin。

这些值进入 guard feature 和 trace metadata。它们不 hardcode 数据集规则，但让 guard 能识别“分类器很自信、但最近邻其实更支持别的 intent”的高风险错误。为兼容旧 artifact，guard feature 的前 5 维保持原顺序；旧 5 维 guard model 加载后会自动截断新增特征。

L2 artifact 还保存 intent calibration index：用 guard split 上的预测结果估计
`predicted intent` 和 `predicted slot signature` 的经验可靠性。Runtime 预测时把以下值输入
guard 并写入 trace metadata：

- predicted slot count / has-slots flag。
- predicted intent 的 frame exact accuracy。
- predicted intent 的 intent-only accuracy。
- predicted intent 在 calibration split 中的 support。
- predicted intent 下 slotless prediction 的比例。
- predicted `(intent, slot-name signature)` 的 frame exact accuracy。
- predicted `(intent, slot-name signature)` 的 support。
- retrieval nearest similarity、similarity margin、retrieval intent 是否匹配 student intent、frame source flag。

这不是直接 hardcode 某个 intent 可以接收，而是让 learned guard 能把 L2 收缩到历史上可靠的子域。最终是否接收仍由 deterministic threshold search 和外层 promotion replay 决定。

Slot 输出还经过两层 schema-aware 后处理：

- 按 predicted intent 的 teacher-visible slot 白名单过滤，避免给该 intent 未见过的 slot。
- 从 teacher examples 自动抽取 slot 左右上下文 pattern，作为 token tagger 漏召回时的 fallback span extractor，例如从 `how old is <person>` 或 `what does <definition_word> mean` 这类 teacher-visible pattern 泛化。主机制不写死数据集 intent 或模板，只依赖当前 examples。
- 对少数 slot-name 本身带有稳定 lexical marker 的场景，允许 guard-protected 的窄 fallback；当前只实现 `list_name` 在 singular `list` marker 前的抽取，例如 `to do list` -> `list_name=to do`。该 fallback 只在当前 predicted intent 的 slot 白名单允许该 slot、且 learned pattern 没有填充时触发；如果它降低 guard confidence，则请求会继续 fallback 到 L4。

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
- predicted slot count / has-slots flag（已实现）
- predicted intent frame exact accuracy（已实现）
- predicted intent intent-only accuracy（已实现）
- predicted intent support（已实现）
- predicted intent slotless rate（已实现）
- predicted slot signature frame exact accuracy（已实现）
- predicted slot signature support（已实现）
- retrieval nearest similarity（已实现）
- retrieval similarity margin（已实现）
- retrieval intent matches student intent（已实现）
- retrieval frame-source flag（已实现）
- slot alignment failure signals
- utterance length bucket

Runtime accept：

```text
artifact.runtime_enabled and guard_probability >= artifact.accept_threshold
```

当前 compiler 已实现 deterministic threshold search：

- grid 默认覆盖 `0.70..0.98`。
- 搜索会额外加入 teacher_train 上观测到的 guard probability 及其相邻阈值，避免安全阈值落在粗 grid 间隙时被跳过。
- 搜索必须先缓存每条 trace 的 L2 prediction，再在缓存上评估所有 threshold；不能为每个 threshold 重新调用 `bundle.predict()`，否则样本扩大后会退化为 O(N²)。
- 搜索输入只来自 `teacher_train` 内部切分出的 calibration window，不读取 promotion holdout 或 MASSIVE gold。
- 默认优先在 chronological residual validation 上校准 threshold：用 calibration train prefix 模拟 L0 exact cache，过滤 validation 中 exact repeat 和已记录的 L0/L1 accepted 请求，只在会真正到达 L2 的 residual validation traces 上评估 L2。
- 如果 residual validation 为空或 calibration train 不足以训练 L2，则回退到旧的 training-scope search，并在 `candidate_metrics["l2_guard_calibration"]` 中记录 fallback reason。
- 先过滤 `wrong_accept_rate <= l2_max_wrong_accept_rate` 且 `accepted_accuracy >= l2_min_guarded_accuracy` 的候选。
- 若存在非零覆盖且 calibration-window zero-observed-wrong 的候选，优先在这组里选 coverage 最高的阈值；否则再在 eligible 候选里按 coverage、accepted accuracy、wrong accept、threshold 排序。
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
- Proposal 只允许影响白名单字段：`frame_source`、`intent_model_family`、`slot_model_family`、`min_examples`、`max_features`、`max_iter`、`mlp_hidden_layer_sizes`、`mlp_alpha`、`mlp_early_stopping`、`word_ngram_range`、`char_ngram_range`。
- Proposal 不直接决定 accept threshold；threshold 仍由 deterministic grid search 选择。

L2 coding-agent path：

- `L2_AGENT_MODE=disabled|dry-run|codex-cli` 控制 L4 coding agent 是否为 L2 生成 patch candidate。默认 disabled，不产生 live LLM cost。
- `dry-run` 应用 fixture patch，只用于 harness 和 artifact 测试。
- `L2_AGENT_MODE` 是 legacy patch-generation path：它仍能产出可审计 patch，但 patch 指向 Darjeeling core 的 L2-owned 文件，因此不能作为 target-dependent L2 evolve 主线。
- 用户决策：真正的 L2 evolve 主路径应拆成 Outer Darjeeling loop 和 Inner L2 target-evolution loop。Darjeeling core 必须 dataset-independent；target workspace 内的 L2 runtime code 可以 target-dependent，并由 L4 coding agent 多轮演化。
- `codex-cli` 使用 GPT-5.5，独立于宿主机个人 config/rules，并使用更长 timeout；auth 仍由 Codex CLI 的 `CODEX_HOME` 机制提供。
- Legacy `codex-cli` 在隔离 autoresearch-style workspace 中运行 Codex CLI。Agent 只能修改 `candidate/` 中的 L2-owned Python source、tests 和模块设计文档；Darjeeling 宿主仓库不直接暴露为可写目标。
- Workspace 使用 `program.md + candidate/ + system/darjeeling/ + data/ + tools/`：
  - `program.md` 是稳定任务说明。
  - `candidate/` 是可 diff 的 L2 研究代码区。
  - `system/darjeeling/` 是固定 system copy，用于 overlay candidate 后验证。
  - `data/` 放 teacher-visible L2 train scope、train-visible hard cases、current metrics、objective、constraints 和命令说明。
  - `tools/inspect_context.py` 和 `tools/run_checks.py` 是标准本地入口；查看 context 使用 `python3 tools/inspect_context.py`，不需要加载 `system/darjeeling` project env。
- Prompt stdin 保持稳定短前缀，只要求 agent 读取 `program.md`。动态 context 不进入 prompt，而是作为 `data/` 文件由 agent 自主读取，以减少上下文膨胀并最大化 KV cache 复用机会。
- `data/slot_error_summary.json` 从 teacher-visible train/hard cases 中汇总 L2 wrong accept，尤其标出 intent 正确但 slot 缺失、多余或值错误的样本。该 summary 用于把下一轮 L2 evolve 的焦点放在 frame exactness，而不是只扩大 coverage。
- 调参由 Optuna 或本地 deterministic tuner 负责；L4 coding agent 负责真正需要 generalized intelligence 的代码、特征、模型 family、calibration、accept policy、验证协议和 search-space 设计。
- Agent patch adoption 以 replay/promotion success 为准；提高 raw L2 coverage 但引入 frame exactness regression 的 patch 必须撤回。
- Dataset-specific intent/slot lexical patch 不能进入 Darjeeling core；但在 isolated target workspace 的 `target/` 内，target-specific lexical/state-machine/feature code 是合法候选，只要它来自 visible target data，不读取 private holdout，并由 target holdout/promotion 指标决定是否采用。
- 当前 compiler 只记录 agent patch artifact，不在同一 Python 进程中热加载 patch：`candidate_metrics["l2_agent_patch_runtime_applied"] = false`。真实采用 patch 必须由外层开发循环应用、提交 Git、重启实验。

Inner L2 target-evolution path：

- `edge-mvp l2 target-evolve --mode agent-session` 是真实 L2 target evolve 主入口。它准备固定 target snapshot 和隔离 workspace，然后启动一个 long-running L4 agent session；agent 在 session 内自主决定 edit、evaluate、Optuna/search、debug 和 stop 的次数。
- Outer Darjeeling loop 负责 teacher-visible data split、workspace/provenance、outer promotion gate 和 core artifact 管理；不承载 target-specific L2 代码。
- Inner target workspace 使用 `program.md + target/ + system/darjeeling/ + data/ + tools/`：
  - `target/` 是唯一可写 target-dependent L2 runtime code。
  - `runs/` 是 agent/local command scratch output，不会被 promotion。
  - `target/config.json` 是 target-specific `L2StudentConfig` overrides；它可由 agent 手写，也可由 agent 调用 `tools/search_config.py`/Optuna 产生。`target_l2.py` 保留代码入口，避免 tuner 覆盖 agent 写出的 feature/postprocess 逻辑。Config overrides 只能用于恢复必要 support 且必须通过 visible audits；visible support 已达标后，不能仅为了提高 raw accepts 降低 `accept_threshold`，应优先保留 target-local veto/postprocess。
  - `system/darjeeling/` 是只读 core/evaluator copy。
  - `data/train.jsonl` 可训练、可读。
  - `data/inner_validation.jsonl` 可读，用于秒级多轮反馈。
  - `data/inner_validation_shadow_*.jsonl` 是可选的额外可见验证 folds，用于扩大 agent-visible validation pressure。
  - `data/objective.json` 可读，定义 gates、优化顺序和无效策略。
  - `data/round_state.json` 可读，只包含 baseline、agent-visible state 和历史可见 validation 聚合，不包含 private selection/promotion holdout 聚合，也不包含由 private gate 推导出的 pass/fail 布尔值。
  - `data/target_diagnostics.json` 可读，只从 visible validation、visible train audit 和可选 visible cross-audit 生成，按 teacher intent family 汇总 rejected-correct、vetoed-correct、accepted-wrong、intent-correct-slot-wrong 和少量高 guard probability 例子，用于选择下一步 target family。它显式暴露 `latest_safety_backlog`：只包含 visible validation accepted-wrong families，排序优先级高于 coverage opportunity，要求 agent 先修这些已接收错误再扩大 near-miss coverage。Accepted-wrong backlog 清空后，`latest_slot_risk_backlog`、`latest_train_audit_slot_risk_backlog` 和 `latest_visible_cross_audit_slot_risk_backlog` 会把 visible intent-correct slot mismatch family 排成队列，用于在停止前检查可能泛化成 hidden wrong accepts 的 slot/schema 风险；每个 slot-risk backlog 同时保留 count-ranked `items`、guard-ranked `high_guard_items`，以及 `missing_slot_keys`、`extra_slot_keys`、`changed_slot_keys` 计数。Slot-risk 之后，`latest_intent_confusion_backlog` 及 train/cross-audit variants 用 teacher intent / predicted intent pair 汇总高 guard wrong-intent examples。`visible_slot_cue_summary` 则从 visible train/validation teacher rows 汇总 slot key、常见 slot value 和少量例子，用于跨 intent 看到稳定 schema cue。它也暴露 `latest_train_audit_safety_backlog`，用于在 validation backlog 清空但 selection 仍失败时，从 visible train labels 里寻找更宽的安全规则；train audit 的 accepted-wrong count 是 visible safety gate，但 train coverage 不是目标。`latest_visible_cross_audit_safety_backlog` 用可见数据做 held-out retraining，提供更接近 private selection 的 safety pressure。
  - `data/commands.md` 可读，提供本地 evaluate/inspect 命令。
  - `workspace_manifest.json` 记录标准命令索引。`inspect_context` 固定为 `python3 tools/inspect_context.py`，只依赖 Python 标准库和 workspace 文件；evaluate/search 命令可使用 `uv run --project system/darjeeling ...`，并在 `commands.md` 中提供当前环境已有依赖时的 `PYTHONPATH=system/darjeeling/src python ...` fallback。
  - selection holdout 和 promotion holdout 不进入 agent workspace，存放在 outer job 的私有目录，只由 outer harness 读取。
  - `tools/evaluate.py` 在固定 split 上训练 core L2 bundle，加载 `target/target_l2.py`，然后评估 `visible_validation` 或单个 visible fold；outer harness 使用同一 evaluator 加载私有 selection/promotion holdout 做 gate。
  - `tools/search_config.py` 在可见 train/validation folds 上运行本地 Optuna config search，只写 `target/config.json`，不读取 private holdout。它是 agent 可调用工具，不是真实方法论里的外层阶段。
- `agent-session` 模式在 session 退出后、candidate evaluation 前执行 workspace scope check。候选代码只能改 `target/`，`runs/` 只作为 scratch output；`data/`、`tools/`、`system/darjeeling/` 和 `program.md` 是 protected surface。越界修改会以 `workspace_scope_violation` 停止 job，不能进入 selection/adoption。
- Inner job 可以在同一批 target data 上让 agent 自主连续迭代，不再受 `compile_every` 或 replay stream 速度限制；`rounds` 在 `agent-session` 主路径中是给 agent 的内部迭代预算提示，不表示 harness 会启动多次 Codex。Summary 必须写入 `loop_cadence.kind=fixed_trace_snapshot_inner_loop` 和 `outer_replay_cadence_bound=false`，让后续 report/agent 明确这不是“收一批样本 evolve 一次”的 outer cadence。
- Target split 默认 `chronological`，用于保持与 stream prefix 相近的时序；`--split-policy intent-stratified` 可用于小样本或窄 target patch 诊断，让 visible validation、selection 和 promotion 都覆盖更多 teacher intent family。该选项只改变 fixed target split 的采样方式，不让 private selection/promotion 进入 agent workspace，也不放宽 adoption gate。
- `--target-scope teacher_train|lower_miss` 决定进入 fixed target split 的 teacher-visible traces。默认 `teacher_train` 保留完整 teacher-labeled snapshot；`lower_miss` 会过滤掉 trace 中已被 L0/L1 接收的请求，只保留真实会落到 L2 或更高层的残差分布。该 scope 用于验证 L2 质量瓶颈是否来自 inner training/validation 分布与 runtime residual 分布不对齐。Summary、`round_state.json` 和 `objective.json` 必须写 `target_scope`，包括 input count、scoped count 和 lower-layer excluded count。Scope 不改变 private holdout 的隔离规则，也不能把 final eval 或 future stream 泄露给 agent。
- `--visible-validation-folds N` 控制 agent-visible validation 强度。未显式传入时，`standard`/`smoke` 默认 1 fold，`fixed-inner` 默认 5 folds。`N=1` 保持旧的 60/20/10/10 split 和单个 `inner_validation.jsonl`；`N>1` 默认使用 50/30/10/10 split，并把这 30% 可见 validation pool 切成 `inner_validation` + `inner_validation_shadow_*`。继续增加 fold 数只切分同一个默认 capped visible pool，不能继续压缩 train；如果需要更大的 validation pool，必须显式传 `--visible-validation-ratio`，并在 summary、`round_state.json` 和 `objective.json` 中记录 requested/effective ratio。这避免把 fold count 变成隐式 train-starvation 参数。Candidate 的 visible gate 使用所有 visible folds 的 aggregate metric，目的是降低对单一 inner split 的过拟合；private selection/promotion 仍不进入 workspace。
- `local-search` 保留为 deterministic protocol probe 和 agent 可调用工具。它不应作为和 `agent-session` 并列的真实 evolve 方法论；真实 L4 agent 可以在 session 内自行调用 `tools/search_config.py`，决定 trials、cross-audit top-k、是否接受结果以及后续代码改动。
- 默认 `compact` search space 只搜索低成本、保守的 `sgd_logreg + token_sgd` 配置和 guard/feature 参数；MLP 与 `slot_model_family=none` 留在 `wide` space 或 L4 agent 明确设计后的实验中，避免默认 tuner 用高成本 trial 或 slotless shortcut 制造 frame exactness 风险。
- `target/target_l2.py` 暴露 `accept_prediction(...)` veto hook。它只能把 core guard 原本会接收的 prediction 改为 abstain，不能强行接收 core guard 已拒绝的 prediction；metrics 记录 `vetoed_accepts` 和最多 8 条 visible `veto_examples`，让 agent 能区分“安全拒绝”与“过度保守”。
- `target/target_l2.py` 的 `postprocess_frame(...)` 是 target-code evolution 的主要修复点之一。它可以用 visible target data 推导出的解析逻辑补全缺失 slot 或修正 frame，但仍必须通过 visible validation、visible support、visible train-audit safety、private selection 和 private promotion gates；对 slot-missing 问题，优先尝试精确 postprocess，而不是继续降低 threshold。
- Evaluator 还记录最多 8 条按 guard probability 排序的 `near_miss_examples`，即 core guard 拒绝但接近阈值的 predictions，并标记 `would_be_correct`。这些 examples 在 agent-visible `round_state.json` 中只来自 visible validation，用于指导 coverage 改进；selection/promotion 的 near-miss 只属于 outer summary，不写回 workspace。由于 8 条样本太薄，workspace 还写入 bounded `target_diagnostics.json`，让 L4 agent 先按 family 选择方向，再按需查看 raw visible rows。
- `target_diagnostics.json` 中的 `latest_safety_backlog` 是 agent 的第一优先级队列。它从 visible validation 的 accepted-wrong family 生成，包含 teacher intent、accepted wrong/correct 计数、slot-wrong 计数、top predicted intents 和少量高 guard probability wrong examples。若该队列非空，L4 agent 应优先通过 `accept_prediction` veto、精确 `postprocess_frame` 或更保守的 target config 清空 backlog；只有在 visible wrong accepts 被消掉后，才应使用 `near_miss_examples` 做 coverage 扩张。
- `latest_slot_risk_backlog` 是 accepted-wrong backlog 清空后的下一队列。它只用 visible intent-correct-slot-wrong examples，让 agent 在停止或扩大 coverage 前检查 slot omission、slot value drift 和 schema-boundary 风险。`items` 按 family 频次排序，`high_guard_items` 按最高 guard probability 排序；后者用于提示低频但已经接近 accept threshold 的 schema 风险。每个 item 还列出最常见的 missing/extra/changed slot key，帮助 agent 直接看到应补全或 veto 的 schema 差异。它不是 selection/adoption gate，也不包含 private holdout rows 或 aggregates。
- `latest_intent_confusion_backlog` 是 slot-risk 之后的诊断队列。它只用 visible wrong-intent examples，按 teacher intent / predicted intent pair 暴露高 guard probability 的 intent 边界风险，例如 podcast/radio、music/radio 或其他 media-intent confusion。它不是 selection/adoption gate，也不包含 private holdout rows 或 aggregates。
- `visible_slot_cue_summary` 不是风险队列，而是可见 schema cue 索引。它按 slot key 汇总可见 teacher rows 的 `slot_key_terms`、常见 value、teacher intents 和少量 utterance examples，让 agent 能从别的 visible intents 学到 `house_place` 这类 slot cue，也能看到 `podcast_name` 这类低频但语义强的 cue，而不是只能等待某个 intent/slot pair 在 validation fold 中出现。它的使用准则是检查 slotless/missing-slot accepted frames 是否遗漏了这些可见 cue；若遗漏，优先做 target-local veto 或精确 postprocess。Agent program 会显式要求检查 podcast cue 被非 podcast intent 接收、room value 被 slotless frame 接收、generic radio station 被当成具体 `radio_name`、radio/music cue 缺 `media_type`、calendar remove 缺 `date`、bare upcoming events 被接成 `recommendation_events`、joke adjective/superlative cue 缺 `joke_type`、以及 spoken volume amount 缺 `change_amount` 的情况，并可用 `tools/evaluate.py --split slot_cue_probes` 运行 visible-only synthetic probes。该 probe split 是 diagnostic-only，不参与 selection/adoption gate。
- `latest_train_audit_safety_backlog` 是第二优先级诊断队列。它用同一个训练好的 L2 bundle 在 visible train rows 上评估 target behavior，目的是发现 validation folds 没覆盖到的 slot-risk family。它不能证明 candidate 可采用，也不能替代 private selection/promotion；它只允许 agent 在不读取 private rows 的情况下设计更稳健的 target safety rules。
- `latest_visible_cross_audit_safety_backlog` 是第三层、但更接近 selection 的诊断队列。它把 visible train + visible validation rows 按 teacher intent 分成若干 held-out folds，每折重新训练 L2 bundle 并在 held-out visible fold 上评估 target code。`fixed-inner` profile 默认启用 3 折；`standard`/`smoke` 默认关闭。该指标用于发现单一 visible validation split 清空后仍可能隐藏的 wrong accepts，不参与 private candidate selection 或 adoption。
- L4 coding agent session 应优先用于 `target/` 中的结构性改动：新特征、模型 family、校准方法、postprocess、abstain 机制和 search-space 设计。超参搜索本身交给 `tools/search_config.py`，由 agent 在同一 session 内按需调用，避免把 GPT-5.5 token 用在手工猜参数上。
- 每个 job 先评估 unmodified baseline，再启动 agent session 并评估最终 target candidate。Visible validation improvement 的排序把 wrong accepts 放在 coverage 之前：提高 raw coverage 但引入 frame exactness regression 不算进步。
- 默认 `standard` 预算策略仍是 cost-capped probe；真实固定 snapshot 探索应显式用 `--budget-profile fixed-inner`。`agent-session` 默认只启动一个 live agent session，agent 在该 session 内自行反复 edit/evaluate/search；`--max-agent-rounds 0` 只准备 workspace/baseline/context，不启动 Codex。旧 `codex-cli` 多 round 模式、`local-search` mode 和 `dry-run` mode 保留用于兼容、smoke 和回归测试，不是新的首选实验路径。
- Summary、`data/round_state.json` 和 `data/objective.json` 必须写入 `budget_policy.profile_intent` 和 `evidence_policy`。`profile_intent` 说明本次 profile 的设计意图；`evidence_policy` 说明本次产物是否有资格支撑质量结论。`standard` 会标为 `cost_capped_probe`，`smoke` 会标为 `connectivity_smoke`，失败都不能被解释为 L2 evolve 方向已耗尽。只有足够预算、teacher-labeled snapshot size 足够、且 `agent-session` 至少完成一次 scoped candidate evaluation 的 `fixed-inner` run 才能标为 `fixed_snapshot_research`；如果 agent 未启动、命令失败、workspace scope violation 或 0 个 candidate 被评估，只能作为 incomplete/no-launch probe。显式把 `--rounds` 或 `--max-agent-rounds` 压得过低时，即使用了 `fixed-inner` profile，也只能作为 short/budget-capped probe。若 teacher-labeled traces 少于 500，则只能作为 small-snapshot probe，用于调试和方向探索，不能作为正式质量证据。`fixed_snapshot_research` 仍不是 adoption，本身还需要 private selection/promotion gates 和 outer e2e replay 证明。
- Candidate selection gate 要求 visible validation gate、visible support gate、visible train-audit accepted-wrong safety gate 和 private selection holdout gate 同时通过。Visible support gate 要求每个 visible validation fold 至少保留 2 个 correct accepts，用来挡住只靠大幅 abstain 取得 0 wrong 的 near-zero coverage candidate；raw private selection 通过但 visible validation、visible support 或 train-audit safety 失败时只能作为诊断信号，不能成为 selected candidate。小 holdout 上的 zero-accept 或 single-accept 结果只适合作为 inner-loop model-selection signal，不能替代外层 e2e replay。
- Summary 同时记录 diagnostic `best_round`、`best_selection_round`、`selection_decision` 和 adoption-oriented `best_adoptable_round` / `adoption_decision`。即使某轮 visible validation 变好，只要 candidate selection/promotion holdout 不过 gate，就不能被视为可采用 target candidate。
- Summary 还记录 `private_holdout_evidence`，用于区分 gate 失败原因：例如 `visible_support_gate_failed` 表示 visible validation 过关但 support 太薄，`selection_zero_accepts_for_inner_passing_rounds` 表示 private selection split 没观察到 candidate accepts，而不是观察到了错误。该字段只写 outer summary/promoted manifest，不写入 agent-visible `round_state.json` 或 `target_diagnostics.json`，避免把 private holdout aggregate feedback 泄露给 L4 agent。
- 每次 agent session 或 legacy round 评估后都必须 snapshot 当时的 `target/` 到 `rounds/round_NNN_target/`，并在 payload 中写入 `target_snapshot`。`best_round` 和 adoption decision 指向的是某次被评估的 target snapshot，因此 promotion 必须复制该 snapshot，而不是盲目复制最终 workspace 的 `target/`。
- `best_round` 的主排序仍以 private selection split 为准；当 selection 指标完全并列时，使用 visible validation 作为 tie-break，再偏向后续轮次。这允许 non-adopted outer replay stage 到真实改进过的 target snapshot，同时不把可见集优化凌驾于 private holdout gate 之上。
- 通过 adoption gate 的 target candidate 可由 `edge-mvp l2 promote-target` 转成 runtime artifact。未通过 adoption gate 的 `best_round` 只能在显式 `--allow-non-adopted` 下 stage 到隔离 run，用于外层 replay 诊断；manifest 必须标记 `l2_target_inner_adopted=false`，不能被误读为 inner-adopted artifact。
- `edge-mvp l2 replay-target` 是 target artifact 的正式 outer replay gate：candidate 默认取 current manifest，baseline 默认取 parent manifest，在同一批 teacher-labeled traces 上输出 `l2-target-outer-replay-v1`。默认 `accuracy_epsilon=0`，并包含 settings L1 Rust worker，确保 target candidate 不能用 frame exactness regression 换 L4 share。无论 inner 是否 adopted，最终是否采用仍以 3k/10k e2e replay 的 frame exactness、wrong accept 和 L4 share 为准。
- `promote-target` 不改 Darjeeling core：它写入新的 `l2_student.joblib` 和 `l2_target` module artifact；runtime replay 与 compiler offline replay 加载同一 target wrapper，保持评估/运行语义一致。若普通 compiler generation 重新训练 core L2 bundle，则必须丢弃继承来的 `l2_target`，除非该 generation 明确做了 target-aware adoption；否则会把 target code 和不兼容 bundle 混用。
- target-dependent lexical/code patches 允许存在于 `target/`，但必须从可见 train/validation-fold 数据推导，不能依赖 MASSIVE 或外部 dataset 知识。单条 visible row 的 exact utterance exception 或 request-id 记忆化不允许作为泛化机制；agent 应优先写有多个 visible 支持或清晰 schema 语义的 pattern-level lexical/slot-support 规则。是否采用由 holdout/promotion 指标和 outer replay 决定，而不是由 dataset-independent core 规则决定。Summary 和 promoted manifest 必须记录 `target_code_policy`，其中明确 `core_must_remain_dataset_independent=true`、`target_specific_code_is_not_rejected_for_dataset_dependence=true` 和 single-row memorization 禁止规则。
- 当前实现状态：已支持 baseline-first `agent-session` single-launch 主路径、legacy `dry-run`/`local-search`/`codex-cli` 兼容路径、target workspace evaluator、可见多 fold validation、private holdout gate、visible `tools/search_config.py`、local-search trial report 和机器可读 `evidence_policy`。下一轮真实 L2 实验应优先使用 `agent-session`，让 agent 在一个 session 内自行调用 evaluate/search。

Optuna tuning path：

- `edge-mvp l2 tune --traces <trace.jsonl> --out <report.json>` 是 L4 coding agent 可调用的本地工具接口。
- `L2_TUNING_MODE=optuna` 时，compiler 在每个 generation 中先对 L2 training scope 做内部 train/validation split，再运行 Optuna search，最后用 best config 训练 runtime candidate。
- Tuning report 使用 `l2-tune-v1`，记录每个 trial 的 params、最终 `L2StudentConfig`、split policy、validation unguarded/guarded metrics 和 p95 latency。
- Optuna 不能读取 promotion holdout、MASSIVE gold、final eval 或 future stream；它只优化 teacher-visible train window 的内部 residual validation。
- `candidate_metrics["l2_tuning"]` 记录 trial 数、best value、best metrics；`artifact_paths["l2_tuning"]` 指向完整 tuning report。
- Tuning validation 默认使用 `chronological` split，即用 teacher-visible train window 的尾部模拟 future stream。`stratified_random` 只作为 ablation 开关保留；它会让每个 intent 更均匀，但容易高估真实 L0/L1 miss 后续分布。
- 在 tuning validation 上也会应用同一套 residual filter：从 validation 中移除 calibration train prefix 的 exact repeat 和已记录 L0/L1 accepted 请求，并记录 `validation_residual_size`、`objective_validation_size` 与 `objective_validation_source`。若 residual 为空，才回退到未过滤 validation，防止样本过小时 tuning 完全失效。
- Compact search 和小样本窗口默认不启用 MLP `early_stopping`，因为 sklearn 会在过小 validation split 上拒绝训练，浪费 trial budget。
- `L2_TUNING_MIN_EXAMPLES` 是 tuning 的硬门槛；样本不足时 compiler 仍训练 deterministic L2 candidate，但跳过 Optuna 并记录 skip reason，避免用几十个样本制造虚假的 tuning 结论。

L2 training scope 是显式实验开关：

- `L2_TRAINING_SCOPE=teacher_train` 是默认值。它让 L2 继续学习完整 teacher-visible 分布，满足“L(N) 应比 L(N-1) 有更高泛化性，只是更慢”的层级假设。
- `L2_TRAINING_SCOPE=lower_miss` 只使用当前 train window 中 L0/L1 未接收、需要更高层处理的 teacher traces。它用于验证“tuning validation 与真实 L0/L1 miss 分布对齐”是否能提升 promoted L2 的实际吸收率。
- `lower_miss` 不是默认主线，因为它有设计风险：若 L2 只学习低层 miss 分布，可能退化为补丁层，而不是对 L1 更泛化的一层。实验报告必须同时记录 full teacher train count、lower-miss count 和实际 target count。
- 无论 scope 如何，promotion holdout、regression sample 和 final eval 都不能进入 tuning 或训练；scope 只改变 candidate-generation 可见的 `teacher_train` 子集。
- 当 scope 不是 `teacher_train` 时，compiler 额外记录 full `teacher_train` 上的 unguarded diagnostics，用来观察 specialization 是否牺牲总体泛化。

## MLP evolve path

MLP 不是替换默认 L2 的硬编码选择，而是一个可复现实验 family：

- `L2_INTENT_MODEL_FAMILY=mlp` 可以直接启用 deterministic MLP candidate。
- `edge-mvp experiment l2-mlp` 固定开启 MLP intent family，用于与 baseline 同场 replay。
- `edge-mvp experiment l2-tuned` 开启 Optuna tuning，用于验证本地搜索是否优于 baseline/固定 MLP。
- `L4_PROPOSAL_MODE=live` 时，L4 可以在 bounded config 中提议 `intent_model_family=mlp` 及相关参数；更强路径是 L4 coding agent 修改 L2 代码或 search space 后再调用 Optuna。
- `candidate_metrics["l2_config"]` 记录最终训练配置，避免实验结果只留下自然语言描述。
