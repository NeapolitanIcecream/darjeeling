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
  - `tools/inspect_context.py` 和 `tools/run_checks.py` 是标准本地入口。
- Prompt stdin 保持稳定短前缀，只要求 agent 读取 `program.md`。动态 context 不进入 prompt，而是作为 `data/` 文件由 agent 自主读取，以减少上下文膨胀并最大化 KV cache 复用机会。
- `data/slot_error_summary.json` 从 teacher-visible train/hard cases 中汇总 L2 wrong accept，尤其标出 intent 正确但 slot 缺失、多余或值错误的样本。该 summary 用于把下一轮 L2 evolve 的焦点放在 frame exactness，而不是只扩大 coverage。
- 调参由 Optuna 或本地 deterministic tuner 负责；L4 coding agent 负责真正需要 generalized intelligence 的代码、特征、模型 family、calibration、accept policy、验证协议和 search-space 设计。
- Agent patch adoption 以 replay/promotion success 为准；提高 raw L2 coverage 但引入 frame exactness regression 的 patch 必须撤回。
- Dataset-specific intent/slot lexical patch 默认不纳入主线，除非 replay-backed hard case 证明必要，且实现方式可泛化。
- 当前 compiler 只记录 agent patch artifact，不在同一 Python 进程中热加载 patch：`candidate_metrics["l2_agent_patch_runtime_applied"] = false`。真实采用 patch 必须由外层开发循环应用、提交 Git、重启实验。

Inner L2 target-evolution path：

- `edge-mvp l2 target-evolve` 是新的内层 target loop 入口。
- Outer Darjeeling loop 负责 teacher-visible data split、workspace/provenance、outer promotion gate 和 core artifact 管理；不承载 target-specific L2 代码。
- Inner target workspace 使用 `program.md + target/ + system/darjeeling/ + data/ + tools/`：
  - `target/` 是唯一可写 target-dependent L2 runtime code。
  - `target/config.json` 是 local-search/Optuna 产生的 target-specific `L2StudentConfig` overrides；`target_l2.py` 保留代码入口，避免 tuner 覆盖 agent 写出的 feature/postprocess 逻辑。
  - `system/darjeeling/` 是只读 core/evaluator copy。
  - `data/train.jsonl` 可训练、可读。
  - `data/inner_validation.jsonl` 可读，用于秒级多轮反馈。
  - `data/objective.json` 可读，定义 gates、优化顺序和无效策略。
  - `data/round_state.json` 可读，只包含 baseline 和历史轮次的 inner validation 聚合，不包含 promotion holdout 聚合。
  - `data/commands.md` 可读，提供本地 evaluate/inspect 命令。
  - selection holdout 和 promotion holdout 不进入 agent workspace，存放在 outer job 的私有目录，只由 outer harness 读取。
  - `tools/evaluate.py` 在固定 split 上训练 core L2 bundle，加载 `target/target_l2.py`，然后评估 inner validation；outer harness 使用同一 evaluator 加载私有 selection/promotion holdout 做 gate。
  - `tools/search_config.py` 在可见 train/inner validation 上运行本地 Optuna config search，只写 `target/config.json`，不读取 private holdout。
- Inner loop 可以在同一批 target data 上连续跑多轮，不再受 `compile_every` 或 replay stream 速度限制；`rounds` 是最大轮数，不是必须烧满的轮数。
- `local-search` round 是 cheap tuning round，不是 LLM round。它在固定 target data 上跑多次 trial，选择 visible inner validation 更优的配置；若没有超过当前 target，则回滚 `target/config.json`。Outer harness 随后私下评估 selection/promotion holdout，local-search 不能自证 adoption。
- 默认 `compact` search space 只搜索低成本、保守的 `sgd_logreg + token_sgd` 配置和 guard/feature 参数；MLP 与 `slot_model_family=none` 留在 `wide` space 或 L4 agent 明确设计后的实验中，避免默认 tuner 用高成本 trial 或 slotless shortcut 制造 frame exactness 风险。
- `target/target_l2.py` 暴露 `accept_prediction(...)` veto hook。它只能把 core guard 原本会接收的 prediction 改为 abstain，不能强行接收 core guard 已拒绝的 prediction；metrics 记录 `vetoed_accepts` 和最多 8 条 visible `veto_examples`，让 agent 能区分“安全拒绝”与“过度保守”。
- Evaluator 还记录最多 8 条按 guard probability 排序的 `near_miss_examples`，即 core guard 拒绝但接近阈值的 predictions，并标记 `would_be_correct`。这些 examples 在 agent-visible `round_state.json` 中只来自 inner validation，用于指导 coverage 改进；selection/promotion 的 near-miss 只属于 outer summary，不写回 workspace。
- L4 coding agent round 应优先用于 `target/` 中的结构性改动：新特征、模型 family、校准方法、postprocess、abstain 机制和 search-space 设计。超参搜索本身交给 `tools/search_config.py` 或 `--mode local-search`，避免把 GPT-5.5 token 用在手工猜参数上。
- 每个 job 先评估 unmodified baseline，再评估后续 target rounds。Inner improvement 的排序把 wrong accepts 放在 coverage 之前：提高 raw coverage 但引入 frame exactness regression 不算进步。
- 默认预算策略是 `inner_patience_rounds=2` 和 `stop_on_selection_gate=true`。连续两轮没有 inner validation improvement 会提前停止；candidate selection gate 通过也会停止。
- Candidate selection gate 要求 visible inner validation gate 和 private selection holdout gate 同时通过；raw private selection 通过但 visible inner 失败时只能作为诊断信号，不能成为 selected candidate。
- Summary 同时记录 diagnostic `best_round`、`best_selection_round`、`selection_decision` 和 adoption-oriented `best_adoptable_round` / `adoption_decision`。即使某轮 inner validation 变好，只要 candidate selection/promotion holdout 不过 gate，就不能被视为可采用 target candidate。
- target-dependent lexical/code patches 允许存在于 `target/`，但必须从可见 train/inner validation 数据推导，不能依赖 MASSIVE 或外部 dataset 知识。是否采用由 holdout/promotion 指标决定，而不是由 dataset-independent core 规则决定。
- 当前实现状态：已支持 baseline-first `dry-run`/`local-search`/`codex-cli` 多轮、target workspace evaluator、private holdout gate、inner-validation patience stop、visible `tools/search_config.py` 和 local-search trial report；`codex-cli` 多轮入口已预留在 harness 中，但尚未作为主实验默认使用。

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
