# MVP Demo Proposal: Profile-Guided Edge Intelligence Runtime

> 交付目标：做一个真实运行的 MVP demo，展示云端 LLM 如何把高频、低熵、可验证的 workload 子分布逐步编译到更便宜的层：cache、CPU program、tiny student、local SLM。这个 demo 的关键不是最终模型有多强，而是实验中真实出现“舍弃泛化性换取局部最优”的过程。

---

## 0. Codex agent 执行摘要

请实现一个 Python repo，使用 `uv` 管理环境，运行一个端侧虚拟助手 NLU replay demo。

运行时层级固定为：

```text
L0: exact / semantic cache
L1: L4-evolved train-free CPU ProgramBank
L2: L4-distilled tiny trainable student + learned guard
L3: L4-optimized local SLM prompt
L4: cloud LLM teacher/compiler/fallback
```

核心循环：

```text
request stream
  -> cascade runtime routes request through L0/L1/L2/L3/L4
  -> trace everything
  -> every K requests, compiler runs
  -> L4 proposes artifacts for L1/L2/L3/guards
  -> system trains/runs/replays candidates for real
  -> evaluator promotes only candidates that improve objective
  -> next replay window uses promoted artifacts
```

The demo must not be a mock. It is acceptable to cache previously observed L4 responses for reproducibility and cost control, but those cached responses must have been produced by real L4 API calls and stored in trace files. If `OPENAI_API_KEY` is absent and no valid teacher cache exists, the main demo must fail fast rather than silently switch to fake labels.

---

## 1. Product thesis to demonstrate

The thesis is:

```text
A strong layer should preserve capability ceiling.
Weaker layers should absorb hot, low-entropy, locally verifiable subdomains.
Guard/gating/fallback prevent weak layers from damaging end-to-end capability.
```

This demo should visibly show:

```text
L1 forced global accuracy: low or mediocre
L1 guarded accepted accuracy: high
L1 coverage: grows with compiler iterations
L2 coverage: grows through distillation
L3 improves through prompt/example/schema optimization
L4 calls, token cost, and latency: decrease over time
full cascade accuracy: remains close to all-L4 baseline
```

The important message is not “small models replace large models.” The important message is:

```text
The system learns where weaker representations are sufficient.
```

---

## 2. Non-goals

Do not spend MVP complexity on:

```text
privacy/compliance stories
security boundary productization
distributed serving
multi-agent framework
RL training
fine-tuning local SLM
mobile app UI
production observability stack
```

Also avoid abstract framework bloat. No LangChain / AutoGen / LlamaIndex unless absolutely necessary. Direct Python modules are preferred.

---

## 3. Dataset and scenario

### 3.1 Primary scenario

Use a text-only virtual assistant command-understanding task.

Input:

```text
"set an alarm for seven tomorrow morning"
```

Output frame:

```json
{
  "intent": "alarm_set",
  "slots": {
    "time": "seven",
    "date": "tomorrow morning"
  }
}
```

### 3.2 Primary dataset

Use `AmazonScience/massive`, config `en-US`, via Hugging Face `datasets`.

Reason:

```text
- real virtual-assistant NLU setting
- intent + slot annotations
- enough data to construct replay streams
- text-only, so no ASR/audio engineering distraction
- English subset is small enough for local iteration
```

Known dataset facts to rely on:

```text
MASSIVE 1.1: >1M utterances, 52 languages, 18 domains, 60 intents, 55 slot types.
en-US split size: train 11514, dev 2033, test 2974.
Hugging Face loader: load_dataset("AmazonScience/massive", "en-US", split="train")
```

References:

- https://huggingface.co/datasets/AmazonScience/massive
- https://arxiv.org/abs/2204.08582

### 3.3 Gold label policy

Gold labels from MASSIVE are allowed only for evaluation/reporting.

They must not be visible to:

```text
- router
- compiler
- L1 proposal prompts
- L2 student training
- L3 prompt optimization
- guard training
```

Lower layers learn from L4 teacher frames, not from dataset gold. This is required to demonstrate “L4 teaches weaker layers.”

Implementation requirement:

```text
DataRecord.gold_frame exists only inside evaluator/report code.
TraceRecord.teacher_frame is the only label visible to compiler/runtime artifacts.
```

Add a unit test that fails if compiler inputs contain `gold_frame`.

---

## 4. Environment and dependency selection

### 4.1 Environment manager

Use `uv`.

Required commands:

```bash
uv sync
uv run edge-mvp prepare --locale en-US
uv run edge-mvp run --stream zipf-heavy --max-requests 3000 --compile-every 500 --teacher live-or-cache
uv run edge-mvp report --run-dir runs/latest
uv run pytest
```

### 4.2 Python version

Use Python 3.12 unless a dependency forces 3.11. Prefer Python 3.12.

### 4.3 Core dependencies

Use these dependencies unless there is a strong implementation reason to change them.

```toml
[project]
name = "edge-intelligence-runtime-mvp"
version = "0.1.0"
requires-python = ">=3.11,<3.13"
dependencies = [
  "openai>=1.0.0",
  "pydantic>=2.0.0",
  "pydantic-settings>=2.0.0",
  "typer>=0.12.0",
  "rich>=13.0.0",
  "datasets>=2.20.0",
  "numpy>=1.26.0",
  "pandas>=2.2.0",
  "pyarrow>=15.0.0",
  "scikit-learn>=1.5.0",
  "joblib>=1.4.0",
  "sentence-transformers>=3.0.0",
  "faiss-cpu>=1.8.0",
  "torch>=2.3.0",
  "transformers>=4.45.0",
  "accelerate>=0.33.0",
  "json-repair>=0.30.0",
  "jinja2>=3.1.0",
  "matplotlib>=3.9.0",
  "plotly>=5.22.0",
]

[project.optional-dependencies]
dev = [
  "pytest>=8.0.0",
  "ruff>=0.6.0",
]

[project.scripts]
edge-mvp = "edge_mvp.cli:app"
```

Dependency rationale:

```text
openai: official OpenAI-compatible L4 client
pydantic: strict schema for frames, traces, artifact proposals
pydantic-settings: env/config loading
Typer + Rich: minimal CLI with readable progress
Hugging Face datasets: no custom dataset downloader
scikit-learn: L2 student and guard models
sentence-transformers + faiss-cpu: semantic cache and nearest-trace features
transformers + torch + accelerate: real local SLM for L3
json-repair: tolerate local SLM JSON formatting failures
jinja2: prompt templates
matplotlib/plotly: reports and experiment curves
```

Avoid adding orchestration frameworks. Keep code explicit.

### 4.4 LLM environment variables

L4 must use local environment variables:

```bash
export OPENAI_API_KEY="..."
export OPENAI_BASE_URL="https://.../v1"   # optional; if unset, use SDK default
export OPENAI_MODEL="..."                 # required by config, default can be set in settings
```

Implementation:

```python
from openai import OpenAI

client = OpenAI(
    api_key=settings.openai_api_key,
    base_url=settings.openai_base_url or None,
)
```

Do not hardcode any API key.

### 4.5 Local SLM choice

Default L3 model:

```text
Qwen/Qwen2.5-0.5B-Instruct
```

Configurable via:

```bash
export LOCAL_SLM_MODEL="Qwen/Qwen2.5-0.5B-Instruct"
```

Optional stronger local model:

```text
Qwen/Qwen2.5-1.5B-Instruct
```

Use `transformers` direct loading for MVP. Do not require vLLM server in the first implementation. Keep the local SLM path real but simple.

---

## 5. Repository layout

```text
edge-intelligence-runtime-mvp/
  pyproject.toml
  uv.lock
  README.md
  .env.example

  src/edge_mvp/
    __init__.py
    cli.py
    settings.py
    schemas.py

    data/
      massive.py
      frames.py
      streams.py

    runtime/
      router.py
      trace.py
      cost.py
      timing.py

    layers/
      base.py
      l0_cache.py
      l1_program_bank.py
      l2_student.py
      l3_local_slm.py
      l4_cloud_llm.py

    compiler/
      loop.py
      mining.py
      l0_compile.py
      l1_program_compiler.py
      l2_distiller.py
      l3_prompt_optimizer.py
      guard_optimizer.py
      objective.py
      replay.py

    artifacts/
      store.py
      generated_l1.py              # generated file, may be absent initially

    eval/
      metrics.py
      experiments.py
      reports.py
      plots.py

  tests/
    test_frame_parser.py
    test_gold_leakage.py
    test_l1_dsl.py
    test_l2_guard.py
    test_replay_promotion.py
```

Generated run files:

```text
runs/<timestamp>/
  settings.json
  traces.jsonl
  teacher_cache.jsonl
  artifacts/
    l0_cache.json
    l0_semantic.faiss
    l1_programs.json
    generated_l1.py
    l2_student.joblib
    l2_guard.joblib
    l3_prompt.json
    router_config.json
  reports/
    summary.md
    curves.html
    metrics.csv
    artifacts.csv
    hard_cases.jsonl
```

---

## 6. Core schemas

Use Pydantic models. Keep schemas simple and explicit.

```python
class Frame(BaseModel):
    intent: str
    slots: dict[str, str] = Field(default_factory=dict)
    is_abstain: bool = False

class LayerResult(BaseModel):
    layer: Literal["L0", "L1", "L2", "L3", "L4"]
    accepted: bool
    frame: Frame | None = None
    confidence: float | None = None
    reason: str = ""
    latency_ms: float
    cost_usd: float = 0.0
    metadata: dict[str, Any] = Field(default_factory=dict)

class TraceRecord(BaseModel):
    request_id: str
    utterance: str
    gold_frame: Frame | None = None          # evaluator only
    teacher_frame: Frame | None = None       # compiler-visible label
    chosen_layer: str
    final_frame: Frame
    layer_results: list[LayerResult]
    l4_usage: dict[str, Any] = Field(default_factory=dict)
    timestamp: str
```

Important separation:

```text
Compiler functions receive TeacherTrace, not TraceRecord.
TeacherTrace must not include gold_frame.
```

---

## 7. Runtime behavior

### 7.1 Router

Use a fixed sequential cascade for MVP:

```text
L0 -> L1 -> L2 -> L3 -> L4
```

No learned global router in MVP. The guards inside each layer are enough.

Pseudocode:

```python
def route(utterance: str) -> Frame:
    results = []
    for layer in [l0, l1, l2, l3]:
        r = layer.try_answer(utterance)
        results.append(r)
        if r.accepted:
            return r.frame

    r4 = l4.answer(utterance)
    results.append(r4)
    return r4.frame
```

Shadow mode:

```text
During cold start, L0/L1/L2/L3 may run in shadow, but final output must come from L4.
During warm run, accepted lower-layer output becomes final.
For audit, optionally call L4 on a small sample of accepted lower-layer outputs to collect disagreement.
```

---

## 8. Layer designs

## 8.1 L0: exact / semantic cache

### Exact cache

Key:

```text
normalize(utterance)
```

Value:

```text
teacher_frame from previous real L4 call
```

Accept condition:

```text
normalized utterance exists in exact cache
```

### Semantic cache

Use sentence embeddings and FAISS.

Artifact:

```json
{
  "clusters": [
    {
      "cluster_id": "...",
      "centroid_id": 12,
      "frame": {"intent": "alarm_set", "slots": {}},
      "support": 35,
      "teacher_purity": 0.98,
      "threshold": 0.88
    }
  ]
}
```

Accept condition:

```text
nearest cluster similarity >= threshold
cluster support >= min_support
cluster teacher purity >= min_purity
```

L4 can propose L0 thresholds, but deterministic replay chooses final thresholds.

---

## 8.2 L1: L4-evolved train-free CPU ProgramBank

L1 must not be a trained decision tree or any fitted statistical model.

L1 is a bank of generated CPU programs. Each program is a partial evaluator:

```text
match narrow condition -> return frame
else -> abstain
```

### Required implementation shape

Use a bounded DSL as the L4 output format, then render it into a real Python module.

Flow:

```text
L4 outputs JSON DSL candidates
  -> validator checks schema
  -> renderer writes generated_l1.py
  -> replay imports generated_l1.py
  -> candidate is evaluated by actually executing Python functions
```

This gives us real CPU programs without arbitrary unbounded code generation.

### Minimal DSL

Operators:

```text
contains(term)
contains_all([term, ...])
contains_any([term, ...])
starts_with([prefix, ...])
regex(pattern)
regex_extract(pattern, slot_map)
not(condition)
and([condition, ...])
or([condition, ...])
```

Actions:

```text
accept(intent, slots)
abstain
```

Example DSL:

```json
{
  "rule_id": "alarm_set_001",
  "description": "set alarm commands with explicit time",
  "condition": {
    "and": [
      {"contains_any": ["alarm", "wake me"]},
      {"regex_extract": {
        "pattern": "(?:for|at) (?P<time>.+)$",
        "slot_map": {"time": "time"}
      }}
    ]
  },
  "action": {
    "accept": {
      "intent": "alarm_set",
      "slots_from_regex": true
    }
  }
}
```

Rendered shape:

```python
def rule_alarm_set_001(q: str) -> Frame | None:
    qn = normalize(q)
    m = re.search(r"(?:for|at) (?P<time>.+)$", qn)
    if ("alarm" in qn or "wake me" in qn) and m:
        return Frame(intent="alarm_set", slots={"time": m.group("time")})
    return None
```

### L1 compiler prompt input

For each hot cluster, provide L4 with:

```text
- task schema
- allowed DSL grammar
- positive examples: utterance + teacher_frame
- negative examples: nearby utterances with different teacher_frame
- hard cases: previous wrong accepts or disagreements
- existing rule summaries
```

Do not provide gold labels.

### L1 promotion rule

A rule is eligible only if replay shows:

```text
accepted_count >= min_support
accepted_accuracy >= l1_min_precision
wrong_accept_count <= max_wrong_accepts
code_size <= max_code_size
```

Suggested defaults:

```text
min_support = 10
l1_min_precision = 0.98
max_wrong_accepts = 1 on replay holdout
```

### L1 metrics

Report both:

```text
forced global accuracy: if L1 had to answer everything, using best matching rule or fallback guess
protected/guarded accuracy: among accepted requests only
coverage: accepted / total
latency: measured wall-clock runtime
```

The forced metric is included to demonstrate that L1 has sacrificed generality.

---

## 8.3 L2: L4-distilled tiny trainable student + learned guard

L2 is the main trainable weak layer.

It should contain:

```text
student model: predicts frame or frame components
guard model: predicts whether student is likely correct on this request
```

### L2 candidate families

L4 may propose bounded config candidates. It must not generate arbitrary training code.

Supported intent model families:

```text
sgd_logreg
mlp
small_decision_tree
```

Default preference:

```text
MLP or SGD logistic regression before decision tree.
Decision tree is allowed only as an L2 candidate, never as L1.
```

Supported slot model families:

```text
token_sgd
mlp_token_classifier
none
```

For MVP, implement `token_sgd` first:

```text
- parse MASSIVE annot_utt into BIO tags
- featurize tokens with local window features
- train sklearn SGDClassifier or MLPClassifier over token labels
- reconstruct slot spans from predicted BIO tags
```

This avoids writing a neural sequence tagger while still being a real trainable tiny model.

### L2 config schema

```json
{
  "config_id": "mlp_128_char_word_v1",
  "intent_model": {
    "family": "mlp",
    "hidden_layer_sizes": [128],
    "alpha": 0.0001,
    "max_iter": 50
  },
  "vectorizer": {
    "word_ngram_range": [1, 2],
    "char_ngram_range": [3, 5],
    "max_features": 50000
  },
  "slot_model": {
    "family": "token_sgd",
    "window": 2
  },
  "guard": {
    "family": "logreg",
    "features": [
      "top1_prob",
      "margin",
      "entropy",
      "slot_avg_prob",
      "slot_invalid_bio",
      "nearest_trace_similarity",
      "predicted_intent_recent_error_rate"
    ],
    "accept_threshold": 0.93
  }
}
```

### L2 distillation data

Training data:

```text
utterance -> L4 teacher_frame
```

Do not train from MASSIVE gold.

### L2 guard training

During shadow/replay:

```text
student_frame = L2.predict(x)
teacher_frame = L4.teacher(x)
correct = frame_equal(student_frame, teacher_frame)
train guard on guard_features -> correct
```

At runtime:

```text
accepted = guard_prob >= threshold
```

### L2 promotion rule

Evaluate each L2 config on replay holdout + hard buffer.

Promote if:

```text
system objective improves
wrong_accept_rate <= l2_max_wrong_accept_rate
guarded_accuracy >= l2_min_guarded_accuracy
coverage is non-trivial
```

Suggested defaults:

```text
l2_min_guarded_accuracy = 0.93
l2_max_wrong_accept_rate = 0.05
min_coverage = 0.10 after enough teacher traces
```

---

## 8.4 L3: L4-optimized local SLM prompt

L3 is a real local SLM. Do not stub it.

### Local model

Default:

```text
Qwen/Qwen2.5-0.5B-Instruct
```

Configurable:

```text
LOCAL_SLM_MODEL
```

### L3 output format

Local SLM must return JSON compatible with `Frame`:

```json
{
  "intent": "alarm_set",
  "slots": {"time": "seven tomorrow morning"},
  "confidence": 0.82
}
```

Use `json-repair` only to recover malformed JSON. Record whether repair was needed.

### L3 prompt artifact

```json
{
  "prompt_id": "l3_prompt_gen_004",
  "system_prompt": "...",
  "schema_text": "...",
  "few_shot_example_ids": ["trace_12", "trace_91"],
  "intent_shortlist_policy": {
    "type": "hot_intents_plus_nearest",
    "k": 12
  },
  "generation": {
    "temperature": 0.0,
    "max_new_tokens": 256
  },
  "acceptance": {
    "require_json_valid": true,
    "min_self_confidence": 0.70,
    "reject_if_slot_schema_invalid": true
  }
}
```

### L3 compiler

L4 proposes prompt candidates using:

```text
- current L3 failures/disagreements
- hot intents
- few-shot candidate pool from teacher traces
- output schema
- allowed prompt fields
```

L4 may propose example IDs from the provided pool. It should not invent few-shot labels.

### L3 promotion rule

Run the real local SLM on replay holdout. Promote prompt if:

```text
end-to-end cascade objective improves
L3 guarded accuracy improves or coverage improves without accuracy drop
JSON parse failure rate does not increase materially
```

---

## 8.5 L4: cloud LLM teacher + compiler + fallback

L4 has three roles:

```text
1. runtime fallback
2. teacher labeler
3. compiler/proposer for L1/L2/L3/guard artifacts
```

L4 must use the OpenAI SDK with `OPENAI_API_KEY` and optional `OPENAI_BASE_URL`.

### L4 teacher call

Prompt requirements:

```text
- provide valid intent list
- provide slot schema
- ask for strict JSON only
- include abstain/oos only if configured for OOS experiments
```

The returned teacher frame is used as the label for L0/L1/L2/L3 optimization.

### L4 compiler calls

Prefer separate calls for clarity:

```text
compile_l1_program_candidates(...)
compile_l2_config_candidates(...)
compile_l3_prompt_candidates(...)
compile_guard_candidates(...)
```

Each call must return strict JSON validated by Pydantic.

### Critical rule

L4 can propose. L4 cannot certify.

Only replay/evaluator can promote artifacts.

---

## 9. Compiler loop

Run compiler every `compile_every` requests after cold start.

Suggested default:

```text
cold_start = 500 requests
compile_every = 500 requests
max_requests = 3000 requests
```

Compiler algorithm:

```text
1. Load teacher-visible traces.
2. Mine hot normalized utterances, hot intents, and embedding clusters.
3. Build hard buffer from disagreements and wrong accepts.
4. Generate L0 candidates deterministically.
5. Ask L4 for L1 ProgramBank candidates.
6. Ask L4 for L2 config candidates.
7. Train each L2 student candidate for real.
8. Ask L4 for L3 prompt candidates.
9. Run each L3 prompt candidate through the real local SLM on replay sample.
10. Ask L4 for guard/gating candidate thresholds, or run deterministic threshold grid search.
11. Evaluate candidate artifact sets with replay.
12. Promote the best artifact set only if it improves objective and respects wrong-accept constraints.
13. Persist artifact lineage and metrics.
```

### Candidate set size

Keep candidate count small for MVP:

```text
L1: up to 5 new rule candidates per compiler iteration
L2: up to 4 model configs per iteration
L3: up to 3 prompt candidates per iteration
Guard thresholds: grid of 5-10 values per layer
```

---

## 10. Replay/evaluator

Replay is the selection authority.

Each candidate is evaluated on:

```text
- recent replay holdout
- hard buffer
- small sample from older traces to avoid regression
```

Do not let L4 override replay results.

### Metrics

Frame-level metrics:

```text
intent_accuracy
slot_micro_f1
frame_exact_match
```

Layer metrics:

```text
coverage
accepted_accuracy
wrong_accept_rate
forced_global_accuracy
p50_latency_ms
p95_latency_ms
cost_usd_per_100_requests
```

System metrics:

```text
L4_calls_per_100_requests
cloud_input_tokens_per_100_requests
cloud_output_tokens_per_100_requests
cloud_cost_per_100_requests
p50_latency_ms
p95_latency_ms
end_to_end_intent_accuracy
end_to_end_frame_exact_match
layer_share: L0/L1/L2/L3/L4
```

### Objective

Use a simple weighted score:

```text
score =
  + 100.0 * frame_exact_match
  - 200.0 * wrong_accept_rate
  -   1.0 * cost_usd_per_100_requests
  -   0.01 * p95_latency_ms
  -   0.001 * artifact_complexity
```

Make weights configurable in `settings.yaml`.

Promotion gate:

```text
candidate_score > current_score
and candidate_end_to_end_accuracy >= current_accuracy - epsilon
and wrong_accept_rate <= configured limit
```

Suggested:

```text
epsilon = 0.02
```

---

## 11. Workload streams

Generate deterministic replay streams from MASSIVE `en-US`.

### Stream types

```text
uniform: sample utterances uniformly
zipf-mild: sample intents/templates with Zipf exponent around 0.8
zipf-heavy: sample intents/templates with Zipf exponent around 1.2
```

The Zipf streams are synthetic distributions over real utterances. This is acceptable because the demo needs controllable hot-path locality. Do not synthesize labels.

### Stream construction

Recommended:

```text
1. Group by intent and coarse normalized template.
2. Assign group weights from Zipf distribution.
3. Sample real utterances from selected groups.
4. Store stream indices in runs/<id>/stream.json for reproducibility.
```

Normalize template by replacing slot spans from `annot_utt` with slot placeholders:

```text
"set alarm for [time]"
"remind me to [todo] at [time]"
```

---

## 12. Experiments

Implement these as CLI subcommands or as named experiment configs under `configs/experiments/*.yaml`.

## 12.1 Experiment A: main evolution curve

Compare:

```text
baseline_all_l4
full_cascade_l0_l1_l2_l3_l4
```

Run on:

```text
zipf-heavy, max_requests=3000, compile_every=500
```

Plot over request index / compiler generation:

```text
L4 calls per 100 requests
cloud cost per 100 requests
p50/p95 latency
end-to-end intent accuracy
end-to-end frame exact match
layer coverage share
L1/L2/L3 guarded accepted accuracy
```

Expected qualitative result:

```text
L4 calls decrease after each compiler generation.
L0/L1/L2 coverage grows.
Full cascade accuracy remains close to all-L4.
```

If this does not happen, report actual results and include failure analysis. Do not fake curves.

---

## 12.2 Experiment B: direct L4 optimization ablation

Compare:

```text
teacher_label_only:
  L4 labels traces
  L1 disabled or static hand-empty
  L2 fixed default config
  L3 fixed default prompt
  guards use static thresholds

l4_compiler:
  L4 proposes L1 programs
  L4 proposes L2 configs
  L4 proposes L3 prompts
  L4 proposes guard candidates
  replay selects artifacts
```

Metrics:

```text
cost reduction speed
coverage by layer
guarded accuracy
wrong accept rate
end-to-end accuracy
```

Purpose:

```text
Show that L4 as compiler/proposer is materially different from L4 as labeler only.
```

---

## 12.3 Experiment C: L2 family search

Compare L2 candidates inside the same cascade:

```text
sgd_logreg + guard
mlp + guard
small_decision_tree + guard
mlp without guard
```

Metrics:

```text
forced accuracy
guarded accuracy
coverage
wrong accept rate
latency
model size
```

Purpose:

```text
Give decision tree a fair place in L2, not L1.
Check whether MLP/logreg generalizes better on this workload.
Show that guard is as important as model family.
```

---

## 12.4 Experiment D: no-guard ablation

Run:

```text
full artifacts, but force L1/L2/L3 to always accept when they produce any frame
```

Expected result:

```text
lower L4 cost
worse accuracy
wrong accepts spike
```

Purpose:

```text
Prove guard/gating is technically necessary, not product decoration.
```

---

## 12.5 Experiment E: no-L2 ablation

Compare:

```text
full_cascade: L0 -> L1 -> L2 -> L3 -> L4
no_l2:        L0 -> L1 -> L3 -> L4
```

Expected result:

```text
no_l2 has more L3/L4 calls and worse cost/latency on medium-difficulty paraphrases.
```

Purpose:

```text
Show L2 is not ignored; it absorbs fuzzy local statistical generalization between hard-coded programs and local SLM.
```

---

## 12.6 Experiment F: workload locality

Run full cascade on:

```text
uniform
zipf-mild
zipf-heavy
```

Expected result:

```text
More hot-path locality -> more L0/L1/L2 coverage -> fewer L4 calls.
Uniform workload should show smaller gains.
```

Purpose:

```text
Demonstrate the system depends on workload locality, not magic compression.
```

---

## 12.7 Experiment G: hard buffer / disagreement replay

Compare:

```text
with_hard_buffer
without_hard_buffer
```

Hard buffer contains:

```text
L1/L2/L3 accepted but disagreed with L4 during audit
candidate artifacts wrong on replay
parse failures
near misses around accepted regions
```

Purpose:

```text
Show teacher disagreement and replay prevent repeated local over-specialization errors.
```

---

## 13. Reports

Generate:

```text
runs/<id>/reports/summary.md
runs/<id>/reports/curves.html
runs/<id>/reports/metrics.csv
runs/<id>/reports/artifacts.csv
runs/<id>/reports/hard_cases.jsonl
```

Required report tables:

### 13.1 Layer summary

```text
layer | coverage | accepted_accuracy | wrong_accept_rate | forced_global_accuracy | p50_ms | p95_ms | cost/100
```

### 13.2 Evolution summary

```text
generation | L4_calls/100 | cost/100 | p95_ms | frame_em | L0_share | L1_share | L2_share | L3_share | L4_share
```

### 13.3 Artifact summary

```text
artifact_id | type | generation | coverage_delta | accuracy_delta | cost_delta | promoted | reason
```

### 13.4 L1 readable programs

Include the latest generated `generated_l1.py` snippets or DSL in the report.

This is important for the thesis: the user should see hot-paths becoming explicit CPU programs.

---

## 14. CLI design

Use `Typer`.

### 14.1 Prepare dataset

```bash
uv run edge-mvp prepare --locale en-US --out data/processed/massive_en_us
```

Responsibilities:

```text
- download MASSIVE en-US
- parse gold frames from intent + annot_utt
- build normalized templates
- save processed parquet/jsonl
```

### 14.2 Run replay

```bash
uv run edge-mvp run \
  --stream zipf-heavy \
  --max-requests 3000 \
  --compile-every 500 \
  --teacher live-or-cache \
  --run-dir runs/dev
```

Teacher modes:

```text
live: always call L4 when needed
cache: use existing teacher cache, fail if missing
live-or-cache: use cache if present; otherwise call L4
```

No mode named `fake` or `mock` for the main demo.

### 14.3 Run experiments

```bash
uv run edge-mvp experiment main-evolution --run-dir runs/main
uv run edge-mvp experiment l2-family --run-dir runs/l2-family
uv run edge-mvp experiment no-guard --run-dir runs/no-guard
uv run edge-mvp experiment workload-locality --run-dir runs/workload-locality
```

### 14.4 Generate report

```bash
uv run edge-mvp report --run-dir runs/main
```

---

## 15. Implementation milestones

### Milestone 1: data + schemas + baseline

Deliver:

```text
- uv project
- MASSIVE loader
- frame parser from annot_utt
- all-L4 teacher baseline
- trace writer
- summary metrics
```

Acceptance:

```text
uv run edge-mvp prepare works
uv run edge-mvp run --teacher live-or-cache --max-requests 50 writes traces
No gold leakage into compiler-visible data structures
```

### Milestone 2: runtime layers without compiler

Deliver:

```text
- L0 exact cache
- L1 empty ProgramBank with DSL validator/renderer
- L2 default student training from teacher traces
- L3 real local SLM prompt
- fixed cascade router
```

Acceptance:

```text
Full cascade runs end-to-end.
L3 calls real local model.
L2 trains real sklearn model.
L1 imports generated_l1.py even if empty.
```

### Milestone 3: compiler loop

Deliver:

```text
- hot cluster mining
- L4 L1 program proposal
- L4 L2 config proposal
- L4 L3 prompt proposal
- guard threshold grid search
- replay promotion
```

Acceptance:

```text
At least one compiler generation produces candidate artifacts.
Candidates are replayed before promotion.
Promoted artifacts affect subsequent routing.
```

### Milestone 4: experiments + reports

Deliver:

```text
- main evolution experiment
- direct L4 optimization ablation
- L2 family search
- no-guard ablation
- no-L2 ablation
- workload locality experiment
- hard buffer experiment
- HTML/Markdown reports
```

Acceptance:

```text
All experiments produce metrics.csv and curves.html.
Report includes layer summary, evolution curves, artifact summary, and L1 generated program snippets.
```

---

## 16. Guardrails for Codex agent

Do not deviate from these without explicit user approval:

```text
1. Do not implement L1 as sklearn DecisionTreeClassifier.
2. Do not train L1 parameters. L1 is train-free ProgramBank.
3. Do not use MASSIVE gold labels for training/compiler/router decisions.
4. Do not replace L3 local SLM with a stub.
5. Do not replace L4 calls with fake labels in the main demo.
6. Do not let L4 self-certify an artifact.
7. Do not use arbitrary generated Python from L4 directly; use DSL -> validated codegen.
8. Do not add a complex learned global router in MVP.
9. Do not optimize for a pretty dashboard at the expense of real evolution.
10. Do not hide failed experiments; report actual metrics.
```

Allowed simplifications:

```text
- Slot extraction can be imperfect in v1; report intent accuracy, slot F1, and frame EM separately.
- Teacher cache can be reused after real L4 calls are recorded.
- Candidate counts can be small.
- Local SLM can be small and weak, as long as it is real.
- L4 pricing can be user-configured rather than hardcoded.
```

---

## 17. Expected qualitative demo story

The expected story after running `main-evolution` on a Zipf-heavy stream:

```text
Generation 0:
  L4 handles nearly everything.
  L0 has only exact repeats.
  L1 is mostly empty.
  L2 is weak.
  L3 has a generic prompt.

Generation 1:
  L0 absorbs exact repeats and near-duplicates.
  L1 gets a few high-precision alarm/weather/reminder rules.
  L2 begins accepting high-confidence common intents.
  L3 prompt becomes more schema-stable.

Generation 2+:
  L1 ProgramBank covers more narrow templates.
  L2 absorbs paraphrases that are too fuzzy for L1.
  L3 catches long-tail commands before L4.
  L4 calls and cost drop.
  End-to-end quality remains close to all-L4.
```

The final report should make this visually obvious.

---

## 18. Minimum definition of done

The MVP is complete when:

```text
- It runs with uv.
- It loads MASSIVE en-US.
- It calls L4 through OPENAI_API_KEY + optional OPENAI_BASE_URL.
- It runs a real local SLM for L3.
- It trains a real L2 student and learned guard.
- It generates and executes real L1 CPU programs from L4-proposed DSL.
- It records traces and teacher disagreements.
- It promotes artifacts only through replay.
- It runs the required experiments.
- It outputs cost/latency/accuracy/coverage curves.
- It includes an explicit no-mock/fail-fast behavior for missing L4 labels.
```

Target result, not a hard guarantee:

```text
On Zipf-heavy stream after several compiler iterations:
  L4 calls per 100 requests decrease materially.
  Cost per 100 requests decreases materially.
  p95 latency decreases materially.
  End-to-end intent accuracy remains within ~2 percentage points of all-L4.
  L1 guarded accuracy is high while forced global accuracy is much worse.
```

If target result is not achieved, the report must show actual metrics and identify whether the bottleneck was:

```text
- insufficient workload locality
- weak L1 rule coverage
- weak L2 guard calibration
- local SLM JSON instability
- teacher inconsistency
- overly strict promotion gate
```

---

## 19. Double-check against user requirements

```text
Requirement: includes all five layers.
Status: yes. L0/L1/L2/L3/L4 are explicitly defined and wired into cascade.

Requirement: real “sacrifice generality for local optimum” process, not mock.
Status: yes. L1 generated CPU programs, L2 trained student, L3 real local SLM, L4 real API, replay promotion required.

Requirement: minimal concept set.
Status: yes. Concepts are trace, teacher, artifact, guard/gating, disagreement, replay. Product/privacy/security abstractions are excluded.

Requirement: L4 generates/optimizes L1/L2/L3 and guard/gating.
Status: yes. L4 proposes L1 DSL, L2 configs, L3 prompts, guard candidates; replay selects.

Requirement: L1 train-free evolved CPU programs, not trained tree.
Status: yes. Decision tree is explicitly banned from L1 and allowed only as L2 candidate.

Requirement: L2 not ignored.
Status: yes. L2 has student + slot model + learned guard + family-search experiment + no-L2 ablation.

Requirement: dependency selection avoids wheel reinvention and uses modern stack.
Status: yes. uv, OpenAI SDK, Hugging Face datasets/transformers, sklearn, sentence-transformers, FAISS, Pydantic, Typer, Rich.

Requirement: use uv.
Status: yes.

Requirement: LLM uses OPENAI_API_KEY + OPENAI_BASE_URL.
Status: yes.

Requirement: include experiments with quantitative curves.
Status: yes. Main evolution, direct-L4 ablation, L2 family, no-guard, no-L2, locality, hard-buffer experiments are specified.

Requirement: proposal is suitable for Codex agent.
Status: yes. It includes repo structure, milestones, CLI, schemas, guardrails, and definition of done.
```
