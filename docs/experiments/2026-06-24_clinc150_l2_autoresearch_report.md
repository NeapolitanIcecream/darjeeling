# CLINC150 L2 AutoResearch Report

Date: 2026-06-24

Decision: **Pause and repair AutoResearch harness**.

The CLINC150 L2 repair was moved from manual guard search to the target-local
L4 AutoResearch path because the previous calibration repair showed a
target-specific failure mode: high-confidence OOS and intent-boundary accepts
were not reliably handled by global threshold, margin, entropy, and hand-picked
intent vetoes. This run reused the existing L2 target-evolution workspace and
added only a CLINC150 bridge plus target-local OOS-risk metadata.

The bridge and accounting path work, and the L4 agent produced a target-local
candidate without reading locked-test evidence. The candidate cleared visible
inner validation and visible train-audit safety, but it did not pass the private
selection holdout and failed the official-validation precision gate required
before locked-test exposure. No new AutoResearch candidate was selected for
locked test.

Primary measurements kept L0, L1, and L3 disabled:

```text
L2 shadow
L2 + L4 replay-oracle fallback
```

## Branch And Worktree

Dedicated branch:

```text
codex/clinc150-l2-autoresearch
```

Dedicated worktree:

```text
/Users/chenmohan/gits/darjeeling-clinc150-l2-autoresearch
```

Experiment root:

```text
runs/clinc150-l2-autoresearch-20260624/
```

Main AutoResearch run:

```text
runs/clinc150-l2-autoresearch-20260624/agent-session-r1/
```

## Reused Artifacts And Cost

The run copied ignored artifacts into the dedicated worktree from the main
Darjeeling checkout and the calibration-repair worktree. Row counts were
validated before use.

| Artifact | Rows | SHA256 |
| --- | ---: | --- |
| `runs/clinc150-l2-cascade-20260623/teacher-traces/train-full-stratified/teacher_live_vs_gold.details.jsonl` | 15,100 | `242ec13214096bb106b389a54fed699a8ec646a25d37afbb250a54d9915be839` |
| `runs/clinc150-l2-cascade-20260623/teacher-traces/validation-full/teacher_live_vs_gold.details.jsonl` | 3,100 | `17cbbdc40ccb80faaeb7b12ead65266018c71f2f9d5c46557d02b99efe35ecdf` |
| `runs/clinc150-l2-cascade-20260623/teacher-traces/test-full/teacher_live_vs_gold.details.jsonl` | 5,500 | `da19fc889f8337218047128af9a1e1ca023066679273ab10de0d57a2b8aef10d` |
| `data/processed/clinc150_data_full/train.jsonl` | 15,100 | checked by row count |
| `data/processed/clinc150_data_full/validation.jsonl` | 3,100 | checked by row count |
| `data/processed/clinc150_data_full/test.jsonl` | 5,500 | checked by row count |
| `runs/clinc150-l2-cascade-20260623/distilled-l2/train-full/l2_student.joblib` | n/a | `2c7fa47be3a55f6a656733c1230a74b6ac606d931d5c47c25b4b7d7a877fbfa4` |
| `runs/clinc150-calibration-repair-20260624/safety-margin-995/clinc150_calibration_repair_summary.json` | n/a | `27c39647bb4beec19bd0b51581cef1afb81a58c4b865b230013939eb8caa876e` |

No new paid benchmark L4 calls were made. The AutoResearch summary reports:

```json
{
  "new_paid_l4_calls": 0,
  "new_paid_spend_usd": 0.0
}
```

No new cost ledger was written. Benchmark spend still points back to the prior
CLINC150 ledger:

```text
docs/experiments/2026-06-23_clinc150_l2_cascade_cost_ledger.json
```

Coding-agent usage is separate from benchmark serving spend. The
`agent_session.jsonl` transcript exposed token usage for the L4 agent session:

```json
{
  "input_tokens": 8344838,
  "cached_input_tokens": 7868928,
  "output_tokens": 18740,
  "reasoning_output_tokens": 4116
}
```

No USD cost for that coding-agent session was present in the experiment
artifacts.

## Harness Changes

The implementation reused existing L2 target-evolution machinery and added a
small CLINC150 bridge:

- `L2Prediction` now exposes class probabilities as target metadata.
- CLINC150 L2 prediction rows now include `intent_probabilities`,
  `oos_probability`, `oos_rank`, and `oos_margin`.
- CLINC150 calibration guard generation can evaluate OOS probability, OOS rank,
  and OOS margin rules.
- `run_clinc150_l2_autoresearch` builds teacher-visible train traces from
  existing teacher detail rows, with gold labels withheld from the L4 agent
  workspace metadata.
- The target-evolution workspace can receive `target/config.json` initial
  overrides and a visible-only `data/target_context.json`.
- A new CLI command runs the bridge:

```bash
uv run --extra dev python -m darjeeling.targets.nlu.main_cli clinc150 l2-autoresearch
```

The generated target artifact stayed inside the ignored AutoResearch run
workspace:

```text
runs/clinc150-l2-autoresearch-20260624/agent-session-r1/target-evolution/rounds/round_001_target/
```

No CLINC150 labels, intents, request ids, or accepted-error examples were moved
into Darjeeling core.

## Data Visibility

Visible to the L4 agent:

- parsed train teacher rows converted into target-evolution traces;
- target-local visible train and inner-validation JSONL files;
- visible cross-audit folds;
- visible diagnostics, safety backlogs, intent-confusion backlogs, and target
  context;
- initial CLINC150 L2 config overrides;
- target-local OOS-risk signal descriptions.

Withheld from the L4 agent:

- official locked-test labels;
- locked-test accepted-error rows, utterances, labels, and confusion families;
- private selection and promotion holdout rows;
- official validation and locked-test replay results until the outer harness
  evaluated the post-session candidate.

The AutoResearch summary confirms primary layers were disabled:

```json
{
  "l0_enabled": false,
  "l1_enabled": false,
  "l3_enabled": false
}
```

## Baseline Reproduction

Fixed teacher-distilled L2 was replayed with L0 disabled and L4 fallback from
existing teacher details.

| Split | Guard | Accepted precision | Coverage | OOS false accept | Cascade delta |
| --- | --- | ---: | ---: | ---: | ---: |
| validation | `threshold >= 0.98` | 99.10% | 50.32% | 1.00% | -0.097pp |
| locked test | `threshold >= 0.98` | 98.77% | 42.73% | 1.50% | 0.000pp |
| validation | `threshold >= 0.995` | 99.78% | 28.87% | 0.00% | 0.000pp |
| locked test | `threshold >= 0.995` | 99.78% | 24.56% | 0.30% | +0.109pp |

The previous calibration-repair selected guard was also replayed exactly:

```text
guard_probability >= 0.985
veto predicted intents: credit_score, directions, spending_history
```

| Split | Accepted precision | Coverage | OOS false accept | Cascade delta |
| --- | ---: | ---: | ---: | ---: |
| validation | 99.506% | 45.74% | 1.00% | +0.032pp |
| locked test | 98.997% | 38.07% | 1.20% | 0.000pp |

This reproduced the prior diagnosis: manual guard repair looked strong on
validation but missed the locked-test precision target, and the remaining
accepted wrong rows were dominated by OOS or intent-boundary risk. The previous
OOS-heavy diagnostic for the selected guard was also reused from the copied
calibration summary: 33.33% accepted precision, 1.64% coverage, and 2.00% OOS
false accept rate.

## AutoResearch Run

Command:

```bash
uv run --extra dev python -m darjeeling.targets.nlu.main_cli clinc150 l2-autoresearch \
  --out-dir runs/clinc150-l2-autoresearch-20260624/agent-session-r1 \
  --mode agent-session \
  --rounds 16 \
  --budget-profile fixed-inner \
  --timeout-s 2400 \
  --local-search-trials 32
```

The outer harness launched one long L4 agent session. The session report says
the agent:

- kept the conservative initial `target/config.json`;
- added target-local `target_l2.py` postprocess and accept-veto rules for
  visible high-risk intent confusions;
- tried `tools/search_config.py`, observed unsafe interim config mutations, and
  restored the safe config;
- stopped with a clean target-local candidate.

### Hypothesis 1: Visible Intent-Boundary Vetoes

Implementation:

- add direct target-local lexical vetoes in `target/target_l2.py`;
- keep global threshold and model family unchanged;
- use visible safety, slot-risk, and intent-confusion backlogs only.

Visible result:

| View | Accepted | Correct | Wrong | Accepted accuracy | Coverage |
| --- | ---: | ---: | ---: | ---: | ---: |
| inner validation baseline | 2,067 | 2,055 | 12 | 99.419% | 45.52% |
| inner validation candidate | 2,043 | 2,043 | 0 | 100.000% | 44.99% |
| visible cross-audit baseline | 6,874 | 6,844 | 30 | 99.564% | 56.95% |
| visible cross-audit candidate | 6,822 | 6,812 | 10 | 99.853% | 56.52% |
| train audit baseline | 6,401 | 6,401 | 0 | 100.000% | 85.02% |
| train audit candidate | 6,366 | 6,366 | 0 | 100.000% | 84.55% |

Gate result:

| Gate | Result |
| --- | --- |
| visible support gate | pass |
| visible train-audit safety gate | pass |
| private promotion gate | pass |
| private selection gate | fail |
| candidate selection gate | fail |

The private selection holdout had no improvement over baseline:

| View | Accepted | Correct | Wrong | Accepted accuracy | Coverage |
| --- | ---: | ---: | ---: | ---: | ---: |
| selection baseline | 709 | 705 | 4 | 99.436% | 46.64% |
| selection candidate | 705 | 701 | 4 | 99.433% | 46.38% |

Interpretation: the target-local visible veto rules removed visible inner
validation errors, but they did not generalize enough to reduce private
selection wrong accepts. The outer harness correctly refused to select the
candidate.

### Hypothesis 2: Config Search

Implementation:

- the agent ran visible-data config search through `tools/search_config.py`;
- unsafe interim configs appeared during the search;
- the agent restored the conservative config and did not adopt a searched
  config.

Result: no searched config became the final candidate. This is a harness
friction point: the search tool should write scratch candidates or restore state
automatically so the agent can use it without risking a half-mutated
`target/config.json`.

## Candidate And Outer Replay

No candidate was selected for locked test. The outer summary marked the final
target as diagnostic only:

```json
{
  "role": "best_round_diagnostic_only",
  "round": 1,
  "selected_for_locked_test": false,
  "locked_test_policy": "official test evaluated only when best_adoptable candidate passes official validation and validation streams"
}
```

Even as a diagnostic candidate, the outer CLINC150 replay evaluated official
validation and validation streams. These results failed the pre-lock gate of
accepted precision >= 99.5%.

| Stream | Accepted precision | Wilson lower 95 | Coverage | OOS false accept | Cascade delta | L4 call reduction |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| validation sequential | 99.100% | 98.494% | 50.16% | 1.00% | -0.097pp | 50.16% |
| validation uniform | 99.349% | 98.806% | 49.58% | 1.92% | +0.161pp | 49.58% |
| validation zipf-heavy | 98.781% | 97.999% | 39.71% | 0.88% | -0.129pp | 39.71% |

Locked-test exposures for the AutoResearch candidate:

```json
{
  "locked_test_exposures": 0,
  "locked_test": null
}
```

## OOS And Intent-Boundary Analysis

The new OOS-risk metadata path is useful but was not enough by itself in this
first run. It exposes `out_of_scope` probability, OOS rank, OOS margin, and full
intent probabilities to target-local guards without teaching Darjeeling core
anything about CLINC150 labels. The L4 agent, however, mostly pursued
visible intent-boundary vetoes and did not produce a broader OOS-risk detector
or probability-margin policy.

The final diagnostic candidate preserved the same official validation OOS false
accept count as the fixed threshold baseline: 1 OOS false accept on validation
sequential. The zipf-heavy stream still showed weaker accepted precision
98.781%, which is the clearest visible sign that the repair remains fragile
under distribution shift.

The important negative result is that visible inner-validation perfection did
not imply private selection improvement. Future AutoResearch context needs more
visible OOS-heavy and cross-fold pressure before the agent stops, not just a
zero-wrong visible inner fold.

## Decision

Do not proceed with an L2 AutoResearch candidate. No candidate was eligible for
locked test:

- private selection gate failed;
- official validation accepted precision was 99.100%, below the 99.5%
  pre-lock gate;
- validation zipf-heavy accepted precision fell to 98.781%;
- no new candidate reached the locked-test stage.

Do not reject the current CLINC150 L2 shape yet. The bridge now supports
target-local AutoResearch with replay-oracle accounting, OOS-risk metadata, and
locked-test withholding. The remaining problem is that the first agent session
overfit visible accepted-wrong cleanup and lacked enough visible pressure to
generalize OOS and intent-boundary safety.

Decision: **Pause and repair AutoResearch harness**.

## Risks And Next Step

Risks:

- visible inner-validation success can be misleading when private selection
  errors are from adjacent but unseen boundary families;
- config search currently mutates `target/config.json` in place during a
  session;
- the target context describes OOS-risk signals, but the workspace diagnostics
  do not yet force the agent to test OOS-heavy folds before stopping;
- official validation is currently evaluated after the agent session, so it is
  useful for selection but not for intra-session correction.

Next repair before any new locked-test candidate:

- add visible train-derived OOS-heavy folds to the AutoResearch workspace;
- add structured OOS probability, OOS rank, and OOS margin diagnostics to
  `target_diagnostics.json` and the visible backlogs;
- make `tools/search_config.py` write scratch candidate configs instead of
  mutating the active config in place;
- run another long agent-session or bounded multi-round AutoResearch attempt
  that must clear visible OOS-heavy/cross-audit pressure before stopping;
- keep official locked test withheld until a candidate passes private selection,
  official validation, and validation stream gates.

## Validation

Validation run so far:

```bash
uv run --extra dev pytest tests/targets/nlu/test_l2_target_evolution.py tests/targets/nlu/test_clinc150_phase1.py tests/targets/nlu/test_l4_teacher.py -q
# 81 passed

uv run --extra dev ruff check \
  src/darjeeling/targets/nlu/layers/l2_student.py \
  src/darjeeling/targets/nlu/clinc150_phase1.py \
  src/darjeeling/targets/nlu/compiler/l2_target_evolution.py \
  src/darjeeling/targets/nlu/main_cli.py \
  tests/targets/nlu/test_clinc150_phase1.py \
  tests/targets/nlu/test_l2_target_evolution.py
# passed

git diff --check
# passed

uv run --extra dev pytest -q
# failed because optional CLINC150 adapter dependency pandas was absent:
# 2 failed, 303 passed

uv run --extra dev --extra massive pytest -q
# 305 passed
```

The full suite requires the same optional adapter extras used by prior CLINC150
work.
