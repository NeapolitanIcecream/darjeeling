# Autonomous Research Methodology Note

Date: 2026-06-25

This note preserves the methodology discussion before the 2026-06-25 daytime
autonomous research sprint completes. It is not yet an AGENTS.md rule and not
yet a Codex skill. It is input for reviewing the daytime sprint and, if the
method works, for later creating a reusable skill family.

## Motivation

The 2026-06-24 overnight autonomous research plan had useful local results, but
it failed the intended long-running research behavior:

- it stopped after about 1.5 hours instead of using the planned time window;
- its cycle count became a checklist rather than a time-bound execution
  discipline;
- it mostly tightened gates and repaired harness issues, but did not materially
  improve L1/L2 effect;
- it reported paid benchmark API spend as `$0`, but did not treat L4
  agent-session usage as part of the experiment L4 budget;
- it did not have a strong enough mechanism for abandoning a low-yield direction
  and moving to a better one.

The problem is therefore not only prompt wording. We need a reusable research
execution method that makes long-horizon agents less likely to prematurely
converge, rationalize a weak direction, or confuse support work with scorecard
progress.

## STAR-PolyaMath Signal

The STAR-PolyaMath work is relevant because it uses explicit long-horizon
reasoning structure rather than relying on generic "be autonomous" wording:

- Repository: https://github.com/Julius-Woo/STAR-PolyaMath
- Paper: https://arxiv.org/abs/2605.19338

The useful ideas for our setting are:

- an explicit Pólya-style problem-solving loop;
- a persistent meta-strategic layer that supervises direction, not just local
  correctness;
- durable problem state across attempts;
- challenge / step / replan structure;
- the ability to issue strategic interventions when the current line of work is
  unproductive.

We do not need to copy its full multi-agent structure immediately. The lower
abstraction-tax first step is to encode the same strategic discipline inside a
single-agent research sprint method.

## Pólya Loop Mapping

Use Pólya's four steps as the default language for long-running research loops:

1. Understand the problem:
   - read the scorecard, constraints, current artifacts, prior failures, and
     cost/usage boundaries;
   - identify the current bottleneck and what evidence would change the
     decision.
2. Devise a plan:
   - propose one or more falsifiable hypotheses;
   - estimate expected scorecard impact, cost, risk, and validation path.
3. Carry out the plan:
   - implement, run experiments, invoke L4 agent-sessions, or run paid API
     validation as needed;
   - keep experiments bounded and checkpointed.
4. Look back:
   - compare against the scorecard;
   - update plots, ledgers, and the strategy state;
   - decide whether to continue, expand, reduce, abandon, or switch direction.

This maps to our previous "hypothesis -> implementation/experiment ->
validation -> iteration" wording, but Pólya is more recognizable to models and
less likely to be interpreted as a simple task checklist.

## Meta-Strategic Review

The main missing behavior is the ability to judge a direction strategically.
Every 60-90 minutes in a long-running sprint, the agent should perform a
meta-strategic review separate from ordinary experiment logging.

The review should answer:

- What is the active hypothesis?
- What scorecard metric can this direction still move?
- Is the evidence supporting the hypothesis, or only causing local repairs?
- Has support work started to substitute for effect improvement?
- If this direction gets another 60-90 minutes, what concrete new evidence is
  expected?
- Is another backlog item now higher expected value?
- Should this direction continue, narrow, pause, retire, or escalate to paid
  validation?

This review does not require a separate agent at first. It can be a required
section in `strategy_state.md`. If single-agent reviews remain weak, a later
skill or orchestrator can introduce a real Meta-Strategist role.

## Strategy State

A long-running research task should maintain a compact durable state file in
addition to a chronological log:

```text
docs/experiments/<date>_<sprint>_strategy_state.md
```

The file should be short and actively updated. Suggested sections:

- current scorecard;
- active bets;
- retired directions;
- strongest evidence so far;
- current bottleneck;
- open risks;
- cost/usage status;
- next strategic move and why.

The research log records what happened. The strategy state records what the
agent currently believes and why. This reduces memory fragmentation when the
task runs for hours.

## Direction Retirement

Long-running plans should explicitly permit abandoning a direction. A direction
should be retired or paused when one or more of these holds:

- repeated cycles repair harness/reporting without moving the scorecard;
- the direction can only reduce risk, not improve coverage/frontier/cost;
- visible metrics improve but transfer diagnostics repeatedly fail and no new
  evidence explains the gap;
- a timeout is repaired or reduced once, but the same bottleneck still prevents
  useful candidate generation;
- continuing would require violating the target/core boundary;
- continuing needs extra experiment L4 budget, product-goal changes, or
  risk-tolerance decisions outside the plan;
- a different backlog item now has clearly higher expected value.

Retiring a direction is not failure. It is evidence hygiene. The final report
should list retired directions and the evidence that retired them.

## Cost And Usage Ledger

Keep three concepts separate:

- `experiment_l4_spend_usd`: the budgeted L4 cost controlled by the experiment.
  It includes priced benchmark/API serving calls and Darjeeling-launched L4
  AutoResearch/evolve agent-sessions.
- L4 agent-session usage: detailed records for each L4 AutoResearch/evolve
  session launched by the experiment harness. Record session count, model if
  visible, elapsed time, timeout, rounds requested/completed, stop reason,
  token usage, artifact path, and observed or estimated dollar cost. If exact
  pricing is unavailable, record token usage and mark the cost estimate as
  pending; do not silently treat it as zero.
The important accounting risk is reporting `$0` experiment L4 spend while
Darjeeling-launched L4 agent-sessions did substantial research work.

## Candidate Skill Family

If the daytime sprint validates this method, create a reusable skill family.
Do not make it Darjeeling-specific.

Recommended first skill:

```text
autonomous-research-sprint
```

Purpose:

- write and execute long-running research plans;
- enforce time-bound execution rather than checklist completion;
- use Pólya loops;
- maintain strategy state, logs, reports, and ledgers;
- require periodic meta-strategic review;
- support direction retirement.

Suggested structure:

```text
autonomous-research-sprint/
├── SKILL.md
├── agents/openai.yaml
├── references/
│   ├── polya-loop.md
│   ├── meta-strategic-review.md
│   ├── time-boundary.md
│   ├── ledgers.md
│   └── templates.md
└── scripts/
    └── init_sprint_docs.py
```

Keep `SKILL.md` concise. Put detailed protocols and templates in references so
Codex loads them only when needed.

Possible later skills:

- `meta-strategic-review`: review a running or completed research sprint and
  decide which directions to continue, retire, or escalate.
- `research-sprint-audit`: verify an agent's report against artifacts, diffs,
  tests, ledgers, and claimed outcomes.

Start with one main skill plus references. Split later only if triggering or
context size becomes a problem.

## Relationship To AGENTS.md

Do not write this into AGENTS.md before the daytime sprint is reviewed.

After a successful or instructive daytime sprint, AGENTS.md can record the
project-local lesson, such as:

- long-running research plans use time as the execution boundary;
- support work cannot substitute for scorecard progress;
- strategy state and retired directions are required for long sprints;
- experiment L4 spend and L4 agent-session usage details must be reported
  together without treating agent-session usage as zero-cost by default.

AGENTS.md should remain project discipline. The reusable method should live in
the generic skill.

## Open Questions For Review After The Daytime Sprint

- Did the time-bound execution rule actually prevent early stopping?
- Did the agent maintain a useful strategy state, or only a chronological log?
- Did meta-strategic review cause a real direction switch or retirement?
- Did the sprint try to improve L1/L2 effect, not only safety gates?
- Were experiment L4 spend and L4 agent-session usage reported correctly?
- Which instructions were useful enough to promote into a skill?
- Which instructions were too Darjeeling-specific and should stay in repo docs?
