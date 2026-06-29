# Darjeeling Public Relations Runbook

This runbook prepares Darjeeling for external attention. The goal is not to
maximize traffic immediately. The goal is to make a first-time visitor
understand the project quickly, run a small demo, and know the current maturity
level.

## Positioning

Use this as the public-facing one-line narrative:

> Darjeeling is a runtime for LLM applications that moves stable, repeated work
> into fast local artifacts, such as deterministic code, compact neural
> networks, and small local models, while falling back to an LLM when the local
> path is not reliable.

The first-screen narrative should avoid internal layer names such as L1, L2,
L3, or L4. Those names are useful implementation language, but they are not how
new users should first understand the project.

Recommended short version:

> Darjeeling helps LLM apps get faster and cheaper by safely moving repeatable
> work out of the main model and into local code or small models.

Recommended longer version:

> Darjeeling is a runtime for LLM applications with repeated structured work. It
> identifies cases that are stable enough to handle locally, turns them into
> deterministic code, compact neural networks, or small local models, and uses
> those artifacts only when they are reliable. Hard or unfamiliar requests still
> go to the main LLM, so the system can reduce latency and inference cost
> without forcing local artifacts to guess.

## What To Avoid

Do not lead with:

- internal layer labels such as L1/L2/L3/L4;
- "teacher model" language in the first paragraph;
- "four-layer architecture" as the primary value proposition;
- "cache" as the main comparison;
- production-ready claims.

It is fine to explain the internal layered model later, after the reader already
understands the product-level idea.

## Audience

The first promotion wave should target people who build or evaluate LLM-backed
products:

- AI infrastructure engineers;
- LLM application engineers;
- evals and reliability engineers;
- founders or researchers working on cost, latency, routing, or local inference.

The pitch should assume they know what an LLM is, but not Darjeeling's internal
terminology.

## Conversion Targets

A good public entry point should let a new visitor do three things:

1. Understand the project in 30 seconds.
2. Run a demo in 5 minutes.
3. See why the runtime can improve latency and inference cost without simply
   lowering quality.

Do not start wide promotion until the repository satisfies those three checks.

## Pre-Launch Work

### 1. Rewrite The README First Screen

The README should open with the product-level narrative, then explain why it
exists.

Suggested structure:

```md
# Darjeeling

Darjeeling is a runtime for LLM applications that moves stable, repeated work
into fast local artifacts, such as deterministic code, compact neural networks,
and small local models, while falling back to an LLM when the local path is not
reliable.

Local artifacts answer only when they are inside a checked reliability boundary.
Otherwise, Darjeeling falls back to the main LLM.

## Why This Exists

Many LLM products have repeated structured work. Some requests become stable
enough to handle locally; others still need a strong model. Darjeeling provides
the runtime and evaluation loop for moving stable behavior into local artifacts
while keeping fallback, validation, tracing, and recompile paths explicit.
```

The README can introduce internal layers later under "How it works".

### 2. Add A Five-Minute Demo

Create a small demo that shows the full loop with minimal setup:

```bash
uv sync --extra dev
uv run darjeeling demo thin-target
```

The demo should show:

- cold start through the reference LLM path;
- a local artifact accepting known-safe requests;
- fallback for unfamiliar requests;
- a report with precision, coverage, latency, fallback share, and estimated
  saving.

The demo may use toy data. It must say so clearly. Do not imply production
benchmark results.

### 3. Make Packaging Names Consistent

The CLI command should remain:

```bash
darjeeling
```

The Python import package may remain:

```python
import darjeeling
```

The PyPI distribution name should not be `darjeeling`, because that name is
already taken by an unrelated package. Prefer one of:

- `darjeeling-ai`
- `darjeeling-runtime`
- `darjeeling-core`

After changing the distribution name, update README install instructions and
lock files.

### 4. Add GitHub Repository Metadata

Set the repository description to:

> LLM runtime for moving stable model behavior into fast local artifacts with
> safe fallback.

Suggested topics:

```text
llm
ai-runtime
model-routing
evals
fallback
local-inference
cost-optimization
latency
python
agentic-ai
```

Add a social preview image with a simple request-to-local-artifact-to-fallback
diagram. Keep the text short:

> Local when safe. Fallback when needed.

### 5. Add Minimum Project Trust Files

Before broader promotion, add only the files that reduce friction:

- `CHANGELOG.md`
- `CONTRIBUTING.md`
- `SECURITY.md`
- `.github/workflows/ci.yml`
- `.github/ISSUE_TEMPLATE/demo_feedback.yml`
- `.github/ISSUE_TEMPLATE/use_case.yml`

Do not add a large documentation tree before the demo and README are clear.

### 6. Cut An Alpha Release

Use an explicit alpha tag such as:

```text
v0.1.0-alpha.1
```

The release notes should say:

- this is an alpha runtime;
- current stores and queues are filesystem/in-memory;
- the project is ready for design validation and early demo feedback;
- production hardening is future work.

## Public Explanation Pattern

Use this order in posts, articles, and README sections:

1. Repeated LLM work creates latency and inference-cost pressure.
2. Some behavior becomes stable enough to run locally.
3. Darjeeling turns that behavior into local artifacts.
4. Local artifacts can refuse, so unfamiliar requests fall back.
5. The system measures precision, coverage, latency, fallback share, and cost.
6. Current status is alpha.

This avoids presenting Darjeeling as "just a cache", "just distillation", or "a
generic model router".

## Demo Success Criteria

The demo is launch-ready when:

- a clean checkout can run it from documented commands;
- output includes at least one local accept and one fallback;
- the report shows precision, coverage, latency, fallback share, and estimated
  saving;
- failure modes are understandable;
- no environment variable is required unless the demo explicitly chooses a live
  reference model path;
- the default demo does not spend paid API credits.

## Private Feedback Wave

Before public launch, ask 10 to 20 technical readers to run the demo.

Ask only these questions:

1. What do you think Darjeeling does?
2. Did the demo run in five minutes?
3. Does the README make it clear why this is not just cache or distillation?
4. What use case would you try with it?

Do not optimize for stars in this wave. Optimize for comprehension and demo
completion.

## Public Launch Wave

After the README and demo are stable:

1. Publish the alpha release.
2. Publish a short article explaining the runtime idea.
3. Share the repository in focused communities.

Potential launch headline:

> Darjeeling: a runtime for moving repeated LLM work into local artifacts with
> safe fallback

Potential short post:

> I released Darjeeling, an alpha runtime for LLM applications that moves
> stable, repeated work into deterministic code, compact neural networks, and
> small local models.
>
> Local artifacts answer only inside a checked reliability boundary. Everything
> else falls back to the main LLM. The goal is lower latency and lower inference
> cost without forcing local artifacts to guess.
>
> I am looking for early feedback on the demo, target definition format, and
> use cases.

## Launch Metrics

Track these weekly:

- repository visitors;
- stars and forks;
- demo feedback issues;
- use case issues;
- quickstart failures;
- external descriptions of the project;
- contributors or users who bring a real target.

The most important early signal is whether readers can describe Darjeeling
without using internal layer terms.

## Non-Goals For The First Launch

Do not claim:

- production readiness;
- benchmark leadership;
- replacement for hosted LLMs;
- general autonomous agent success;
- zero-cost inference.

Do not make broad claims about arbitrary open-ended chat workloads. The clearest
initial fit is repeated structured LLM work where local artifacts can safely
accept some requests and fall back on the rest.

## Done Criteria

The public launch preparation is complete when:

- README first screen explains moving stable repeated LLM work into local
  artifacts with fallback;
- packaging and command names are consistent;
- a five-minute demo exists and passes from a clean checkout;
- GitHub metadata is filled in;
- minimal trust files and CI exist;
- an alpha release is ready;
- 10 to 20 private readers have tried the demo or reviewed the README;
- feedback no longer shows basic confusion about whether Darjeeling is a cache,
  a normal router, or one-off distillation.
