# Alpha Release Checklist

Use this checklist for `v0.1.0-alpha.1`.

## Repository Checks

- [ ] README first screen explains local-when-safe with fallback-when-needed.
- [ ] `uv run darjeeling demo thin-target` runs without API keys.
- [ ] Demo output includes one local accept and one fallback.
- [ ] Demo report includes precision, coverage, latency, fallback share, and
      estimated saving.
- [ ] `pyproject.toml` distribution name is `darjeeling-ai`.
- [ ] Python import remains `import darjeeling`.
- [ ] CLI command remains `darjeeling`.
- [ ] `CHANGELOG.md`, `CONTRIBUTING.md`, and `SECURITY.md` exist.
- [ ] CI workflow passes on the release commit.

## Private Feedback Wave

Ask 10 to 20 technical readers to run the demo before broad promotion.

Ask only:

1. What do you think Darjeeling does?
2. Did the demo run in five minutes?
3. Does the README make it clear why this is not just cache or distillation?
4. What use case would you try with it?

Do not optimize for stars during this wave. Optimize for comprehension and demo
completion.

## Tag And Release

- [ ] Confirm `docs/releases/v0.1.0-alpha.1.md` matches the release commit.
- [ ] Tag the release:

  ```bash
  git tag v0.1.0-alpha.1
  git push origin v0.1.0-alpha.1
  ```

- [ ] Publish GitHub release notes from `docs/releases/v0.1.0-alpha.1.md`.
- [ ] Publish the PyPI package as `darjeeling-ai` with Python version
      `0.1.0a1`.
- [ ] Confirm the published package keeps the `darjeeling` CLI command.

## Launch Gate

Start public promotion only after feedback no longer shows basic confusion
about whether Darjeeling is a cache, a normal router, or one-off distillation.
