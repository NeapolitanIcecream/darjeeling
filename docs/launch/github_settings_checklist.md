# GitHub Settings Checklist

Some launch settings live in GitHub repository settings rather than tracked
files. Description, topics, issues, and private vulnerability reporting were
applied on 2026-06-29. Keep the commands below for repeatability.

Repository:

- `NeapolitanIcecream/darjeeling`
- URL: `https://github.com/NeapolitanIcecream/darjeeling`

## Description

Status: applied.

Set the repository description to:

```text
LLM runtime for moving stable model behavior into fast local artifacts with safe fallback.
```

Command:

```bash
gh repo edit NeapolitanIcecream/darjeeling \
  --description "LLM runtime for moving stable model behavior into fast local artifacts with safe fallback."
```

## Topics

Status: applied.

Set these topics:

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

Command:

```bash
gh repo edit NeapolitanIcecream/darjeeling \
  --add-topic llm \
  --add-topic ai-runtime \
  --add-topic model-routing \
  --add-topic evals \
  --add-topic fallback \
  --add-topic local-inference \
  --add-topic cost-optimization \
  --add-topic latency \
  --add-topic python \
  --add-topic agentic-ai
```

## Issues

Status: applied.

Issue forms are tracked in `.github/ISSUE_TEMPLATE/`. Confirm issues are
enabled:

```bash
gh repo view NeapolitanIcecream/darjeeling --json hasIssuesEnabled
```

## Social Preview

Status: manual.

Upload `docs/assets/social-preview.png`. GitHub recommends PNG, JPG, or GIF
under 1 MB; this PNG is 1280x640 and about 46 KB. The source SVG lives at
`docs/assets/social-preview.svg`. The design uses the launch text:

Reference: `https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository/customizing-your-repositorys-social-media-preview`

```text
Local when safe. Fallback when needed.
```

Manual steps:

1. Open `https://github.com/NeapolitanIcecream/darjeeling/settings`.
2. Scroll to Social preview.
3. Click Edit.
4. Upload `docs/assets/social-preview.png`.
5. Save the setting and check the preview.

## Private Vulnerability Reporting

Status: applied.

Command:

```bash
gh api -X PUT repos/NeapolitanIcecream/darjeeling/private-vulnerability-reporting \
  -H "Accept: application/vnd.github+json"
```

Verification:

```bash
gh api repos/NeapolitanIcecream/darjeeling/private-vulnerability-reporting \
  -H "Accept: application/vnd.github+json"
```

Manual check:

1. Open `https://github.com/NeapolitanIcecream/darjeeling/settings/security_analysis`.
2. Confirm private vulnerability reporting is enabled.
3. Confirm `SECURITY.md` appears under the repository Security tab.

## Release

Status: manual.

Manual steps:

1. Open `https://github.com/NeapolitanIcecream/darjeeling/releases/new`.
2. Choose tag `v0.1.0-alpha.1`.
3. Set the release title to `Darjeeling v0.1.0-alpha.1`.
4. Paste the contents of `docs/releases/v0.1.0-alpha.1.md`.
5. Mark the release as a pre-release.
6. Publish only after CI passes and the private feedback wave is complete.
