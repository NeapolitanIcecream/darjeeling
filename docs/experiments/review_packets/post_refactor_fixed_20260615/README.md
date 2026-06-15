# Post-Refactor Fixed Workload Review Packet

This directory is the GitHub-safe review packet for:

```text
runs/post-refactor-fixed-20260615-0954
```

It is intentionally not a full copy of the experiment run root. It contains the
aggregate evidence needed for repository review while excluding raw request-level
records, teacher caches, logs, and generated artifacts that are not needed for a
normal design/code review.

## Uploaded

The packet includes:

- preflight status JSON for cache, live, and L3 guarded readiness;
- a sanitized L3 benchmark aggregate report;
- suite metadata for `cache-full` and `live-residual-500`;
- sanitized suite results for both suites;
- comparison CSV files for both suites;
- per-run `experiment.json`;
- per-run `settings.sanitized.json`;
- per-run `quality.json`;
- per-run `metrics.csv`;
- `packet_manifest.json`, which records the exact included and excluded classes
  of files.

Per-run files live under each suite's `run-details/` directory. The packet avoids
using a nested directory named `runs/` because the repository ignores that name
for full local experiment outputs.

These files are intended to support review of:

- experiment scope and request counts;
- suite commands and return codes;
- cache-backed routing results;
- live L4 cost and latency summaries;
- layer takeover rates;
- field-level metrics;
- L3 benchmark/readiness status;
- per-run configuration after removing provider endpoint details.

## Sanitization

The generated packet applies these transformations:

- absolute repository paths are replaced with `$REPO`;
- `openai_base_url` is removed from sanitized settings;
- `openai_api_key_present` is removed from sanitized settings;
- per-request L3 benchmark `request_results` are omitted;
- full run traces and teacher caches are not copied.

The packet still includes target-level aggregate names and metrics, such as layer
names, experiment names, stream names, model names, and schema-level metric
labels. That is expected and necessary for review.

## Not Uploaded

The packet deliberately excludes:

- `traces.jsonl`;
- `teacher_cache.jsonl`;
- raw live teacher cache contents;
- per-request utterances, gold frames, teacher frames, and final frames;
- `suite.log`;
- `reports/hard_cases.jsonl`;
- `reports/promotions.jsonl`;
- `reports/summary.md`;
- `reports/curves.html`;
- `reports/artifacts.csv`;
- generated artifact directories under `artifacts/`.

The excluded files may contain raw dataset examples, model outputs, API failure
details, local paths, detailed training/calibration internals, or generated
workspace artifacts. They are useful for local debugging, but they are not needed
for ordinary repository review.

## Future Constraint

For future Darjeeling experiment uploads, do not commit an ignored `runs/`
directory wholesale. Instead:

1. Keep complete run roots local or in a controlled artifact store.
2. Commit a review packet under `docs/experiments/review_packets/`.
3. Include aggregate metrics, comparison tables, preflight metadata, suite
   metadata, sanitized settings, and compact quality reports.
4. Exclude raw traces, teacher caches, hard-case records, logs, full prompt/model
   outputs, and generated workspaces by default.
5. If a reviewer needs a raw example, add a separate hand-curated excerpt with
   explicit justification and redaction rather than uploading the full trace.

This packet is a review aid, not a replacement for the local full experiment
record.
