# Agent Instructions

- Darjeeling core must remain dataset- and application-independent. Core includes framework runtime, compiler/evaluation harnesses, reusable prompts/program text, default diagnostics, default probe generation, and shared tests.
- Core may operate on schemas, intent names, slot keys, utterances, labels, and examples only as runtime/input data. It must not hard-code application-specific intent or slot names, dataset utterances, labels, request ids, or experiment failure cases.
- Application or dataset adapter code may contain application-specific schema names and dataset-independent business logic required to connect Darjeeling to a concrete task, but that code must stay separated from core.
- Task-specific isolated L1/L2/L3 workspaces and generated target artifacts are owned by the L4 agent flow. Repository coding agents must not directly edit them; change the repo-level harnesses, prompts, tests, adapters, or contracts that govern those workspaces instead.
- Experiment artifacts and experiment docs may record dataset-specific evidence and failure examples, but that evidence must not become a core default or reusable core rule.
