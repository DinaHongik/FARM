# FARM: 150-case agent comparison

This is a new configuration-workflow experiment, separate from the completed
retrieval benchmark and the Round 9 public diagnostics. No result is assumed.

The three arms are:

1. **Strong single agent:** a unified configurator with the same catalog evidence,
   deterministic checks, and opportunity to review and repair its draft.
2. **FARM with feedback:** planner, trigger specialist, action specialist, and
   verifier, followed by bounded repair using actual checker feedback.
3. **FARM without feedback/repair:** the same planner, trigger specialist, and
   action specialist produce the initial draft; no verifier/checker feedback is
   delivered and no repair is performed. The checker runs afterward for scoring.

The user's duplicated ablation is one arm, not an additional identical run.

## Frozen inputs and controls

- Use the existing 150-case FARM-v2 Round 9 sample, in its recorded order. These
  cases were previously exposed; this is a controlled repeat, not an untouched test.
- **This run is conditional configuration with oracle endpoints.** Supply one
  annotated valid trigger/action pair equally to all arms, selecting the pair by
  stable canonical ordering before inference. Endpoint selection is not scored
  as a model achievement. The final retriever and its candidates are not used in
  this configuration-only experiment; an end-to-end comparison remains separate.
- Every arm receives identical query, selected endpoints, metadata, and evidence.
- Use Ollama Cloud `deepseek-v4-flash:0731`, as in the earlier public diagnostics.
  The user explicitly instructed use of their Ollama API for this experiment.
  This authorizes the 150 configuration inputs to that provider; their private
  classification and local-only retention of detailed traces remain unchanged.
  Do not relabel FARM data as public to pass the old Round 9 cloud guard.
- Pin the provider model identifier and reported digest when available. Local
  tiny-model connectivity tests are not part of the experiment.
- Temperature 0; fixed inference seed; at most seven semantic model calls per arm
  and case; at most 2,048 generated tokens per call. All arms have the same cap,
  but actual usage differs and must be reported.
- No live platform calls, invented resource observations, user-answer simulation,
  training, or post-outcome prompt selection. Missing information stays unresolved.
- FARM's two arms share their initial three calls, so the ablation starts from
  exactly the same draft. That draft is cached once per case. Each arm's accounting
  includes the shared logical calls even though transport is not duplicated.
- Single-agent review has the same metadata and checker access. The multi-agent
  arm changes role/context decomposition, not the available gold information.
- Source examples, sample strata, private gold, and heuristically generated
  reference bindings are excluded from all prompts.
- Private cases, prompts, outputs, and gold are retained in mode-0600 files. Only
  aggregate results enter the paper package; no credentials enter prompts.

## Outcomes available immediately

Primary: explicit field-decision completeness and absence of demonstrated
provenance violations. Report both separately and their conjunction, plus model
protocol failures and unresolved metadata. These are structural/provenance
outcomes, not independent semantic binding accuracy. Every failed response stays
in the 150-case denominator. Do not count supplied endpoint gold as successful
retrieval or count unknown metadata as a verified compatible type.

Compare initial and final predictions to count rescues and regressions. Pair all
arms by case, report raw denominators, and bootstrap request families 10,000 times.
Report model calls, input/output tokens, provider time, and wall time. No dollar
cost is inferred. Gold scoring occurs only after terminal predictions are saved.

Independent binding accuracy, whole-configuration semantic accuracy, and execution
success remain null until their required references or environment exist. The
binding plan is in `BINDING_METRICS.md`.

## Checks before the full run

Synthetic smoke cases exercise all three workflows. Tests ensure gold is excluded
from prompts, ablated workflows receive no checker feedback, shared initial drafts
match, invalid outputs count as failures, and resume rejects changed identities.
Keep the supplied-endpoint scope explicit. The separate retrieved-candidate
experiment requires the final DGX checkpoint and is not silently substituted.
