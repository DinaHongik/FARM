# Codebase map

The release has a portable public entry point plus the full recovered research implementation. Historical rounds are retained for inspection; the primary code paths below identify the current components.

| Component | Primary source |
|---|---|
| Public command-line interface | `src/farm_release/cli.py` |
| Local model adapter | `src/farm_release/providers.py` |
| Planner → trigger specialist → action specialist → verifier/repair | `src/farm_release/workflows.py` |
| Evidence-aware field configuration, observations, and receipts | `src/farm_release/configuration.py` |
| Independent controlled binding evaluator | `src/farm_release/evaluate.py`, `research/binding/scripts/evaluate.py` |
| Original dataset construction and audit | `research/FARM/build_dataset_v2.py`, `check_dataset_v2.py`, `farm/dataset_v2.py` |
| Original encoders, indexing, retrieval, and selection agents | `research/FARM/train/`, `rag/`, `agents/`, `train_stage1.py`, `e2e_applet.py` |
| Corrected latest training recipe | `research/FARM/experiments/20260905-controlled-six-training/` |
| Checkpoint reload fix | `20260905-controlled-six-training/checkpoint_loading.py` |
| Refreshed negative mining and objectives | `20260905-controlled-six-training/runtime.py`, `objectives.py`, `experiment.py` |
| Stronger retrieval/holdout evaluation | `research/FARM/experiments/20260905-strong-validation/scripts/` |
| Endpoint selection and public benchmark adapters | `research/FARM/experiments/20260904T170552Z-farm-round9-agentic-benchmarks/src/farm_r9/` |
| Earlier round variants | Other dated directories under `research/FARM/experiments/` |
| Latest completed binding experiment | `research/binding/` |
| Separate structural experiment | `research/structural/` |
| Independent annotation and semantic scoring | `research/semantic/` |

The top-level research project layout is preserved because some scripts locate siblings relative to `__file__`. Launch/status scripts from the original experiments may contain historical lab paths or GPU indices; they are not the public quickstart. Use the explicit commands in [REPRODUCE.md](REPRODUCE.md).

Reported benchmark values are retained in the root README and `docs/reported_metrics.json`. Historical evaluation/report-generation code must not overwrite these values using old run aggregates. Manuscript files and dedicated manuscript/rebuttal build scripts are excluded from this code release.

The public examples supply an annotated endpoint pair. They do not exercise Stage 1's complete function catalog. Full retrieval reproduction uses the training/evaluation paths above and the complete data from Young Yoon.
