# final_update validation — version 2.0.1

The current release includes code, 100 IFTTT examples, reported benchmark values, and reproduction guides. Manuscript files and dedicated manuscript/rebuttal build scripts are excluded.

The release checks cover example integrity, retained source checksums, credentials, and rejection of manuscript artifacts. Regression tests cover the local model request format, response truncation, caching, resume behavior, shared drafts, validation, and publication checks.

The source layout uses `research/FARM/experiments/`. Script references, test paths, and provenance paths follow that layout. The source manifest records release checksums for files adapted during packaging.

Run the checks from the repository root:

```bash
python3 scripts/verify_release.py
PYTHONPATH=src python3 -m unittest discover -s tests -v
python3 -m unittest discover -s research/FARM/experiments/20260905T0848KST-farm-architecture-research/tests -v
python3 -m unittest discover -s research/binding/tests -p test_output_health.py -v
python3 -m unittest discover -s research/FARM/experiments/20260905-strong-validation/scripts -p test_retrieval_metrics.py -v
```

Offline integration uses the actual 100-example file across all three workflows and makes no model-service requests. It verifies orchestration, not LLM accuracy. The earlier 18 corrected training-objective tests passed on the recovered source with GPUs disabled; no new training or full-data benchmark is claimed for this packaging update.
