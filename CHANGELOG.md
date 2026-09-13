# Changelog

## 2.0.1 — final_update

- Removed manuscript and appendix sources, table TeX files, PDFs, and manuscript/rebuttal authoring scripts from the code release.
- Retained reported benchmark values in the README and moved aggregate provenance to `docs/reported_metrics.json`.
- Updated the CLI, guides, and source manifest to work without manuscript files.
- Added ignore rules and publishing checks to reject manuscript artifacts.
- Fixed publishing checks for the existing data documentation and source scripts.
- Standardized experiment paths and documentation, and retained local inference and offline validation in the public runner.

## 2.0.0 — 2026-09-13

- Consolidated the FARM source code from the workstation and DGX, including dataset construction, retrieval, training, selection, configuration, validation, and evaluation.
- Recovered the completed `binding_8192_direct` workflow and the corrected six-run training and stronger holdout-evaluation code.
- Added 100 actual IFTTT examples, schema metadata, selection provenance, and reproducible file checksums.
- Added a portable command-line runner for local models, with shared FARM drafts, per-call caching, resumable records, and offline orchestration checks.
- Published snapshots of all 36 tables in the current manuscript and appendix, using the paper's reported results rather than DGX rerun aggregates.
- Added reproduction, full-data training, code-navigation, provider, and publishing guides.
- Configured DinaHongik/FARM as the intended GitHub destination and excluded credentials, private datasets, checkpoints, and generated runs.
