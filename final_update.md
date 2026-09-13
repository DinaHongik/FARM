# final_update — FARM 2.0.1

This update retains FARM's code, 100 IFTTT JSON examples, reproduction guides, and reported benchmark values in the README. It removes manuscript and appendix PDFs, LaTeX sources, table-source snapshots, and dedicated manuscript/rebuttal build scripts from the current release.

Experiment code uses `research/FARM/experiments/`, with updated script and guide paths. The public runner supports local inference and offline validation.

Author-reported aggregate provenance remains in [docs/reported_metrics.json](docs/reported_metrics.json). The publishing checks reject manuscript artifacts and continue to block private datasets, keys, and checkpoints.

Publish this committed update from the release folder:

```bash
python3 scripts/publish.py
```

The release is consolidated into one `final_update` commit after the repository's original first commit. Its ancestry excludes the intermediate release commits.
