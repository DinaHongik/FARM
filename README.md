# FARM — Field-Aware Resolution Model

**Version 2.0.1 · final_update · September 13, 2026 · [DinaHongik/FARM](https://github.com/DinaHongik/FARM)**

FARM combines trigger/action retrieval with planning, field assignment, validation, and conditional repair. This release contains our implementation recovered from the workstation and DGX, the latest completed binding workflow, the updated training and evaluation code, **100 real IFTTT examples in JSON**, reproduction guides, and reported benchmark results below. Manuscript sources and PDFs are excluded from this release.

The reported results below are transcribed from the paper. They are not replaced by the DGX rerun scores. The public examples exercise the supplied-endpoint configuration workflow; the full catalog, training data, private evaluation references, and trained checkpoints are available by contacting **Young Yoon: [young.yoon@hongik.ac.kr](mailto:young.yoon@hongik.ac.kr)**.

## Start here

Use Linux, macOS, or Windows with WSL, and Python 3.10 or later. Run commands from this repository's root.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
farm --version
farm validate
python scripts/verify_release.py
```

The public runner has no third-party runtime dependencies. Verify all 100 examples without a model or API key:

```bash
farm run --provider offline --output outputs/offline-check
```

The offline provider tests the orchestration and checker using deterministic fixture responses. It does not measure LLM accuracy. To run an actual model, follow [local LLM setup](docs/LOCAL_LLM.md).

```bash
# Use the exact name of a model already pulled into your local Ollama server.
farm run --provider ollama --model "$FARM_LOCAL_MODEL" --output outputs/local-100
```

The default runs all three workflows: `single_agent`, `farm_feedback`, and `farm_no_feedback`. Each has at most seven logical model calls per example. Both FARM workflows reuse the same initial planner/trigger/action draft. An all-arm run can make up to 1,400 physical model calls over 100 examples because the three initial FARM calls are shared. Use `--limit 1` to check your setup first and a separate output directory for each configuration. Repeat the exact command to resume.

## Results reported in the paper

**Binding re-evaluation — main paper Table 5**

| Workflow | Whole-configuration accuracy (%) |
|---|---:|
| Strong single agent | 75.50 |
| **FARM with repair** | **82.20** |
| FARM without repair | 48.67 |

These are author-reported aggregate results. Sample counts and aggregation details were not supplied with this re-evaluation; we do not infer them or attach counts from the separate DGX experiment. See the [aggregate provenance](docs/reported_metrics.json).

**Service-pair accuracy on 993 requests — main paper Table 2**

| Method | Service-pair accuracy (%) |
|---|---:|
| TARGE | 62.5 |
| RecipeGen++ | 58.6 |
| **FARM** | **71.90** |

**Function retrieval on the same 993 requests — main paper Table 3**

| Method | Correct / requests | Top-1 (%) | Top-5 candidate coverage (%) | Top-10 candidate coverage (%) |
|---|---:|---:|---:|---:|
| FARM | 622 / 993 | 62.64 | 79.66 | 83.79 |

The full catalog has 1,985 trigger functions and 1,520 action functions. Candidate coverage considers the product of the two top-k lists; it is distinct from final function selection and field binding.

**Generation quality — main paper Table 7 (0–1)**

| LLM | Faithfulness | Topic adherence |
|---|---:|---:|
| LLaMA 3 70B | 0.38 | 0.74 |
| Qwen2 72B | 0.41 | 0.76 |
| Mistral Large | 0.36 | 0.71 |
| DeepSeek-Coder | 0.33 | 0.68 |
| IBM Granite 4.0 Small | 0.50 | 0.82 |
| **DeepSeek-V4-Flash** | **0.75** | **0.95** |
| Qwen3.5 397B | 0.62 | 0.86 |
| GPT-OSS 120B | 0.48 | 0.81 |
| Gemma 4 31B | 0.51 | 0.85 |

The results above retain the reported values and metric definitions. The manuscript and appendix are not distributed with the code.

## Code and reproduction guide

| Location | Contents |
|---|---|
| [src/farm_release](src/farm_release) | Portable runner, local model adapter, recovered field checker and three workflows |
| [data/ifttt_examples.json](data/ifttt_examples.json) | Exactly 100 actual IFTTT requests with annotated endpoints and field/ingredient metadata |
| [research/FARM](research/FARM) | Original dataset construction, retrieval, training, agents, evaluation, and experiment sources |
| [research/binding](research/binding) | Exact source from the last completed DGX binding run |
| [research/structural](research/structural) | Three-workflow structural evaluation |
| [research/semantic](research/semantic) | Independent reference preparation and scoring tools |
| [docs/reported_metrics.json](docs/reported_metrics.json) | Author-reported aggregate metrics and their provenance |
| [docs/REPRODUCE.md](docs/REPRODUCE.md) | Dataset format, public runs, output interpretation, and full-data training commands |
| [docs/CODEBASE.md](docs/CODEBASE.md) | Which source modules implement each part of FARM |
| [docs/PROVENANCE.md](docs/PROVENANCE.md) | Code recovery, result provenance, and release adaptations |
| [docs/PUBLISH.md](docs/PUBLISH.md) | Where to push and how to integrate into the existing GitHub repository |

The 100 examples contain endpoint labels, not independently verified field-binding answers. Their output reports completeness, detected structural/provenance errors, unresolved inputs, and calls. It deliberately leaves semantic binding accuracy unset. Reproducing the paper's full-data metrics requires the original data, references, checkpoints, and protocols from Young Yoon. Model choice and hardware can change output quality and latency.

## Verification

```bash
python -m unittest discover -s tests -v
python scripts/verify_release.py
```

[CHANGELOG.md](CHANGELOG.md) records this version. [Publishing instructions](docs/PUBLISH.md) identify the repository destination and verification steps. Local keys, model weights, runtime caches, and the full dataset are excluded from Git. No API key is included.
