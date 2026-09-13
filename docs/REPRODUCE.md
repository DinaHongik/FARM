# Reproducing FARM

## Public 100-example configuration run

Install the repository using the root README, then run `farm validate`. The release contains exactly 100 real requests sampled from the FARM Dataset-v2 test split. The deterministic selection rule and file checksums are in [data/manifest.json](../data/manifest.json). Examples are selected without looking at model predictions, with breadth across service pairs.

Each JSON example contains:

| Field | Meaning |
|---|---|
| `case_id` | Public ID, `ifttt-001` through `ifttt-100` |
| `query` | Actual applet request from the dataset |
| `endpoints.trigger`, `endpoints.action` | One annotated valid endpoint pair, supplied to all workflows |
| `fields` | Input-field slugs, requiredness, and available type/binding metadata |
| `ingredients` | Available trigger outputs; example values are omitted |
| `source_applet_url` | Original IFTTT applet provenance |
| `semantic_binding_reference` | `null`; these records contain no independent semantic field-value gold |

Unknown type or capability information remains unknown. Input-field types are not guessed from their names. A field can be assigned a query span, a trigger ingredient, an optional omission, or a request for missing input. The checker validates the declared constraints and provenance of the proposal. It cannot prove that an ingredient is semantically the correct one or that an automation executes successfully.

Run [local Ollama](LOCAL_LLM.md) using a separate output directory for each configuration. `--arm` selects one workflow; the default runs all three. The endpoint condition and prompts are held constant across arms. The two FARM arms share the initial three calls. The single agent can revise up to its seven-call cap; FARM with feedback uses a verifier and conditional repair; FARM without feedback returns the initial draft unchanged.

Each output directory contains:

| Path | Contents |
|---|---|
| `manifest.json` | Input/code hashes, provider/model, token budget, arms, and metric scope |
| `cache/` | Credential-free request/response cache for resumption |
| `records/<arm>/` | Initial/final drafts, checker reports, revisions, and logical calls |
| `summary.json` | Counts of complete decisions, records with no detected errors, local validity, and calls |

All examples stay in the denominators. Malformed or truncated model responses are failures, while infrastructure errors stop the run for resumption. `semantic_binding_accuracy` stays `null`, and `paper_results_reproduced` stays `false`: this public subset is not the full private paper evaluation. Offline runs are additionally labeled `offline_fixture: true`.

## Full data and checkpoint access

For the full FARM dataset, canonical split manifests, independent reference annotations, exposure manifests, and trained checkpoints, contact **Young Yoon at [young.yoon@hongik.ac.kr](mailto:young.yoon@hongik.ac.kr)**. The release does not bundle these private artifacts. Paper aggregate binding rates alone cannot reconstruct per-example answers, denominators, confidence intervals, or machine-specific timing.

Use the private data only according to the access terms supplied by the authors. Obtain public third-party datasets and baseline packages from their original releases when using the RecipeGen++, TARGE, Interactive IFTTT, or BFCL adapters. Their full datasets and BFCL simulator are not vendored here.

## Full-data retrieval training

The latest corrected trainers live at:

```text
research/FARM/experiments/20260905-controlled-six-training/
```

The recorded environment is Python 3.10, PyTorch 2.6.0+cu124, Transformers 4.57.6, and Sentence Transformers 5.7.0. [requirements-training.txt](../requirements-training.txt) records the observed package versions. Install the PyTorch build appropriate for your platform before installing the remaining requirements. The public configuration runner does not need these packages.

```bash
python -m pip install -r requirements-training.txt
cd research/FARM
```

If the authors provide the raw source dataset rather than prepared Dataset-v2:

```bash
python build_dataset_v2.py --raw /path/to/authorized/full_ifttt.json --out data/v2
python check_dataset_v2.py --root data/v2 --raw /path/to/authorized/full_ifttt.json --verify-rebuild --verify-order
```

If Dataset-v2 is provided directly, retain its supplied manifest and checksums. The following preparation keeps the historical directory structure expected by the six-run experiment. Use one GPU that you are authorized to use; the number below is an example local device index.

The historical preparer loads the base model from the local Hugging Face cache. Download the exact revision first (authenticate with your own account if the model requires access):

```bash
python -c 'from huggingface_hub import snapshot_download; snapshot_download("google/embeddinggemma-300m", revision="57c266a740f537b4dc058e1b0cda161fd15afa75")'
```

```bash
CUDA_VISIBLE_DEVICES=0 python experiments/20260905-five-training-techniques/prepare.py \
  --data-root data/v2 \
  --out experiments/20260905-five-training-techniques/derived

python experiments/20260905-controlled-six-training/prepare_run.py

CUDA_VISIBLE_DEVICES=0 python experiments/20260905-controlled-six-training/train.py \
  --run-id f3_refresh_seed42 \
  --run-root experiments/20260905-controlled-six-training \
  --derived experiments/20260905-controlled-six-training/derived
```

The fixed-negative control is `f3_fixed_seed42`; seed-1337 counterparts are also supported. Service-only and hierarchical controls are `s1_seed1337` and `h6_seed42`. Use the individual trainer command to select your own GPU rather than the historical multi-job launchers, which retain lab-specific device assignments. The trainer enforces exactly one visible GPU. Preparation and training require the full data and model downloads; they cannot be reproduced from the 100-example inference subset.

The primary refresh recipe uses three epochs, batch size 16, maximum sequence length 512, four negatives per query, temperature 0.05, learning rate 2e-5, and refresh after epochs 1 and 2. `experiment.py` and `configs.json` are the detailed configuration sources. `checkpoint_loading.py` preserves the serialized attention window when reloading a checkpoint.

## Full-catalog holdout evaluation

The sources in `experiments/20260905-strong-validation/scripts/` implement family/exposure exclusions, function-pair ranking, service projection, candidate-product coverage, and paired statistics. This release resolves their FARM root relative to the checked-out repository. The private exposure manifest and training artifacts remain required.

After obtaining the original exposure manifest and recreating the training layout above:

```bash
mkdir -p experiments/20260905-strong-validation/results
python experiments/20260905-strong-validation/scripts/prepare_holdout.py
CUDA_VISIBLE_DEVICES=0 python experiments/20260905-strong-validation/scripts/evaluate_frozen_holdout.py --seed 42
```

The evaluator checks development predictions before scoring the holdout. It expects both the fixed and refresh checkpoints for the chosen seed; train or restore both first. The exact paper population requires the original split and prior-exposure records. The public sample should not be substituted for this holdout.

## Binding and structural research protocols

`research/binding/` preserves the last completed DGX binding implementation, including independent controlled-reference scoring and fixture materialization. Its historical runner expects the original 150-case private reference package and the original Ollama Cloud configuration. Its tests that read `private/inputs.jsonl` cannot run without that package. The portable runner uses the same three-workflow logic with a local model adapter and the public 100-example input format.

`research/structural/` contains the separate 150-case structural study. `research/semantic/` contains annotation/reference scoring tools. Their original protocols and private-data dependencies remain explicit. No new DGX numbers are imported into this release's paper tables.

## Checks without private data

From the repository root:

```bash
python -m unittest discover -s tests -v
python -m unittest discover -s research/binding/tests -p test_output_health.py -v
python -m unittest discover -s research/FARM/experiments/20260905T0848KST-farm-architecture-research/tests -v
python scripts/verify_release.py
```

With the training dependencies installed, the corrected objective tests can also be run from their directory:

```bash
cd research/FARM/experiments/20260905-controlled-six-training
CUDA_VISIBLE_DEVICES= python -m unittest discover -s . -p 'test_*.py' -v
```
