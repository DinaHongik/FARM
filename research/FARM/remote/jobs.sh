#!/bin/bash
# FARM job definitions. Sourced by run_job.sh on the server.
# Add a new experiment by adding one case entry - nothing else changes.

WD=/raid/session/aicontents/farm
PY=$WD/.venv/bin/python
GPUS=${FARM_ALLOWED_GPUS:-0,1,2}
JOBDIR=$WD/jobs
export HF_HOME=$HOME/.cache/huggingface
export OLLAMA_MODELS=$WD/.ollama_models
export OLLAMA_HOST=127.0.0.1:11434
export PATH=$HOME/.local/ollama/bin:$HOME/.local/bin:$PATH

job_cmd() {
  case "$1" in
    stage1)
      # Eval-only on the ORIGINAL (leaky) splits. Reproduces the published table. ~6 min.
      echo "CUDA_VISIBLE_DEVICES=0 $PY $JOBDIR/reproduce_stage1.py gold noisy oneshot --testdir data/test --out results/REPRODUCED_stage1.json"
      ;;
    stage1_runB)
      # Valid clean eval: only meaningful AFTER trainB has retrained on the deduped
      # train set. Evaluating the OLD checkpoints against data/test_dedup is NOT a
      # clean measurement - those weights were trained on the old 90% partition,
      # which covers 94% of the deduped test items. Verified with:
      #   check_leakage.py --train data/train_applets.json --testdir data/test_dedup
      echo "CUDA_VISIBLE_DEVICES=1 $PY $JOBDIR/reproduce_stage1.py gold noisy oneshot --testdir data/test_dedup --out results/RUNB_stage1_clean.json"
      ;;
    leakcheck)
      # Reports train/test overlap of the current splits.
      echo "$PY $JOBDIR/check_leakage.py"
      ;;
    abquery)
      # A/B: planner's extracted fragment vs the full user query, retrieval only.
      # GPU 1 so it does not contend with trainB on GPU 0.
      echo "CUDA_VISIBLE_DEVICES=1 $PY $JOBDIR/ab_planner_query.py"
      ;;
    bindeval)
      # Real field-level binding metrics on EXISTING outputs. CPU only.
      # Replaces the vacuous 1.0s that evaluate_e2e.py returns by empty-collection default.
      echo "$PY $JOBDIR/bindeval.py"
      ;;
    tests)
      # New wiring tests + the existing suite. CPU only, no LLM.
      # agents/tests has 2 PRE-EXISTING failures (verified against the unpatched tree).
      echo "cd $WD && $PY -m pytest $JOBDIR/tests_wiring/ agents/tests/ -q --no-header -p no:cacheprovider"
      ;;
    wiring_check)
      # Which mechanism patches are applied.
      echo "$PY $JOBDIR/wiring/patch_wiring.py --check"
      ;;
    splitfix)
      # Regenerates splits WITH dedup. Writes data/test_dedup/ - does not clobber data/test/.
      echo "$PY $JOBDIR/split_data_dedup.py"
      ;;
    trainB)
      # RUN B: retrain all configs on the deduped split. Needs HF token. Hours.
      # Goes through runB.py, NOT train_lora_ablation.py directly - that script
      # hardcodes the leaky train file, the leaky eval split, and output dirs that
      # would overwrite the published checkpoints. runB.py redirects all three.
      # SINGLE GPU on purpose. With >1 GPU, HF Trainer uses DataParallel and treats
      # per_device_train_batch_size as PER DEVICE, so batch 16 becomes 48 on 3 cards.
      # MultipleNegativesRankingLoss is InfoNCE over IN-BATCH negatives, so that
      # would change the loss itself - Run B would then differ from the published
      # run in two variables and the leakage delta would be uninterpretable.
      echo "CUDA_VISIBLE_DEVICES=0 $PY $JOBDIR/runB.py --ranks 8 16 32 --epochs 3 --batch-size 16 --output results/RUNB_ablation.json"
      ;;
    trainB_dryrun)
      # Verify the three redirections without training anything.
      echo "$PY $JOBDIR/runB.py --dry-run"
      ;;
    mine)
      # Mine hard negatives with Run B's full_finetune encoders. GPU 1 (trainB holds 0).
      # The paper claims "including hard negatives" but training emits only
      # (anchor, positive) - negatives are random in-batch. This produces real ones.
      echo "CUDA_VISIBLE_DEVICES=1 $PY $JOBDIR/mine_negatives.py --mode similarity --n-neg 1"
      ;;
    trainC)
      # RUN C: full_finetune on mined triplets. ONE variable changed vs Run B.
      # GPU 1, batch 16 - same effective batch as Run B, so the comparison holds.
      # expandable_segments reduces the fragmentation that made the first OOM worse.
      echo "CUDA_VISIBLE_DEVICES=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $PY $JOBDIR/runC.py --epochs 3 --batch-size 16 --output results/RUNC_ablation.json"
      ;;
    trainC_dryrun)
      echo "$PY $JOBDIR/runC.py --dry-run"
      ;;
    new_trainb)
      # RUN B2: identical to trainB except the seed. GPU 0.
      # Run B measured layer_freeze 0.440 vs full_finetune 0.570 on the clean split -
      # the reverse of the published 0.580 vs 0.560. That inversion changes a CLAIM,
      # not just a number, and it rests on n=100 at a single seed. Re-running at
      # seed 1337 is the cheapest test of whether the sign is real or sampling noise.
      # If the sign flips between seeds, neither direction is reportable.
      echo "CUDA_VISIBLE_DEVICES=0 $PY $JOBDIR/run_ablation_x.py --tag runB2 --seed 1337 --ranks 8 16 32 --freeze-layers 12 --epochs 3 --batch-size 16"
      ;;
    new_trainc)
      # RUN C with corrected negatives. GPU 1. Guarded against double-start,
      # because 'trainC' is the same experiment under a different job name.
      echo "FARM_SELF_JOB=new_trainc bash $JOBDIR/new_trainc.sh"
      ;;
    freeze0)
      # The experiment that could SAVE the freezing claim rather than retract it. GPU 2.
      # freeze_layers=12 leaves 18% of parameters trainable, and 80% of what it freezes
      # is embed_tokens (201M of 252M) - so "selective layer freezing" is mostly
      # embedding-table freezing. freeze_layers=0 freezes ONLY the embedding table:
      # still 65% of the model frozen, but ~35% trainable instead of 18%.
      # If freeze0 beats full_finetune while freeze12 loses, the mechanism is the
      # embedding table, not the layers - a sharper and more defensible claim, and a
      # direct answer to Reviewer 1(iv) ("primarily a regularization choice").
      # --ranks with no values skips LoRA: only full_finetune + layer_freeze are needed.
      echo "CUDA_VISIBLE_DEVICES=2 $PY $JOBDIR/run_ablation_x.py --tag freeze0 --seed 42 --ranks --freeze-layers 0 --epochs 3 --batch-size 16"
      ;;
    watchdog)
      # Waits for the training jobs, then runs lift + compare_all by itself. CPU only.
      echo "bash $JOBDIR/watchdog.sh"
      ;;
    compare)
      # Rebuild results/FINAL_COMPARISON.md from whatever result files exist. CPU only.
      echo "$PY $JOBDIR/compare_all.py"
      ;;
    lift)
      # The headline comparison: lift over independence, Run B vs Run C.
      echo "$PY $JOBDIR/lift.py"
      ;;
    stage2)
      # Stage 2 agentic end-to-end eval through Ollama.
      echo "CUDA_VISIBLE_DEVICES=0 $PY eval/evaluate_e2e.py"
      ;;
    ollama)
      # (Re)start the Ollama server.
      echo "CUDA_VISIBLE_DEVICES=$GPUS ollama serve"
      ;;
    *)
      echo ""
      ;;
  esac
}

job_desc() {
  case "$1" in
    stage1)       echo "Stage 1 eval, ORIGINAL splits - reproduces the published table (~6 min)" ;;
    stage1_runB)  echo "Stage 1 eval on deduped splits - ONLY valid after trainB" ;;
    leakcheck) echo "Report train/test leakage in the current splits" ;;
    abquery)   echo "A/B planner fragment vs full user query, retrieval only (GPU 1)" ;;
    bindeval)  echo "Real field-level binding metrics on existing outputs (CPU)" ;;
    tests)     echo "Run wiring tests + existing suite (CPU)" ;;
    wiring_check) echo "Show which mechanism patches are applied" ;;
    splitfix)  echo "Regenerate splits with dedup -> data/test_dedup/" ;;
    trainB)        echo "RUN B: retrain encoders on deduped split (needs HF token, hours)" ;;
    trainB_dryrun) echo "Verify Run B redirections without training (seconds)" ;;
    mine)          echo "Mine hard negatives -> data/triplets_*.json (GPU 1, ~3 min)" ;;
    trainC)        echo "RUN C: full_finetune with MINED negatives (GPU 1, ~80 min: grad-checkpointing)" ;;
    trainC_dryrun) echo "Verify Run C redirections without training (seconds)" ;;
    new_trainb)    echo "RUN B2: same as trainB at seed 1337 - is the freeze inversion real? (GPU 0, ~3h)" ;;
    new_trainc)    echo "RUN C with corrected negatives (GPU 1, ~80 min) - guarded vs trainC" ;;
    freeze0)       echo "freeze_layers=0: embedding table only, 35% trainable (GPU 2, ~70 min)" ;;
    watchdog)      echo "Wait for training, then auto-run lift + compare (CPU)" ;;
    compare)       echo "Rebuild results/FINAL_COMPARISON.md (CPU, instant)" ;;
    lift)          echo "Compare lift-over-independence: Run B vs Run C" ;;
    stage2)    echo "Stage 2 end-to-end agentic eval via Ollama" ;;
    ollama)    echo "Start the Ollama server" ;;
    *)         echo "" ;;
  esac
}

ALL_JOBS="stage1 leakcheck abquery bindeval tests wiring_check splitfix trainB_dryrun trainB mine trainC_dryrun trainC new_trainb new_trainc freeze0 watchdog compare lift stage1_runB stage2 ollama"
