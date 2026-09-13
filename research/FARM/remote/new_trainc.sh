#!/bin/bash
# Run C with the CORRECTED mined negatives, guarded against double-starting.
#
# WHY THE GUARD
#   run_job.sh's duplicate check is per job NAME. `trainC` and `new_trainc` are two
#   names for the same experiment: same GPU (1), same output file
#   (results/RUNC_ablation.json), same checkpoint prefix (models/runC_ablation_*).
#   Starting the second while the first runs would put two processes on one V100 and
#   have them overwrite each other's weights mid-epoch. That is silent corruption,
#   not a crash, so it has to be refused up front.
#
# WHY THE NEGATIVES ARE "CORRECTED"
#   The first mining pass took top-K by query similarity with no ceiling. 47% of the
#   resulting negatives had cosine > 0.95 to their own positive - the IFTTT catalog
#   carries near-duplicate entries for the same function, differing only by an
#   inserted "[category]" tag. Training InfoNCE to push those apart destroyed the
#   encoder: trigger R@1 fell 0.750 -> 0.030. mine_negatives.py now applies a
#   similarity ceiling (--max-sim 0.85) and floor (--min-sim 0.35); median candidate
#   similarity dropped 0.945 -> 0.640 and the >0.95 rate is 0.0%.
set -u
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "$DIR/jobs.sh"

RUNDIR=$WD/jobs/run

for other in trainC new_trainc; do
  pf="$RUNDIR/$other.pid"
  [ -f "$pf" ] || continue
  pid=$(cat "$pf" 2>/dev/null) || continue
  [ -n "$pid" ] || continue
  if kill -0 "$pid" 2>/dev/null && [ "$other" != "${FARM_SELF_JOB:-}" ]; then
    echo "REFUSING: '$other' is already running (pid $pid)."
    echo "It is the same experiment - same GPU, same output file, same checkpoints."
    echo "Watch it instead:  .\\farm.ps1 logs $other"
    exit 1
  fi
done

if [ ! -f "$WD/data/triplets_trigger.json" ] || [ ! -f "$WD/data/triplets_action.json" ]; then
  echo "missing mined triplets - running the mining pass first"
  CUDA_VISIBLE_DEVICES=1 $PY "$JOBDIR/mine_negatives.py" --mode similarity --n-neg 1 \
    --max-sim 0.85 --min-sim 0.35 || exit 1
fi

exec env CUDA_VISIBLE_DEVICES=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  $PY "$JOBDIR/runC.py" --epochs 3 --batch-size 16 --output results/RUNC_ablation.json
