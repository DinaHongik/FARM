#!/bin/bash
# Wait for the training jobs to finish, then build the final report by itself.
#
# WHY
#   The three training runs finish at different times over several hours, and the
#   comparison is only meaningful once they are all in. Without this, coming back
#   to the machine means remembering which job wrote which file and running lift +
#   compare_all by hand in the right order.
#
#   It also recovers the one failure mode Run C already hit once: training finished
#   but the eval call raised, leaving good checkpoints and no result file. If that
#   happens again the watchdog scores the checkpoints instead of losing the run.
#
# CPU only. Never trains, never writes into models/.
set -u
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "$DIR/jobs.sh"

RUNDIR=$WD/jobs/run
WATCH="new_trainb new_trainc trainC freeze0"
MAX_HOURS=20
POLL=120

alive() {
  local pf="$RUNDIR/$1.pid"
  [ -f "$pf" ] || return 1
  local pid; pid=$(cat "$pf" 2>/dev/null)
  [ -n "$pid" ] || return 1
  kill -0 "$pid" 2>/dev/null
}

echo "watchdog started $(date -Is)"
echo "watching: $WATCH"
echo "will poll every ${POLL}s, giving up after ${MAX_HOURS}h"
echo

deadline=$(( $(date +%s) + MAX_HOURS * 3600 ))

while :; do
  running=""
  for j in $WATCH; do
    if alive "$j"; then running="$running $j"; fi
  done

  if [ -z "$running" ]; then
    echo "[$(date -Is)] all training jobs finished"
    break
  fi

  # heartbeat: which jobs are alive, and the newest progress line from each
  echo "[$(date -Is)] running:$running"
  for j in $running; do
    last=$(grep -oE '[0-9]+/[0-9]+ \[[0-9:]+<[0-9:]+' "$RUNDIR/$j.log" 2>/dev/null | tail -1)
    [ -n "$last" ] && echo "    $j  $last"
  done
  # nounits + comma separator: the default CSV pads with spaces AND appends units,
  # which shifts awk's whitespace fields and hides half the GPUs.
  nvidia-smi --query-gpu=index,utilization.gpu,memory.used \
             --format=csv,noheader,nounits 2>/dev/null \
    | awk -F', *' '$2+0>0 || $3+0>512 {printf "    gpu%s  util %3d%%  mem %5d MiB\n",$1,$2,$3}'

  if [ "$(date +%s)" -gt "$deadline" ]; then
    echo "[$(date -Is)] TIMEOUT after ${MAX_HOURS}h - still running:$running"
    echo "building the report from whatever finished."
    break
  fi
  sleep "$POLL"
done

echo
echo "=================================================================="
echo "  post-processing"
echo "=================================================================="

# Recovery: Run C's first attempt trained fine and then died in the eval call.
# If that shape recurs (checkpoints present, result file absent) score them here.
if [ ! -f "$WD/results/RUNC_ablation.json" ] \
   && [ -d "$WD/models/runC_ablation_full_finetune_trigger/final" ]; then
  echo "-> RUNC_ablation.json missing but runC checkpoints exist; scoring them"
  $PY "$JOBDIR/eval_runC.py" || echo "   (eval_runC.py failed - continuing)"
fi

echo
echo "-> lift over independence"
$PY "$JOBDIR/lift.py" || echo "   (lift.py failed - continuing)"

echo
echo "-> full comparison"
$PY "$JOBDIR/compare_all.py" || echo "   (compare_all.py failed)"

echo
echo "watchdog done $(date -Is)"
echo "read: results/FINAL_COMPARISON.md   (pull it with: .\\farm.ps1 pull)"
