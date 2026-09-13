#!/bin/bash
# One-shot progress summary for running jobs.
#
# Why this exists: `logs <job> -Follow` streams a raw tqdm bar and needs Ctrl-C to
# exit, which reads like it killed the job (it doesn't). This prints a snapshot and
# returns immediately - safe to run as often as you like.

WD=/raid/session/aicontents/farm
cd "$WD" || exit 1
RUNDIR=$WD/jobs/run

printf "server time %s\n\n" "$(date +%H:%M:%S)"
printf "%-14s %-9s %-22s %-10s %s\n" JOB STATE PROGRESS ELAPSED "STAGE"
printf -- "------------------------------------------------------------------------------\n"

for j in trainB trainC mine stage1 stage1_runB stage2 appletlab; do
  pf=$RUNDIR/$j.pid
  lg=$RUNDIR/$j.log
  [ -f "$lg" ] || continue

  state="idle"
  elapsed="-"
  if [ -f "$pf" ]; then
    p=$(cat "$pf" 2>/dev/null)
    if [ -n "$p" ] && kill -0 "$p" 2>/dev/null; then
      state="RUNNING"
      elapsed=$(ps -o etime= -p "$p" 2>/dev/null | tr -d ' ')
    fi
  fi

  # last tqdm line -> "  60%|#####   | 1426/2373 [10:03<06:33, 2.41it/s]"
  bar=$(tail -c 400 "$lg" 2>/dev/null | tr '\r' '\n' | grep -oE '[0-9]+%\|[^|]*\| *[0-9]+/[0-9]+ \[[^]]*\]' | tail -1)
  if [ -n "$bar" ]; then
    pct=$(echo "$bar"  | grep -oE '^[0-9]+%')
    steps=$(echo "$bar"| grep -oE '[0-9]+/[0-9]+' | tail -1)
    eta=$(echo "$bar"  | grep -oE '<[0-9:]+' | tr -d '<')
    prog="$pct  $steps  eta $eta"
  else
    prog=$(tail -c 200 "$lg" | tr '\r' '\n' | grep -v '^\s*$' | tail -1 | cut -c1-22)
  fi

  # which method / phase
  stage=$(grep -E '^\[[0-9]+/[0-9]+\]' "$lg" 2>/dev/null | tail -1)
  [ -z "$stage" ] && stage=$(grep -E '^=== |^--- ' "$lg" 2>/dev/null | tail -1 | cut -c1-38)

  printf "%-14s %-9s %-22s %-10s %s\n" "$j" "$state" "$prog" "$elapsed" "$stage"
done

printf -- "------------------------------------------------------------------------------\n"
echo
echo "GPUs:"
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu \
  --format=csv,noheader | head -3 | sed 's/^/  /'

echo
echo "Results written so far:"
for f in RUNB_ablation.json RUNC_ablation.json REPRODUCED_stage1.json BINDING_eval.json; do
  [ -f "results/$f" ] && printf "  %-26s %s\n" "$f" "$(stat -c %y "results/$f" | cut -d. -f1)"
done
