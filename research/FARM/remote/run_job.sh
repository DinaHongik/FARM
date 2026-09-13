#!/bin/bash
# Server-side job runner. Launch / status / stop / logs, tracked by pidfile.
#
#   run_job.sh start  <job>
#   run_job.sh status [job]
#   run_job.sh stop   <job>
#   run_job.sh log    <job> [lines]
#
# Pidfiles (not pgrep patterns) are used deliberately: a pgrep -f pattern also
# matches the ssh command carrying it, which silently kills the wrong process.

set -u
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "$DIR/jobs.sh"

RUNDIR=$WD/jobs/run
mkdir -p "$RUNDIR"

pidfile() { echo "$RUNDIR/$1.pid"; }
logfile() { echo "$RUNDIR/$1.log"; }

is_running() {
  local pf; pf=$(pidfile "$1")
  [ -f "$pf" ] || return 1
  local pid; pid=$(cat "$pf" 2>/dev/null)
  [ -n "$pid" ] || return 1
  kill -0 "$pid" 2>/dev/null
}

case "${1:-help}" in

  start)
    JOB="${2:?usage: run_job.sh start <job>}"
    CMD=$(job_cmd "$JOB")
    if [ -z "$CMD" ]; then echo "unknown job: $JOB"; echo "available: $ALL_JOBS"; exit 1; fi
    if is_running "$JOB"; then
      echo "ALREADY RUNNING: $JOB (pid $(cat "$(pidfile "$JOB")"))"; exit 1
    fi
    LOG=$(logfile "$JOB")
    cd "$WD" || exit 1
    {
      echo "=== job:$JOB started $(date -Is) ==="
      echo "=== cmd: $CMD"
      echo
    } > "$LOG"
    setsid bash -c "$CMD" >> "$LOG" 2>&1 < /dev/null &
    echo $! > "$(pidfile "$JOB")"
    echo "started $JOB  pid=$!  log=$LOG"
    ;;

  status)
    if [ $# -ge 2 ]; then LIST="$2"; else LIST="$ALL_JOBS"; fi
    printf "%-11s %-9s %-21s %s\n" JOB STATE "LAST LOG" DESCRIPTION
    for j in $LIST; do
      if is_running "$j"; then st="RUNNING"; else st="idle"; fi
      lg=$(logfile "$j")
      if [ -f "$lg" ]; then ts=$(stat -c '%y' "$lg" | cut -d. -f1); else ts="-"; fi
      printf "%-11s %-9s %-21s %s\n" "$j" "$st" "$ts" "$(job_desc "$j")"
    done
    ;;

  stop)
    JOB="${2:?usage: run_job.sh stop <job>}"
    if is_running "$JOB"; then
      pid=$(cat "$(pidfile "$JOB")")
      # kill the whole process group (setsid gave it its own)
      kill -TERM -"$pid" 2>/dev/null || kill -TERM "$pid" 2>/dev/null
      sleep 2
      if kill -0 "$pid" 2>/dev/null; then kill -KILL -"$pid" 2>/dev/null || kill -KILL "$pid" 2>/dev/null; fi
      rm -f "$(pidfile "$JOB")"
      echo "stopped $JOB"
    else
      echo "$JOB is not running"
    fi
    ;;

  log)
    JOB="${2:?usage: run_job.sh log <job> [lines]}"
    N="${3:-60}"
    tail -n "$N" "$(logfile "$JOB")" 2>/dev/null || echo "no log yet for $JOB"
    ;;

  follow)
    JOB="${2:?usage: run_job.sh follow <job>}"
    tail -f -n 40 "$(logfile "$JOB")"
    ;;

  *)
    echo "usage: run_job.sh {start|status|stop|log|follow} <job>"
    echo "jobs: $ALL_JOBS"
    ;;
esac
