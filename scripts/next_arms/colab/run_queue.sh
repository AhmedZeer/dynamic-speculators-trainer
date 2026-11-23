#!/usr/bin/env bash
# Work through this VM's queue (one job ID per line on Drive) one job at a time.
# Per-job status:  $DRIVE/logs/<vm>/<job>/STATUS   (RUNNING|DONE|FAILED:<rc> <ts>)
# Queue status:    $DRIVE/logs/<vm>/QUEUE_STATUS   (watched by watch.sh)
# Finished jobs are skipped; a failed job is retried once, then the queue moves on.
set -uo pipefail
VM=$1
DRIVE=${DRIVE:-/content/drive/MyDrive/dynspec-eagle3-v2}
export DRIVE
LOGS=$DRIVE/logs/$VM
mkdir -p "$LOGS" /content/logs
ts() { date -u +%FT%TZ; }
echo "SETUP $(ts)" > "$LOGS/QUEUE_STATUS"
bash /content/repo/scripts/next_arms/colab/setup.sh > /content/logs/setup.log 2>&1 || {
  cp /content/logs/setup.log "$LOGS/"; echo "FAILED:setup $(ts)" > "$LOGS/QUEUE_STATUS"; exit 1; }
cp /content/logs/setup.log "$LOGS/"
failures=0
while true; do
  next=""
  while read -r job; do
    job=${job%%#*}; job=$(echo "$job" | xargs)
    [ -z "$job" ] && continue
    st=$(cat "$LOGS/$job/STATUS" 2>/dev/null || true)
    case "$st" in DONE*) continue;; FAILED:*x2*) continue;; esac
    next=$job; break
  done < "$DRIVE/queues/$VM.txt"
  [ -z "$next" ] && break
  mkdir -p "$LOGS/$next"
  prior=$(cat "$LOGS/$next/STATUS" 2>/dev/null || true)
  echo "RUNNING $next $(ts)" > "$LOGS/QUEUE_STATUS"
  echo "RUNNING $(ts)" > "$LOGS/$next/STATUS"
  bash /content/repo/scripts/next_arms/colab/job.sh "$next" > "/content/logs/$next.log" 2>&1
  rc=$?
  cp "/content/logs/$next.log" "$LOGS/$next/job.log"
  if [ $rc -eq 0 ]; then echo "DONE $(ts)" > "$LOGS/$next/STATUS"
  else
    failures=$((failures+1))
    case "$prior" in FAILED:*) echo "FAILED:${rc}x2 $(ts)" > "$LOGS/$next/STATUS";;
                     *) echo "FAILED:${rc} $(ts)" > "$LOGS/$next/STATUS";; esac
  fi
done
if [ $failures -gt 0 ]; then echo "DONE_WITH_FAILURES $(ts)" > "$LOGS/QUEUE_STATUS"
else echo "DONE $(ts)" > "$LOGS/QUEUE_STATUS"; fi
