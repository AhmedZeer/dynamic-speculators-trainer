#!/usr/bin/env bash
# Zero-token watcher. Start detached:  (setsid nohup ./watch.sh > watch.out 2>&1 &)
# Every INTERVAL s: run status.py on the VM, log one line to watch.log, and stop the VM on
# DONE / FAILED / job process gone / low balance / 4 consecutive unreachable checks.
SESSION=${SESSION:-bench}
WORKDIR=${WORKDIR:-$(dirname "$0")}      # holds colab.sh and status.py; logs go here
INTERVAL=${INTERVAL:-900}
MIN_BALANCE=${MIN_BALANCE:-12}
MAX_MISSES=${MAX_MISSES:-4}
cd "$WORKDIR" || exit 1
misses=0
while true; do
  out=$(timeout 180 ./colab.sh exec -s "$SESSION" -f status.py </dev/null 2>&1)
  st=$(echo "$out" | grep -oE "STATUS=\S+ \S+ ALIVE=\S+" | head -1)
  bal=$(timeout 60 ./colab.sh usage </dev/null 2>&1 | grep -oE "balance: [0-9.]+" | grep -oE "[0-9.]+")
  echo "$(date -u +%FT%TZ) $st balance=$bal" >> watch.log
  echo "$out" > last_status.txt
  reason=""
  case "$st" in *STATUS=DONE*) reason=done;; *STATUS=FAILED*) reason=failed;; *ALIVE=False*) reason=process_died;; "") reason=no_status;; esac
  if [ -n "$bal" ] && awk "BEGIN{exit !($bal < $MIN_BALANCE)}"; then reason=low_balance; fi
  # An unreachable VM is not evidence of failure: tolerate transient connection loss.
  if [ "$reason" = no_status ]; then
    misses=$((misses+1)); echo "  status unreachable ($misses/$MAX_MISSES)" >> watch.log
    if [ $misses -lt "$MAX_MISSES" ]; then sleep "$INTERVAL"; continue; fi
  else misses=0; fi
  if [ -n "$reason" ]; then
    echo "$out" > final_status.txt
    timeout 180 ./colab.sh stop -s "$SESSION" </dev/null >> watch.log 2>&1
    echo "STOPPED reason=$reason" | tee -a watch.log; exit 0
  fi
  sleep "$INTERVAL"
done
