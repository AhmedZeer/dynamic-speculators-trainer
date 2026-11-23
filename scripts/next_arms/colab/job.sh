#!/usr/bin/env bash
# Run one queue job by ID on this VM. Secrets come from /root/.secrets.env.
#   GEN24K-<i>   response-generation worker i of 2 for the new-data 24K release
#   GENFIN24K    finalize responses + publish train/validation Parquets (waits)
#   P0-* / R*    training run from scripts/next_arms/runs.yaml
set -euo pipefail
JOB=$1
DRIVE=${DRIVE:-/content/drive/MyDrive/dynspec-eagle3-v2}
set -a; source /root/.secrets.env; set +a
export HF_HUB_ENABLE_HF_TRANSFER=0 TOKENIZERS_PARALLELISM=false
cd /content/repo
CFG=examples/eagle3-lora/newdata_v1
case "$JOB" in
  GEN24K-*)
    index=${JOB#GEN24K-}
    prov=$DRIVE/provenance/gen24k-$index/server-responses
    mkdir -p "$prov"
    for d in math code stem chat; do
      speculators lora-bank prepare --config $CFG/$d.yaml --stage select
    done
    python scripts/launch_vllm.py responses Qwen/Qwen3-8B --provenance-dir "$prov" -- \
      --revision b968826d9c46dd6066d109eabc6255188de91218 \
      --served-model-name Qwen/Qwen3-8B --host 127.0.0.1 --port 8000 \
      --dtype bfloat16 --max-model-len 8192 --gpu-memory-utilization 0.92 \
      --max-num-seqs 256 --max-num-batched-tokens 32768 > /content/logs/vllm-gen.log 2>&1 &
    server=$!
    trap 'kill $server 2>/dev/null || true' EXIT
    for i in $(seq 1 180); do
      curl -sf http://127.0.0.1:8000/health >/dev/null && break
      kill -0 $server 2>/dev/null || { tail -50 /content/logs/vllm-gen.log; exit 1; }
      sleep 10
    done
    for d in math code stem chat; do
      mkdir -p out/data/eagle3/newdata-v1-24k/$d/provenance/server-responses
      cp -a "$prov/." out/data/eagle3/newdata-v1-24k/$d/provenance/server-responses/
      speculators lora-bank prepare --config $CFG/$d.yaml --stage responses \
        --num-machines 2 --machine-index "$index"
    done
    ;;
  GENFIN24K)
    for d in math code stem chat; do
      speculators lora-bank prepare --config $CFG/$d.yaml --stage select
    done
    for attempt in $(seq 1 30); do
      ok=1
      for d in math code stem chat; do
        speculators lora-bank finalize-responses --config $CFG/$d.yaml || { ok=0; break; }
      done
      [ $ok = 1 ] && break
      echo "finalize not ready (attempt $attempt); waiting 10 min"; sleep 600
    done
    [ $ok = 1 ] || exit 1
    for d in math code stem chat; do
      speculators lora-bank prepare --config $CFG/$d.yaml --stage data
    done
    ;;
  P0-*|R*)
    python scripts/next_arms/train_run.py "$JOB" --output-root /content/runs
    root=/content/runs/$JOB
    dest=$DRIVE/runs/$JOB
    mkdir -p "$dest"
    cp -a "$root/launcher/." "$dest/launcher/" 2>/dev/null || true
    for run in "$root"/pretraining/*/; do
      name=$(basename "$run")
      mkdir -p "$dest/$name"
      for f in metrics.json timing.json result.json resolved_experiment.json \
               configuration.json train_command.txt speculators.patch \
               hub_checkpoints.json; do
        [ -f "$run/$f" ] && cp "$run/$f" "$dest/$name/"
      done
    done
    ;;
  *) echo "unknown job $JOB"; exit 2;;
esac
