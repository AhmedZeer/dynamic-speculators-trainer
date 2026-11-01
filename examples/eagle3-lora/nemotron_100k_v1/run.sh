#!/usr/bin/env bash
# Run from the repository root. Use separate terminals for serve and responses.
set -euo pipefail
config_dir=examples/eagle3-lora/nemotron_100k_v1
run_root=out/data/eagle3/nemotron-100k
case "${1:-}" in
  select|data)
    for domain in math code stem chat; do
      speculators lora-bank prepare --config "$config_dir/$domain.yaml" --stage "$1"
    done
    ;;
  serve)
    python scripts/launch_vllm.py responses Qwen/Qwen3-8B \
      --provenance-dir "$run_root/math/provenance/server-responses" -- \
      --revision b968826d9c46dd6066d109eabc6255188de91218 \
      --served-model-name Qwen/Qwen3-8B \
      --host 127.0.0.1 --port 8000 \
      --dtype bfloat16 --max-model-len 8192 \
      --gpu-memory-utilization 0.90 \
      --max-num-seqs 256 --max-num-batched-tokens 524288
    ;;
  responses)
    for domain in math code stem chat; do
      # The single response server is shared by all four categories. Publish
      # its launch provenance with each category's response-stage record.
      if [[ "$domain" != math ]]; then
        mkdir -p "$run_root/$domain/provenance/server-responses"
        cp -a "$run_root/math/provenance/server-responses/." \
          "$run_root/$domain/provenance/server-responses/"
      fi
      speculators lora-bank prepare --config "$config_dir/$domain.yaml" --stage responses
    done
    ;;
  *)
    echo "Usage: bash $config_dir/run.sh {select|serve|responses|data}" >&2
    exit 2
    ;;
esac
