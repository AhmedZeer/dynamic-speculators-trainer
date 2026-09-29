#!/bin/bash
# Single-GPU offline LoRA smoke run for the published Qwen3-8B EAGLE-3 drafter.
# Requires an editable install with the LoRA extra: pip install -e '.[lora]'

set -euo pipefail

MODEL="Qwen/Qwen3-8B"
DRAFTER="RedHatAI/Qwen3-8B-speculator.eagle3"
DATASET="hf:openeurollm/Nemotron-Post-Training-Dataset-v2-decontaminated:chat"
RUN_DIR="/content/drive/MyDrive/dynamic-speculators-dump-v2/eagle3_qwen3_8b_nemotron_lora_smoke"
REGENERATED_DATA="$RUN_DIR/regenerated/qwen3_8b.jsonl"
DATA_DIR="$RUN_DIR/data"
HIDDEN_STATES_DIR="$RUN_DIR/hidden_states"
CHECKPOINT_DIR="$RUN_DIR/checkpoints"
VLLM_PORT=8000
MAX_SAMPLES=64
SEQ_LENGTH=4096
VLLM_MAX_MODEL_LEN=8192
TARGET_LAYER_IDS="2 18 33"

mkdir -p "$RUN_DIR/regenerated" "$HIDDEN_STATES_DIR"

VLLM_PID=""
cleanup() {
    if [[ -n "$VLLM_PID" ]] && kill -0 "$VLLM_PID" 2>/dev/null; then
        kill "$VLLM_PID" 2>/dev/null || true
        wait "$VLLM_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT

echo "=== Launching Qwen3-8B for response and hidden-state generation ==="
python scripts/launch_vllm.py train "$MODEL" \
    --provenance-dir "$RUN_DIR" \
    --hidden-states-path "$HIDDEN_STATES_DIR" \
    --target-layer-ids $TARGET_LAYER_IDS \
    -- \
    --port "$VLLM_PORT" \
    --max-model-len "$VLLM_MAX_MODEL_LEN" \
    --gpu-memory-utilization 0.90 &
VLLM_PID=$!

until curl -sf "http://localhost:${VLLM_PORT}/health" >/dev/null 2>&1; do
    sleep 2
done

echo "=== Regenerating Nemotron responses with Qwen3-8B ==="
speculators regenerate-responses \
    --endpoint "http://localhost:${VLLM_PORT}/v1/chat/completions" \
    --model "$MODEL" \
    --dataset "$DATASET" \
    --limit "$MAX_SAMPLES" \
    --concurrency 16 \
    --max-tokens 1024 \
    --sampling-params '{"temperature":0,"chat_template_kwargs":{"enable_thinking":false}}' \
    --outfile "$REGENERATED_DATA" \
    --resume

echo "=== Preparing regenerated token rows ==="
speculators prepare-data \
    --model "$MODEL" \
    --data "$REGENERATED_DATA" \
    --output "$DATA_DIR" \
    --max-samples "$MAX_SAMPLES" \
    --seq-length "$SEQ_LENGTH"

echo "=== Generating offline verifier hidden states ==="
speculators generate-offline-data \
    --model "$MODEL" \
    --endpoint "http://localhost:${VLLM_PORT}/v1" \
    --preprocessed-data "$DATA_DIR" \
    --output "$HIDDEN_STATES_DIR" \
    --max-samples "$MAX_SAMPLES" \
    --concurrency 16 \
    --validate-outputs \
    --fail-on-error

echo "=== Stopping vLLM and freeing the GPU ==="
cleanup
VLLM_PID=""

echo "=== Fine-tuning EAGLE-3 with LoRA for 10 steps ==="
python -m speculators.train \
    --verifier-name-or-path "$MODEL" \
    --from-pretrained "$DRAFTER" \
    --data-path "$DATA_DIR" \
    --hidden-states-path "$HIDDEN_STATES_DIR" \
    --save-path "$CHECKPOINT_DIR" \
    --speculator-type eagle3 \
    --target-layer-ids $TARGET_LAYER_IDS \
    --total-seq-len "$SEQ_LENGTH" \
    --epochs 1 \
    --max-steps 10 \
    --optimizer adamw \
    --lr 1e-4 \
    --scheduler-type none \
    --lora-r 8 \
    --lora-alpha 16 \
    --lora-dropout 0.05 \
    --num-workers 2 \
    --prefetch-factor 2 \
    --on-missing raise

echo "Adapter: $CHECKPOINT_DIR/0/adapter/"
echo "Merged drafter: $CHECKPOINT_DIR/0/"
echo "Serve with: vllm serve $MODEL --speculative-config '{\"model\":\"$CHECKPOINT_DIR/0\",\"num_speculative_tokens\":3,\"method\":\"eagle3\"}'"
