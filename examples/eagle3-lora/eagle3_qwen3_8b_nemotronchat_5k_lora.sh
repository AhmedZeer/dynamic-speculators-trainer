#!/bin/bash
# Single-GPU offline LoRA smoke run for the published Qwen3-8B EAGLE-3 drafter.
# Requires an editable install with the LoRA extra: pip install -e '.[lora]'

set -euo pipefail

SKIP_REGENERATE=0
SKIP_MISSING=0
while (($#)); do
    case "$1" in
        --skip-regenerate)
            SKIP_REGENERATE=1
            ;;
        --skip-missing)
            SKIP_MISSING=1
            ;;
        -h|--help)
            echo "Usage: $0 [--skip-regenerate] [--skip-missing]"
            echo "  --skip-regenerate  Reuse existing regenerated/prepared data; do not call the response-generation endpoint."
            echo "  --skip-missing     Do not generate missing hidden states; train on available hidden-state samples only."
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            echo "Usage: $0 [--skip-regenerate] [--skip-missing]" >&2
            exit 2
            ;;
    esac
    shift
done

MODEL="Qwen/Qwen3-8B"
DRAFTER="RedHatAI/Qwen3-8B-speculator.eagle3"
DATASET="hf:openeurollm/Nemotron-Post-Training-Dataset-v2-decontaminated:math"
DUMP_ROOT="/content/drive/MyDrive/dynamic-speculators-dump-v2"
VLLM_PORT=8000
# MAX_SAMPLES is the shared cache target; TRAIN_SAMPLES selects a per-run prefix.
MAX_SAMPLES="${MAX_SAMPLES:-5000}"
TRAIN_SAMPLES="${TRAIN_SAMPLES:-$MAX_SAMPLES}"
TRAIN_DATA_RATIO="${TRAIN_DATA_RATIO:-0.9}"
SEQ_LENGTH=8192
VLLM_MAX_MODEL_LEN=8192
TARGET_LAYER_IDS="2 18 33"
EPOCHS=10
LORA_RANK=64
LORA_ALPHA=128
LR=1e-4
SCHEDULER=linear # cosine
# Generation outputs are shared by compatible hyperparameter runs. The default
# points at the existing v2 cache; override DATA_ROOT to relocate it. Use a
# separate cache if dataset, verifier, sequence length, or target layers change.
DATA_ROOT="${DATA_ROOT:-$DUMP_ROOT/eagle3_qwen3_8b_nemotron_lora_v2}"
# Training artifacts stay isolated per run. Reuse RUN_ID to resume a run.
RUN_ID="${RUN_ID:-5kv1}"
RUN_DIR="${RUN_DIR:-$DUMP_ROOT/runs/$RUN_ID}"
REGENERATED_DATA="$DATA_ROOT/regenerated/qwen3_8b.jsonl"
DATA_DIR="$DATA_ROOT/data"
HIDDEN_STATES_DIR="$DATA_ROOT/hidden_states"
CHECKPOINT_DIR="$RUN_DIR/checkpoints"
WANDB_PROJECT="${WANDB_PROJECT:-dynamic-speculators-v2}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-qwen3-8b-nemotronmath-5k-lora}"
export WANDB_PROJECT

mkdir -p "$DATA_ROOT/regenerated" "$HIDDEN_STATES_DIR" "$RUN_DIR"
echo "Shared data cache: $DATA_ROOT"
echo "Run outputs: $RUN_DIR"

VLLM_PID=""
cleanup() {
    if [[ -n "$VLLM_PID" ]] && kill -0 "$VLLM_PID" 2>/dev/null; then
        kill "$VLLM_PID" 2>/dev/null || true
        wait "$VLLM_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT

start_vllm() {
    if [[ -n "$VLLM_PID" ]] && kill -0 "$VLLM_PID" 2>/dev/null; then
        return
    fi

    echo "=== Launching Qwen3-8B because a generation stage has missing outputs ==="
    python scripts/launch_vllm.py train "$MODEL" \
        --provenance-dir "$DATA_ROOT" \
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
}

# --resume skips completed prompt IDs, but --limit applies to newly queued IDs.
# Count completed conversations first so rerunning this script does not generate
# another MAX_SAMPLES rows after the original batch is already on Drive.
COMPLETED_PROMPTS=$(python -c 'import json, pathlib, re, sys; p=pathlib.Path(sys.argv[1]); ids=set();
if p.exists():
    for line in p.open(encoding="utf-8"):
        try:
            row=json.loads(line); key=row.get("primary_id") or re.sub(r"_gen\d+$", "", str(row.get("id", "")))
            if key: ids.add(str(key))
        except (json.JSONDecodeError, AttributeError): pass
print(len(ids))' "$REGENERATED_DATA")

if (( SKIP_REGENERATE )); then
    echo "=== --skip-regenerate: found $COMPLETED_PROMPTS regenerated prompts; response generation is disabled ==="
elif (( COMPLETED_PROMPTS < MAX_SAMPLES )); then
    REMAINING_PROMPTS=$((MAX_SAMPLES - COMPLETED_PROMPTS))
    start_vllm
    echo "=== Regenerating $REMAINING_PROMPTS remaining Nemotron prompts with Qwen3-8B ==="
    speculators regenerate-responses \
        --endpoint "http://localhost:${VLLM_PORT}/v1/chat/completions" \
        --model "$MODEL" \
        --dataset "$DATASET" \
        --limit "$REMAINING_PROMPTS" \
        --concurrency 16 \
        --max-tokens 1024 \
        --sampling-params '{"temperature":0,"chat_template_kwargs":{"enable_thinking":false}}' \
        --outfile "$REGENERATED_DATA" \
        --resume
else
    echo "=== Found $COMPLETED_PROMPTS generated prompts; skipping response generation ==="
fi

PREPARED_SAMPLES=0
if compgen -G "$DATA_DIR/*.arrow" >/dev/null; then
    PREPARED_SAMPLES=$(python -c 'from datasets import load_from_disk; import sys; print(len(load_from_disk(sys.argv[1])))' "$DATA_DIR")
fi

if (( PREPARED_SAMPLES >= MAX_SAMPLES )); then
    echo "=== Found $PREPARED_SAMPLES prepared rows; reusing cache for target $MAX_SAMPLES ==="
else
    if [[ ! -f "$REGENERATED_DATA" ]]; then
        echo "No prepared Arrow dataset or regenerated response file found under $DATA_ROOT; cannot prepare training data." >&2
        exit 1
    fi
    PREPARE_OVERWRITE=()
    if (( PREPARED_SAMPLES > 0 )); then
        if (( SKIP_REGENERATE && COMPLETED_PROMPTS < MAX_SAMPLES )); then
            echo "Prepared cache has $PREPARED_SAMPLES rows, below MAX_SAMPLES=$MAX_SAMPLES, and --skip-regenerate left only $COMPLETED_PROMPTS response prompts. Increase the cache with regeneration enabled or point DATA_ROOT to a separate cache." >&2
            exit 1
        fi
        echo "=== Expanding prepared data cache from $PREPARED_SAMPLES to target $MAX_SAMPLES rows ==="
        PREPARE_OVERWRITE=(--overwrite)
    else
        echo "=== Preparing regenerated token rows ==="
    fi
    speculators prepare-data \
        --model "$MODEL" \
        --data "$REGENERATED_DATA" \
        --output "$DATA_DIR" \
        --max-samples "$MAX_SAMPLES" \
        --seq-length "$SEQ_LENGTH" \
        "${PREPARE_OVERWRITE[@]}"
fi

MISSING_HIDDEN_STATES=$(python -c 'from datasets import load_from_disk; from pathlib import Path; import sys; n=min(int(sys.argv[3]), len(load_from_disk(sys.argv[1]))); out=Path(sys.argv[2]); existing=set();
if out.exists():
    for p in out.glob("hs_*.safetensors"):
        try: existing.add(int(p.stem[3:]))
        except ValueError: pass
print(sum(i not in existing for i in range(n)))' "$DATA_DIR" "$HIDDEN_STATES_DIR" "$MAX_SAMPLES")

if (( SKIP_MISSING )); then
    echo "=== --skip-missing: leaving $MISSING_HIDDEN_STATES hidden-state files ungenerated ==="
elif (( MISSING_HIDDEN_STATES > 0 )); then
    start_vllm
    echo "=== Generating $MISSING_HIDDEN_STATES missing hidden-state files ==="
    speculators generate-offline-data \
        --model "$MODEL" \
        --endpoint "http://localhost:${VLLM_PORT}/v1" \
        --preprocessed-data "$DATA_DIR" \
        --output "$HIDDEN_STATES_DIR" \
        --max-samples "$MAX_SAMPLES" \
        --concurrency 16 \
        --validate-outputs \
        --fail-on-error
else
    echo "=== Found all requested hidden states; skipping extraction ==="
fi

DATA_COUNTS=$(python -c 'from datasets import load_from_disk; from pathlib import Path; import sys; data=load_from_disk(sys.argv[1]); n=min(int(sys.argv[3]), int(len(data)*float(sys.argv[4]))); out=Path(sys.argv[2]); present=set();
if out.exists():
    for p in out.glob("hs_*.safetensors"):
        try:
            i=int(p.stem[3:])
            if 0 <= i < n: present.add(i)
        except ValueError: pass
print(f"{n} {len(present)} {n-len(present)}")' "$DATA_DIR" "$HIDDEN_STATES_DIR" "$TRAIN_SAMPLES" "$TRAIN_DATA_RATIO")
read -r DATASET_SAMPLES AVAILABLE_SAMPLES SKIPPED_SAMPLES <<< "$DATA_COUNTS"
echo "=== Selected $DATASET_SAMPLES rows for this run: $AVAILABLE_SAMPLES have hidden states; $SKIPPED_SAMPLES will be skipped ==="
if (( SKIP_MISSING && AVAILABLE_SAMPLES == 0 )); then
    echo "No training samples have hidden states; cannot start training with --skip-missing." >&2
    exit 1
fi
ON_MISSING=raise
if (( SKIP_MISSING )); then
    ON_MISSING=skip
fi

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
    --train-data-ratio "$TRAIN_DATA_RATIO" \
    --epochs "$EPOCHS" \
    --optimizer adamw \
    --lr "$LR" \
    --scheduler-type "$SCHEDULER" \
    --lora-r "$LORA_RANK" \
    --lora-alpha "$LORA_ALPHA" \
    --lora-dropout 0.05 \
    --num-workers 2 \
    --prefetch-factor 2 \
    --on-missing "$ON_MISSING" \
    --max-train-samples "$TRAIN_SAMPLES" \
    --logger wandb \
    --run-name "$WANDB_RUN_NAME" \
    --log-dir "$RUN_DIR/logs"

echo "Adapter: $CHECKPOINT_DIR/0/adapter/"
echo "Merged drafter: $CHECKPOINT_DIR/0/"
echo "Serve with: vllm serve $MODEL --speculative-config '{\"model\":\"$CHECKPOINT_DIR/0\",\"num_speculative_tokens\":3,\"method\":\"eagle3\"}'"
