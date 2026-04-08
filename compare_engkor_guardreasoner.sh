#!/bin/bash
# Compare GuardReasoner-VL on Video-SafetyBench: English vs Korean
# Splits: harmful (explicit) and benign (implicit)
# Metric: F1 on prompt harmfulness detection (GuardReasoner-VL criteria)

set -e

MODEL_PATH="${MODEL_PATH:-/home/work/MLLM_Safety/guardrail/models/GuardReasoner-VL-3B}"
DATA_DIR="${DATA_DIR:-./Video-SafetyBench}"
VIDEO_DIR="${VIDEO_DIR:-./Video-SafetyBench}"
RESULTS_DIR="${RESULTS_DIR:-./results}"
NUM_GPUS=4

echo "=============================="
echo " Video-SafetyBench EN vs KO"
echo " Model   : $MODEL_PATH"
echo " Num GPUs: $NUM_GPUS"
echo "=============================="

# Step 1: Extract videos if not yet done
if [ ! -d "$DATA_DIR/video" ]; then
    echo "[1/3] Extracting videos..."
    tar -xzf "$DATA_DIR/video.tar.gz" -C "$DATA_DIR"
else
    echo "[1/3] Videos already extracted, skipping."
fi

# Step 2: Launch 4 parallel inference processes, one per GPU
mkdir -p "$RESULTS_DIR"
echo "[2/3] Running GuardReasoner-VL inference (4 GPUs in parallel)..."
PIDS=()
for GPU_ID in 0 1 2 3; do
    CUDA_VISIBLE_DEVICES=$GPU_ID python inference_guardreasoner.py \
        --model_path "$MODEL_PATH" \
        --data_dir   "$DATA_DIR" \
        --video_dir  "$VIDEO_DIR" \
        --output_dir "$RESULTS_DIR" \
        --gpu_id     $GPU_ID \
        --num_gpus   $NUM_GPUS \
        &
    PIDS+=($!)
done

FAILED=0
for i in "${!PIDS[@]}"; do
    if ! wait "${PIDS[$i]}"; then
        echo "[ERROR] GPU $i process failed (PID ${PIDS[$i]})"
        FAILED=1
    fi
done
[ "$FAILED" -eq 1 ] && echo "[ERROR] One or more GPU processes failed. Check output above." && exit 1
echo "[2/3] All GPUs finished."

# Step 3: Merge shards and evaluate
echo "[3/3] Computing F1 scores..."
python evaluate_guardreasoner.py \
    --results_dir "$RESULTS_DIR" \
    --num_gpus    $NUM_GPUS

echo ""
echo "[DONE] All results saved to $RESULTS_DIR"
