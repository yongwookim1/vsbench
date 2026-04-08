#!/bin/bash
# Compare GuardReasoner-VL on Video-SafetyBench: English vs Korean
# Splits: harmful (explicit) and benign (implicit)
# Metric: F1 on prompt harmfulness detection (GuardReasoner-VL criteria)

set -e

MODEL_PATH="${MODEL_PATH:-yueliu1999/GuardReasoner-VL-7B}"
DATA_DIR="${DATA_DIR:-./Video-SafetyBench}"
VIDEO_DIR="${VIDEO_DIR:-./Video-SafetyBench}"
RESULTS_DIR="${RESULTS_DIR:-./results}"

echo "=============================="
echo " Video-SafetyBench EN vs KO"
echo " Model: $MODEL_PATH"
echo "=============================="

# Step 1: Extract videos if not yet done
if [ ! -d "$DATA_DIR/video" ]; then
    echo "[1/3] Extracting videos..."
    tar -xzf "$DATA_DIR/video.tar.gz" -C "$DATA_DIR"
else
    echo "[1/3] Videos already extracted, skipping."
fi

# Step 2: Inference — all splits and languages in one pass
echo "[2/3] Running GuardReasoner-VL inference..."
python inference_guardreasoner.py \
    --model_path "$MODEL_PATH" \
    --data_dir   "$DATA_DIR" \
    --video_dir  "$VIDEO_DIR" \
    --output_dir "$RESULTS_DIR"

# Step 3: Evaluate
echo "[3/3] Computing F1 scores..."
python evaluate_guardreasoner.py --results_dir "$RESULTS_DIR"

echo ""
echo "[DONE] All results saved to $RESULTS_DIR"
