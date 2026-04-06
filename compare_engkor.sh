#!/bin/bash
# Compare ASR between English (original) and Korean Video-SafetyBench
# Target model : Qwen3-VL-32B-Instruct
# Judge model  : Qwen2.5-72B-Instruct

set -e

MODEL_PATH="${MODEL_PATH:-../models_cache/Qwen3-VL-32B-Instruct}"
JUDGE_MODEL="${JUDGE_MODEL:-../models_cache/Qwen2.5-72B-Instruct}"
DATA_DIR="${DATA_DIR:-./Video-SafetyBench}"
VIDEO_DIR="${VIDEO_DIR:-./Video-SafetyBench/video}"
RESULTS_DIR="${RESULTS_DIR:-./results}"

echo "=============================="
echo " Video-SafetyBench EN vs KO"
echo "=============================="
echo " Target : $MODEL_PATH"
echo " Judge  : $JUDGE_MODEL"
echo " Results: $RESULTS_DIR"
echo "=============================="

# Step 1: Extract videos if not yet done
if [ ! -d "$VIDEO_DIR" ]; then
    echo "[1/5] Extracting videos..."
    tar -xzf "$DATA_DIR/video.tar.gz" -C "$DATA_DIR"
else
    echo "[1/5] Videos already extracted, skipping."
fi

# Step 2–3: Inference (harmful)
echo "[2/5] Inference — harmful EN..."
python inference_qwen3vl.py \
    --data_dir    "$DATA_DIR" \
    --video_dir   "$VIDEO_DIR" \
    --model_path  "$MODEL_PATH" \
    --query_type  harmful \
    --lang        en \
    --output_dir  "$RESULTS_DIR"

echo "[3/5] Inference — harmful KO..."
python inference_qwen3vl.py \
    --data_dir    "$DATA_DIR" \
    --video_dir   "$VIDEO_DIR" \
    --model_path  "$MODEL_PATH" \
    --query_type  harmful \
    --lang        ko \
    --output_dir  "$RESULTS_DIR"

# Step 4–5: Inference (benign)
echo "[4/5] Inference — benign EN..."
python inference_qwen3vl.py \
    --data_dir    "$DATA_DIR" \
    --video_dir   "$VIDEO_DIR" \
    --model_path  "$MODEL_PATH" \
    --query_type  benign \
    --lang        en \
    --output_dir  "$RESULTS_DIR"

echo "[5/5] Inference — benign KO..."
python inference_qwen3vl.py \
    --data_dir    "$DATA_DIR" \
    --video_dir   "$VIDEO_DIR" \
    --model_path  "$MODEL_PATH" \
    --query_type  benign \
    --lang        ko \
    --output_dir  "$RESULTS_DIR"

# Evaluate & compare
echo ""
echo "[EVAL] Evaluating harmful queries..."
python evaluate_asr.py \
    --results_dir "$RESULTS_DIR" \
    --query_type  harmful \
    --judge_model "$JUDGE_MODEL"

echo ""
echo "[EVAL] Evaluating benign queries..."
python evaluate_asr.py \
    --results_dir "$RESULTS_DIR" \
    --query_type  benign \
    --judge_model "$JUDGE_MODEL"

echo ""
echo "[DONE] All results saved to $RESULTS_DIR"
