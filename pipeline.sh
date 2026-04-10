#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# pipeline.sh — Full GuardReasoner-VL evaluation pipeline
#
# Steps (each skipped automatically if outputs already exist):
#   1. Download dataset
#   2. Translate questions to Korean
#   3. Run GuardReasoner-VL inference (multi-GPU parallel)
#   4. Merge shards & compute F1
#
# Usage:
#   bash pipeline.sh --dataset video_safetybench \
#       --data_dir ./Video-SafetyBench \
#       --model_path /path/to/GuardReasoner-VL-3B
#
#   bash pipeline.sh --dataset videochatgpt \
#       --data_dir ./data/videochatgpt \
#       --download_videos
#
# Force flags (re-run a specific step even if outputs exist):
#   --force-download  --force-translate  --force-inference  --force-evaluate
# ─────────────────────────────────────────────────────────────────────────────
set -e

# ── Defaults ──────────────────────────────────────────────────────────────────
DATASET=""
MODEL_PATH="${MODEL_PATH:-/home/work/MLLM_Safety/guardrail/models/GuardReasoner-VL-3B}"
DATA_DIR=""           # default derived from DATASET below
RESULTS_DIR=""        # default derived from DATASET below
NUM_GPUS=4
DOWNLOAD_VIDEOS=0
DOWNLOAD_METHOD="datasets"
TRANSLATE_BACKEND="google"
FORCE_DOWNLOAD=0
FORCE_TRANSLATE=0
FORCE_INFERENCE=0
FORCE_EVALUATE=0

usage() {
    cat <<EOF
Usage: $0 --dataset <name> [options]

Required:
  --dataset <name>          Dataset name: video_safetybench | videochatgpt

Paths:
  --model_path <path>       GuardReasoner-VL model (default: \$MODEL_PATH)
  --data_dir <path>         Dataset data dir (default: ./data/<dataset>)
  --results_dir <path>      Results output dir (default: ./results/<dataset>)

Options:
  --num_gpus <n>            Number of GPUs for inference (default: 4)
  --download_videos         Also download videos during the download step
  --method datasets|git-lfs Download method for videochatgpt (default: datasets)
  --translate_backend       google | qwen (default: google)
  --qwen_model_path <path>  Qwen model for translation (only with --translate_backend qwen)

Force re-run a step:
  --force-download
  --force-translate
  --force-inference
  --force-evaluate
EOF
    exit 1
}

# ── Argument parsing ──────────────────────────────────────────────────────────
QWEN_MODEL_PATH="../models_cache/Qwen3-VL-32B-Instruct"

while [[ $# -gt 0 ]]; do
    case $1 in
        --dataset)           DATASET="$2";           shift 2 ;;
        --model_path)        MODEL_PATH="$2";        shift 2 ;;
        --data_dir)          DATA_DIR="$2";          shift 2 ;;
        --results_dir)       RESULTS_DIR="$2";       shift 2 ;;
        --num_gpus)          NUM_GPUS="$2";          shift 2 ;;
        --download_videos)   DOWNLOAD_VIDEOS=1;             shift ;;
        --method)            DOWNLOAD_METHOD="$2";          shift 2 ;;
        --translate_backend) TRANSLATE_BACKEND="$2";        shift 2 ;;
        --qwen_model_path)   QWEN_MODEL_PATH="$2";  shift 2 ;;
        --force-download)    FORCE_DOWNLOAD=1;       shift ;;
        --force-translate)   FORCE_TRANSLATE=1;      shift ;;
        --force-inference)   FORCE_INFERENCE=1;      shift ;;
        --force-evaluate)    FORCE_EVALUATE=1;       shift ;;
        -h|--help)           usage ;;
        *) echo "[ERROR] Unknown option: $1"; usage ;;
    esac
done

[[ -z "$DATASET" ]] && echo "[ERROR] --dataset is required." && usage

DATA_DIR="${DATA_DIR:-./data/$DATASET}"
RESULTS_DIR="${RESULTS_DIR:-./results/$DATASET}"

# ── Banner ────────────────────────────────────────────────────────────────────
echo "══════════════════════════════════════════════════════"
echo " GuardReasoner-VL Pipeline"
echo " Dataset    : $DATASET"
echo " Model      : $MODEL_PATH"
echo " Data dir   : $DATA_DIR"
echo " Results    : $RESULTS_DIR"
echo " GPUs       : $NUM_GPUS"
echo " Translate  : $TRANSLATE_BACKEND"
echo "══════════════════════════════════════════════════════"

# ── Helper: Python adapter check ──────────────────────────────────────────────
adapter_check() {
    # $1 = method name on the adapter (is_downloaded | is_translated | is_videos_ready)
    python - <<PYEOF
from adapters import get_adapter
from pathlib import Path
a = get_adapter("$DATASET")
print("1" if a.$1(Path("$DATA_DIR")) else "0")
PYEOF
}

shard_check() {
    python - <<PYEOF
from adapters import get_adapter
from pathlib import Path
import sys
a = get_adapter("$DATASET")
results_dir = Path("$RESULTS_DIR")
for split in a.splits:
    for lang in a.langs:
        for gpu_id in range($NUM_GPUS):
            p = results_dir / f"{split}_{lang}_guardreasoner_gpu{gpu_id}.jsonl"
            if not p.exists():
                print("0")
                sys.exit()
print("1")
PYEOF
}

# ══ Step 1: Download ══════════════════════════════════════════════════════════
echo ""
echo "━━ [1/4] Download ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

DOWNLOADED=$(adapter_check is_downloaded)

if [[ "$DOWNLOADED" == "1" && "$FORCE_DOWNLOAD" == "0" ]]; then
    echo "  [SKIP] Dataset already present in $DATA_DIR"
else
    VIDEOS_FLAG=""
    [ "$DOWNLOAD_VIDEOS" = "1" ] && VIDEOS_FLAG="--download_videos"
    bash download.sh --dataset "$DATASET" --data_dir "$DATA_DIR" --method "$DOWNLOAD_METHOD" $VIDEOS_FLAG
fi

# ── Video extraction (video_safetybench only) ─────────────────────────────────
if [[ "$DATASET" == "video_safetybench" ]]; then
    VIDEOS_READY=$(adapter_check is_videos_ready)
    if [[ "$VIDEOS_READY" == "0" && "$DOWNLOAD_VIDEOS" == "1" ]]; then
        echo "  [INFO] Extracting videos..."
        tar -xzf "$DATA_DIR/video.tar.gz" -C "$DATA_DIR"
    elif [[ "$VIDEOS_READY" == "0" ]]; then
        echo "  [WARN] Videos not extracted. Pass --download_videos or extract manually:"
        echo "         tar -xzf $DATA_DIR/video.tar.gz -C $DATA_DIR"
    else
        echo "  [SKIP] Videos already extracted."
    fi
fi

# ══ Step 2: Translate to Korean ═══════════════════════════════════════════════
echo ""
echo "━━ [2/4] Translate to Korean ━━━━━━━━━━━━━━━━━━━━━━━━"

TRANSLATED=$(adapter_check is_translated)

if [[ "$TRANSLATED" == "1" && "$FORCE_TRANSLATE" == "0" ]]; then
    echo "  [SKIP] Korean translations already exist in $DATA_DIR"
else
    QWEN_FLAG=""
    [ "$TRANSLATE_BACKEND" = "qwen" ] && QWEN_FLAG="--model_path $QWEN_MODEL_PATH"

    python translate.py \
        --dataset  "$DATASET" \
        --data_dir "$DATA_DIR" \
        --backend  "$TRANSLATE_BACKEND" \
        $QWEN_FLAG
fi

# ══ Step 3: Inference (multi-GPU) ════════════════════════════════════════════
echo ""
echo "━━ [3/4] Inference ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

mkdir -p "$RESULTS_DIR"
INFERENCE_DONE=$(shard_check)

if [[ "$INFERENCE_DONE" == "1" && "$FORCE_INFERENCE" == "0" ]]; then
    echo "  [SKIP] All GPU shards already exist in $RESULTS_DIR"
else
    echo "  Launching $NUM_GPUS GPU processes..."
    PIDS=()
    for GPU_ID in $(seq 0 $((NUM_GPUS - 1))); do
        CUDA_VISIBLE_DEVICES=$GPU_ID python inference_guardreasoner.py \
            --dataset    "$DATASET" \
            --model_path "$MODEL_PATH" \
            --data_dir   "$DATA_DIR" \
            --output_dir "$RESULTS_DIR" \
            --gpu_id     $GPU_ID \
            --num_gpus   $NUM_GPUS \
            &
        PIDS+=($!)
        echo "  GPU $GPU_ID — PID ${PIDS[-1]}"
    done

    FAILED=0
    for i in "${!PIDS[@]}"; do
        if ! wait "${PIDS[$i]}"; then
            echo "  [ERROR] GPU $i failed (PID ${PIDS[$i]}) — check $RESULTS_DIR/gpu${i}.log"
            FAILED=1
        fi
    done
    [ "$FAILED" -eq 1 ] && echo "[ERROR] Inference failed." && exit 1
    echo "  All GPU processes finished."
fi

# ══ Step 4: Evaluate ═════════════════════════════════════════════════════════
echo ""
echo "━━ [4/4] Evaluate ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

SUMMARY="$RESULTS_DIR/summary.json"

if [[ -f "$SUMMARY" && "$FORCE_EVALUATE" == "0" ]]; then
    echo "  [SKIP] Summary already exists: $SUMMARY"
else
    python evaluate_guardreasoner.py \
        --dataset     "$DATASET" \
        --results_dir "$RESULTS_DIR" \
        --num_gpus    "$NUM_GPUS"
fi

# ══ Done ══════════════════════════════════════════════════════════════════════
echo ""
echo "══════════════════════════════════════════════════════"
echo " [DONE]  Results → $RESULTS_DIR"
echo "══════════════════════════════════════════════════════"
