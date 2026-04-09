#!/bin/bash
# Unified dataset download script.
#
# Usage:
#   bash download.sh --dataset video_safetybench [--data_dir ./data/video_safetybench] [--download_videos]
#   bash download.sh --dataset videochatgpt      [--data_dir ./data/videochatgpt]       [--download_videos]
#
# For video_safetybench, HF_TOKEN must be set (gated dataset):
#   export HF_TOKEN=hf_xxxxxxxxxxxx

set -e

DATASET=""
DATA_DIR=""
DOWNLOAD_VIDEOS=0

usage() {
    echo "Usage: $0 --dataset <name> [--data_dir <path>] [--download_videos]"
    echo "  Datasets: video_safetybench, videochatgpt"
    exit 1
}

while [[ $# -gt 0 ]]; do
    case $1 in
        --dataset)        DATASET="$2";        shift 2 ;;
        --data_dir)       DATA_DIR="$2";       shift 2 ;;
        --download_videos) DOWNLOAD_VIDEOS=1;  shift ;;
        *) echo "[ERROR] Unknown option: $1"; usage ;;
    esac
done

[[ -z "$DATASET" ]] && echo "[ERROR] --dataset is required." && usage

# ── video_safetybench ─────────────────────────────────────────────────────────
if [[ "$DATASET" == "video_safetybench" ]]; then
    DATA_DIR="${DATA_DIR:-./data/video_safetybench}"

    if [ -z "$HF_TOKEN" ]; then
        echo "[ERROR] HF_TOKEN is not set."
        echo "  export HF_TOKEN=hf_xxxxxxxxxxxx"
        exit 1
    fi

    for cmd in git git-lfs; do
        if ! command -v $cmd &>/dev/null; then
            echo "[ERROR] $cmd is not installed."
            exit 1
        fi
    done

    git lfs install

    AUTH_URL="https://user:${HF_TOKEN}@huggingface.co/datasets/BAAI/Video-SafetyBench"

    if [ "$DOWNLOAD_VIDEOS" = "1" ]; then
        echo "[INFO] Cloning with videos (~13 GB)..."
        git clone "$AUTH_URL" "$DATA_DIR"
    else
        echo "[INFO] Cloning JSON metadata only..."
        GIT_LFS_SKIP_SMUDGE=1 git clone "$AUTH_URL" "$DATA_DIR"
        cd "$DATA_DIR"
        git lfs pull --include="benign_data.json,harmful_data.json"
        cd -
    fi

    echo "[DONE] Video-SafetyBench → $DATA_DIR"

# ── videochatgpt ──────────────────────────────────────────────────────────────
elif [[ "$DATASET" == "videochatgpt" ]]; then
    DATA_DIR="${DATA_DIR:-./data/videochatgpt}"

    VIDEOS_FLAG=""
    [ "$DOWNLOAD_VIDEOS" = "1" ] && VIDEOS_FLAG="--download_videos"

    python download_videochatgpt.py \
        --output_dir "$DATA_DIR" \
        $VIDEOS_FLAG

    echo "[DONE] VideoChatGPT → $DATA_DIR"

else
    echo "[ERROR] Unknown dataset: $DATASET"
    echo "  Available: video_safetybench, videochatgpt"
    exit 1
fi
