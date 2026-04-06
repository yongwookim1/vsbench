#!/bin/bash
# Download BAAI/Video-SafetyBench dataset using git lfs
# Requires: git, git-lfs, and HF_TOKEN env var set (gated dataset)
#
# Usage:
#   export HF_TOKEN=hf_xxxxxxxxxxxx
#   bash download.sh
#
# To also download videos (13.2 GB), set: DOWNLOAD_VIDEOS=1
#   DOWNLOAD_VIDEOS=1 bash download.sh

set -e

DATASET_URL="https://huggingface.co/datasets/BAAI/Video-SafetyBench"
DEST_DIR="./Video-SafetyBench"
DOWNLOAD_VIDEOS="${DOWNLOAD_VIDEOS:-0}"

# Check dependencies
if ! command -v git &>/dev/null; then
    echo "[ERROR] git is not installed."
    exit 1
fi

if ! command -v git-lfs &>/dev/null; then
    echo "[ERROR] git-lfs is not installed. Install it with:"
    echo "  sudo apt install git-lfs   # Ubuntu/Debian"
    echo "  brew install git-lfs       # macOS"
    exit 1
fi

if [ -z "$HF_TOKEN" ]; then
    echo "[ERROR] HF_TOKEN is not set. Export your HuggingFace token:"
    echo "  export HF_TOKEN=hf_xxxxxxxxxxxx"
    exit 1
fi

# Initialize git lfs
git lfs install

# Embed token into clone URL
AUTH_URL="https://user:${HF_TOKEN}@huggingface.co/datasets/BAAI/Video-SafetyBench"

if [ "$DOWNLOAD_VIDEOS" = "1" ]; then
    echo "[INFO] Cloning with videos (13.2 GB)..."
    git clone "$AUTH_URL" "$DEST_DIR"
else
    echo "[INFO] Cloning JSON files only (skipping video.tar.gz)..."
    # Skip LFS blobs during clone, then selectively pull only JSON files
    GIT_LFS_SKIP_SMUDGE=1 git clone "$AUTH_URL" "$DEST_DIR"
    cd "$DEST_DIR"
    # Pull only the JSON metadata files (not the 13.2 GB video archive)
    git lfs pull --include="benign_data.json,harmful_data.json"
    cd ..
fi

echo "[DONE] Dataset downloaded to: $DEST_DIR"
echo "  benign_data.json  : $(wc -l < $DEST_DIR/benign_data.json) lines"
echo "  harmful_data.json : $(wc -l < $DEST_DIR/harmful_data.json) lines"
