"""
Download lmms-lab/VideoChatGPT and convert to standard JSON format.

Two download methods for metadata:
  --method datasets  (default) — uses HuggingFace `datasets` library
  --method git-lfs             — clones the HF repo with git-lfs, then converts parquet

Steps:
  1. Download parquet files from HuggingFace.
  2. Normalise into {split}_data.json with fields:
       question_id, video_name, question, answer  (+ question_2 for consistency)
  3. Optionally download MP4s from YouTube via yt-dlp (strips "v_" prefix → YT ID).

Usage:
    # datasets library (default)
    python download_videochatgpt.py --output_dir ./data/videochatgpt

    # git-lfs clone
    python download_videochatgpt.py --output_dir ./data/videochatgpt --method git-lfs

    # With videos
    python download_videochatgpt.py --output_dir ./data/videochatgpt --download_videos
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path


HF_REPO = "https://huggingface.co/datasets/lmms-lab/VideoChatGPT"

CONFIGS = {
    "generic":     ("Generic",     "question"),
    "temporal":    ("Temporal",    "question"),
    "consistency": ("Consistency", "question_1"),   # question_1 → normalised "question"
}


# ── Parquet → JSON conversion (shared by both methods) ───────────────────────

def convert_parquet(parquet_path: Path, split_name: str, question_field: str) -> list[dict]:
    try:
        import pandas as pd
    except ImportError:
        print("[ERROR] `pandas` not installed. Run: pip install pandas pyarrow")
        sys.exit(1)

    df = pd.read_parquet(parquet_path)
    records = []
    for i, row in df.iterrows():
        record = {
            "question_id": f"vcgpt_{split_name}_{i:05d}",
            "video_name":  row["video_name"],
            "question":    row[question_field],
            "answer":      row["answer"],
        }
        if "question_2" in df.columns:
            record["question_2"] = row["question_2"]
        records.append(record)
    return records


# ── Method 1: HuggingFace datasets library ────────────────────────────────────

def download_via_datasets(output_dir: Path):
    try:
        from datasets import load_dataset
    except ImportError:
        print("[ERROR] `datasets` library not installed. Run: pip install datasets")
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    for split_name, (config_name, question_field) in CONFIGS.items():
        out_path = output_dir / f"{split_name}_data.json"
        if out_path.exists():
            print(f"[SKIP] {out_path} already exists.")
            continue

        print(f"[INFO] Loading {config_name} from HuggingFace...")
        ds = load_dataset("lmms-lab/VideoChatGPT", name=config_name, split="test")

        records = []
        for i, row in enumerate(ds):
            record = {
                "question_id": f"vcgpt_{split_name}_{i:05d}",
                "video_name":  row["video_name"],
                "question":    row[question_field],
                "answer":      row["answer"],
            }
            if "question_2" in row:
                record["question_2"] = row["question_2"]
            records.append(record)

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=False, indent=2)
        print(f"[DONE] {len(records)} records → {out_path}")


# ── Method 2: git-lfs clone ───────────────────────────────────────────────────

def download_via_gitlfs(output_dir: Path):
    for cmd in ["git", "git-lfs"]:
        if subprocess.run(["which", cmd], capture_output=True).returncode != 0:
            print(f"[ERROR] `{cmd}` is not installed.")
            sys.exit(1)

    clone_dir = output_dir / "_clone"

    if not clone_dir.exists():
        print(f"[INFO] Cloning {HF_REPO} with git-lfs...")
        subprocess.run(["git", "lfs", "install"], check=True)
        subprocess.run(["git", "clone", HF_REPO, str(clone_dir)], check=True)
    else:
        print(f"[SKIP] Clone already exists at {clone_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Convert each subset's parquet to JSON
    for split_name, (config_name, question_field) in CONFIGS.items():
        out_path = output_dir / f"{split_name}_data.json"
        if out_path.exists():
            print(f"[SKIP] {out_path} already exists.")
            continue

        # Layout: _clone/<ConfigName>/*.parquet
        parquet_files = list((clone_dir / config_name).glob("*.parquet"))
        if not parquet_files:
            print(f"[WARN] No parquet found for {config_name} in {clone_dir / config_name}")
            continue

        print(f"[INFO] Converting {config_name} ({len(parquet_files)} parquet file(s))...")
        records = []
        for pf in sorted(parquet_files):
            records.extend(convert_parquet(pf, split_name, question_field))

        for i, r in enumerate(records):
            r["question_id"] = f"vcgpt_{split_name}_{i:05d}"

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=False, indent=2)
        print(f"[DONE] {len(records)} records → {out_path}")

    # Extract videos.zip if present and videos/ not yet extracted
    videos_zip = clone_dir / "videos.zip"
    videos_dir = output_dir / "videos"
    if videos_zip.exists() and not videos_dir.is_dir():
        print(f"[INFO] Extracting {videos_zip} → {videos_dir} ...")
        import zipfile
        with zipfile.ZipFile(videos_zip) as zf:
            zf.extractall(output_dir)
        print(f"[DONE] Videos extracted to {videos_dir}")
    elif videos_dir.is_dir():
        print(f"[SKIP] Videos already extracted at {videos_dir}")


# ── Video download (yt-dlp) ───────────────────────────────────────────────────

def download_videos(output_dir: Path):
    videos_dir = output_dir / "videos"
    videos_dir.mkdir(exist_ok=True)

    video_names: set[str] = set()
    for json_file in output_dir.glob("*_data.json"):
        with open(json_file, encoding="utf-8") as f:
            for r in json.load(f):
                video_names.add(r["video_name"])

    print(f"[INFO] {len(video_names)} unique videos to download.")
    failed = []

    for video_name in sorted(video_names):
        out_path = videos_dir / f"{video_name}.mp4"
        if out_path.exists():
            continue

        # "v_p1QGn0IzfW0" → YouTube ID "p1QGn0IzfW0"
        yt_id = video_name[2:] if video_name.startswith("v_") else video_name
        url   = f"https://www.youtube.com/watch?v={yt_id}"

        result = subprocess.run(
            [
                "yt-dlp",
                "-f", "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
                "--merge-output-format", "mp4",
                "-o", str(out_path),
                "--quiet",
                "--no-warnings",
                url,
            ],
            capture_output=True,
        )
        if result.returncode != 0:
            print(f"[WARN] Failed: {video_name} ({url})")
            failed.append(video_name)

    print(f"[DONE] Videos downloaded. Failed: {len(failed)}/{len(video_names)}")
    if failed:
        fail_log = output_dir / "failed_videos.txt"
        fail_log.write_text("\n".join(failed))
        print(f"[INFO] Failed list → {fail_log}")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir",      default="./data/videochatgpt")
    parser.add_argument("--method",          default="datasets", choices=["datasets", "git-lfs"],
                        help="Download method: datasets (default) | git-lfs")
    parser.add_argument("--download_videos", action="store_true",
                        help="Also download MP4s from YouTube via yt-dlp")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)

    if args.method == "git-lfs":
        download_via_gitlfs(output_dir)
    else:
        download_via_datasets(output_dir)

    if args.download_videos:
        download_videos(output_dir)
    else:
        print("[INFO] Metadata only. Pass --download_videos to fetch MP4s.")


if __name__ == "__main__":
    main()
