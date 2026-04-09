"""
Download lmms-lab/VideoChatGPT and convert to standard JSON format.

Steps:
  1. Pull parquet files from HuggingFace via the `datasets` library.
  2. Normalise into {split}_data.json with fields:
       question_id, video_name, question, answer  (+ question_2 for consistency)
  3. Optionally download MP4s from YouTube via yt-dlp (strips "v_" prefix → YT ID).

Usage:
    # Metadata only (no videos)
    python download_videochatgpt.py --output_dir ./data/videochatgpt

    # With videos (~may take a long time)
    python download_videochatgpt.py --output_dir ./data/videochatgpt --download_videos
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path


CONFIGS = {
    "generic":     ("Generic",     "question"),
    "temporal":    ("Temporal",    "question"),
    "consistency": ("Consistency", "question_1"),   # question_1 → normalised "question"
}


def download_metadata(output_dir: Path):
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
            # Keep question_2 for the consistency subset (useful for future eval)
            if "question_2" in row:
                record["question_2"] = row["question_2"]
            records.append(record)

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=False, indent=2)
        print(f"[DONE] {len(records)} records → {out_path}")


def download_videos(output_dir: Path):
    videos_dir = output_dir / "videos"
    videos_dir.mkdir(exist_ok=True)

    # Collect unique video names across all splits
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir",      default="./data/videochatgpt")
    parser.add_argument("--download_videos", action="store_true",
                        help="Also download MP4s from YouTube via yt-dlp")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    download_metadata(output_dir)

    if args.download_videos:
        download_videos(output_dir)
    else:
        print("[INFO] Metadata only. Pass --download_videos to fetch MP4s.")


if __name__ == "__main__":
    main()
