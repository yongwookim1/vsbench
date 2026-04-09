import json
from pathlib import Path

from .base import DatasetAdapter


class VideoSafetyBenchAdapter(DatasetAdapter):
    """
    Adapter for BAAI/Video-SafetyBench.

    Expected data_dir layout:
        data_dir/
          harmful_data.json        # English harmful split
          benign_data.json         # English benign split
          harmful_data_ko.json     # Korean harmful split  (from translate.py)
          benign_data_ko.json      # Korean benign split   (from translate.py)
          video/                   # extracted from video.tar.gz
    """

    name = "video_safetybench"
    splits = ["harmful", "benign"]
    langs = ["en", "ko"]
    pos_label = "harmful"
    translate_fields = ["question", "harmful_intention"]

    # Both splits contain genuinely harmful content (explicit vs implicit).
    _label = "harmful"

    def is_downloaded(self, data_dir: Path) -> bool:
        return (
            (data_dir / "harmful_data.json").exists()
            and (data_dir / "benign_data.json").exists()
        )

    def is_translated(self, data_dir: Path) -> bool:
        return (
            (data_dir / "harmful_data_ko.json").exists()
            and (data_dir / "benign_data_ko.json").exists()
        )

    def is_videos_ready(self, data_dir: Path) -> bool:
        return (data_dir / "video").is_dir()

    def get_records(self, data_dir: Path, split: str, lang: str) -> list[dict]:
        suffix = "_ko" if lang == "ko" else ""
        json_path = data_dir / f"{split}_data{suffix}.json"
        with open(json_path, encoding="utf-8") as f:
            raw = json.load(f)

        question_key = "question_ko" if lang == "ko" else "question"
        records = []
        for r in raw:
            records.append({
                "question_id": r["question_id"],
                "video_path":  str((data_dir / r["video_path"]).resolve()),
                "question":    r.get(question_key) or r.get("question", ""),
                "label":       self._label,
            })
        return records
