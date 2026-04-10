import json
from pathlib import Path

from .base import DatasetAdapter


class VideoChatGPTAdapter(DatasetAdapter):
    """
    Adapter for lmms-lab/VideoChatGPT.

    Expected data_dir layout (produced by download_videochatgpt.py):
        data_dir/
          generic_data.json        # English
          temporal_data.json       # English
          consistency_data.json    # English  (question_1 normalised → question)
          generic_data_ko.json     # Korean   (from translate.py)
          temporal_data_ko.json    # Korean
          consistency_data_ko.json # Korean
          videos/                  # MP4s downloaded via yt-dlp
            v_p1QGn0IzfW0.mp4
            ...

    All records are benign VQA content → label = "unharmful".
    F1 with pos_label="unharmful" measures false-positive rate.
    """

    name = "videochatgpt"
    splits = ["generic", "temporal", "consistency"]
    langs = ["en", "ko"]
    pos_label = "unharmful"
    translate_fields = ["question"]

    _label = "unharmful"

    def is_downloaded(self, data_dir: Path) -> bool:
        return all((data_dir / f"{s}_data.json").exists() for s in self.splits)

    def is_translated(self, data_dir: Path) -> bool:
        return all((data_dir / f"{s}_data_ko.json").exists() for s in self.splits)

    def _videos_dir(self, data_dir: Path) -> Path:
        for candidate in ("Test_Videos", "videos"):
            d = data_dir / candidate
            if d.is_dir():
                return d
        return data_dir / "videos"

    def is_videos_ready(self, data_dir: Path) -> bool:
        videos_dir = self._videos_dir(data_dir)
        return videos_dir.is_dir() and any(videos_dir.glob("*.mp4"))

    def get_records(self, data_dir: Path, split: str, lang: str) -> list[dict]:
        suffix = "_ko" if lang == "ko" else ""
        json_path = data_dir / f"{split}_data{suffix}.json"
        with open(json_path, encoding="utf-8") as f:
            raw = json.load(f)

        question_key = "question_ko" if lang == "ko" else "question"
        records = []
        for r in raw:
            video_name = r["video_name"]
            video_path = str((self._videos_dir(data_dir) / video_name).resolve()) + ".mp4"
            records.append({
                "question_id": r["question_id"],
                "video_path":  video_path,
                "question":    r.get(question_key) or r.get("question", ""),
                "label":       self._label,
            })
        return records
