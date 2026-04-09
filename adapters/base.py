from abc import ABC, abstractmethod
from pathlib import Path


class DatasetAdapter(ABC):
    """
    Abstract base for dataset adapters.

    Each adapter normalises a dataset into the standard internal record format:
        {
            "question_id": str,
            "video_path":  str,   # absolute path to the video file
            "question":    str,   # in the requested language
            "label":       str,   # "harmful" | "unharmful"
        }
    """

    # ── class-level attributes to override ─────────────────────────────────────
    name: str               # slug used for directories / filenames
    splits: list[str]       # ordered split names (e.g. ["harmful", "benign"])
    langs: list[str]        # language codes (e.g. ["en", "ko"])
    pos_label: str          # positive class for F1 ("harmful" | "unharmful")
    translate_fields: list[str]  # JSON fields that need KO translation

    # ── readiness checks (pipeline skip logic) ─────────────────────────────────

    @abstractmethod
    def is_downloaded(self, data_dir: Path) -> bool:
        """Return True if the required JSON files exist in data_dir."""
        ...

    @abstractmethod
    def is_translated(self, data_dir: Path) -> bool:
        """Return True if all *_ko.json files exist in data_dir."""
        ...

    def is_videos_ready(self, data_dir: Path) -> bool:
        """Return True if videos are present. Override for datasets with videos."""
        return True

    # ── record access ──────────────────────────────────────────────────────────

    @abstractmethod
    def get_records(self, data_dir: Path, split: str, lang: str) -> list[dict]:
        """
        Return list of normalised records for the given split and language.
        video_path must be an absolute path string.
        """
        ...
