"""
Translate BAAI/Video-SafetyBench JSON fields to Korean
using Google Translate via deep-translator (no API key required).

Usage:
    pip install deep-translator
    python translate_deepl.py \
        --data_dir ./Video-SafetyBench \
        --output_dir ./translated
"""

import argparse
import json
import time
from pathlib import Path

from deep_translator import GoogleTranslator
from tqdm import tqdm

TRANSLATE_FIELDS = ["question", "harmful_intention"]

translator = GoogleTranslator(source="en", target="ko")


def translate_text(text: str) -> str:
    try:
        return translator.translate(text)
    except Exception as e:
        print(f"[WARN] Translation failed: {e}")
        return ""


def translate_records(
    records: list[dict],
    checkpoint_path: Path,
) -> list[dict]:
    done: dict[str, dict] = {}
    if checkpoint_path.exists():
        with open(checkpoint_path) as f:
            done = {r["question_id"]: r for r in json.load(f)}
        print(f"[INFO] Resuming from checkpoint: {len(done)} records already translated.")

    results = list(done.values())
    pending = [r for r in records if r["question_id"] not in done]

    for record in tqdm(pending, desc="Translating"):
        for field in TRANSLATE_FIELDS:
            text = record.get(field, "")
            if text:
                record[f"{field}_ko"] = translate_text(text)
                time.sleep(0.1)  # rate limit 방지

        results.append(record)

        with open(checkpoint_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

    return results


def process_file(
    input_path: Path,
    output_path: Path,
    checkpoint_path: Path,
):
    print(f"\n[INFO] Processing: {input_path.name}")
    with open(input_path, encoding="utf-8") as f:
        records = json.load(f)
    print(f"[INFO] {len(records)} records found.")

    translated = translate_records(records, checkpoint_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(translated, f, ensure_ascii=False, indent=2)
    print(f"[DONE] Saved to {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="./Video-SafetyBench")
    parser.add_argument("--output_dir", default="./translated")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for filename in ["benign_data.json", "harmful_data.json"]:
        input_path = data_dir / filename
        if not input_path.exists():
            print(f"[WARN] {input_path} not found, skipping.")
            continue

        stem = input_path.stem
        process_file(
            input_path=input_path,
            output_path=output_dir / f"{stem}_ko.json",
            checkpoint_path=output_dir / f"{stem}_checkpoint.json",
        )


if __name__ == "__main__":
    main()
