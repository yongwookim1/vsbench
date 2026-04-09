"""
Evaluate GuardReasoner-VL results for a given dataset.

Merges per-GPU shard JSONL files produced by inference_guardreasoner.py,
saves merged JSONL, then reports F1 score per split × language.

Usage:
    python evaluate_guardreasoner.py --dataset video_safetybench \\
        --results_dir ./results/video_safetybench --num_gpus 4
"""

import argparse
import json
from pathlib import Path

from sklearn.metrics import f1_score

from adapters import get_adapter


def load_shards(results_dir: Path, split: str, lang: str, num_gpus: int) -> list[dict]:
    records = []
    for gpu_id in range(num_gpus):
        path = results_dir / f"{split}_{lang}_guardreasoner_gpu{gpu_id}.jsonl"
        if not path.exists():
            raise FileNotFoundError(f"Missing shard: {path}")
        with open(path, encoding="utf-8") as f:
            records.extend(json.loads(line) for line in f)
    return records


def cal_f1(records: list[dict], pos_label: str) -> float:
    labels   = [r["label"]   for r in records]
    predicts = [r["predict"] for r in records]
    return f1_score(labels, predicts, pos_label=pos_label) * 100


def fmt(v):
    return f"{v:>12.2f}" if v is not None else f"{'N/A':>12}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",     required=True, help="Dataset name (see adapters/)")
    parser.add_argument("--results_dir", required=True)
    parser.add_argument("--num_gpus",    type=int, default=4)
    args = parser.parse_args()

    adapter     = get_adapter(args.dataset)
    results_dir = Path(args.results_dir)
    splits      = adapter.splits
    langs       = adapter.langs
    pos_label   = adapter.pos_label

    header_cols = "".join(f"{s:>12}" for s in splits)
    print("=" * (10 + 12 * len(splits)))
    print(f" GuardReasoner-VL — {args.dataset}")
    print(f" F1 (pos_label={pos_label!r})")
    print("=" * (10 + 12 * len(splits)))
    print(f"{'':10}{header_cols}")
    print("-" * (10 + 12 * len(splits)))

    summary = {}
    for lang in langs:
        scores = []
        for split in splits:
            try:
                records = load_shards(results_dir, split, lang, args.num_gpus)

                merged_path = results_dir / f"{split}_{lang}_guardreasoner.jsonl"
                with open(merged_path, "w", encoding="utf-8") as f:
                    for item in records:
                        f.write(json.dumps(item, ensure_ascii=False) + "\n")

                scores.append(cal_f1(records, pos_label))

                for gpu_id in range(args.num_gpus):
                    (results_dir / f"{split}_{lang}_guardreasoner_gpu{gpu_id}.jsonl").unlink(missing_ok=True)
            except FileNotFoundError as e:
                print(f"[WARN] {e}")
                scores.append(None)

        summary[lang] = dict(zip(splits, scores))
        print(f"{lang.upper():10}" + "".join(fmt(s) for s in scores))

    print("=" * (10 + 12 * len(splits)))

    summary_path = results_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"[INFO] Summary saved to {summary_path}")


if __name__ == "__main__":
    main()
