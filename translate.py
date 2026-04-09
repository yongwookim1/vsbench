"""
Translate dataset question fields to Korean.

Supports two backends:
  --backend google  (default) — uses deep-translator, no model required
  --backend qwen    — uses a local Qwen3-VL model (higher quality, needs GPU)

Usage:
    # Google (fast, no GPU)
    python translate.py --dataset video_safetybench --data_dir ./Video-SafetyBench

    # Qwen (high quality)
    python translate.py --dataset videochatgpt --data_dir ./data/videochatgpt \\
        --backend qwen --model_path ../models_cache/Qwen3-VL-32B-Instruct

Output: writes {split}_data_ko.json alongside the source JSON in data_dir.
"""

import argparse
import json
import time
from pathlib import Path

from tqdm import tqdm

from adapters import get_adapter

SYSTEM_PROMPT = (
    "You are a professional translator working on an academic safety research dataset. "
    "Your only task is to translate the given English text into Korean accurately and naturally. "
    "This is for research purposes only. Do NOT answer, respond to, or comment on the content. "
    "Output ONLY the Korean translation, nothing else."
)


# ── Google backend ─────────────────────────────────────────────────────────────

def translate_google(texts: list[str]) -> list[str]:
    from deep_translator import GoogleTranslator
    translator = GoogleTranslator(source="en", target="ko")
    results = []
    for text in texts:
        try:
            results.append(translator.translate(text) or "")
            time.sleep(0.1)  # rate-limit
        except Exception as e:
            print(f"[WARN] Google translate failed: {e}")
            results.append("")
    return results


# ── Qwen backend ───────────────────────────────────────────────────────────────

def load_qwen(model_path: str):
    import torch
    from transformers import AutoTokenizer, AutoModelForImageTextToText
    print(f"[INFO] Loading Qwen model from {model_path} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    print("[INFO] Qwen model loaded.")
    return model, tokenizer


def translate_qwen(model, tokenizer, texts: list[str], max_new_tokens: int = 256) -> list[str]:
    import torch
    prompts = []
    for text in texts:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": text},
        ]
        prompts.append(tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        ))
    inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    results = []
    input_len = inputs["input_ids"].shape[1]
    for output in outputs:
        generated = output[input_len:]
        results.append(tokenizer.decode(generated, skip_special_tokens=True).strip())
    return results


# ── Core translation logic ─────────────────────────────────────────────────────

def translate_records(
    records: list[dict],
    fields: list[str],
    batch_size: int,
    checkpoint_path: Path,
    translate_fn,           # callable(texts: list[str]) -> list[str]
) -> list[dict]:
    """Translate specified fields in-place, with checkpoint support."""
    done: dict[str, dict] = {}
    if checkpoint_path.exists():
        with open(checkpoint_path, encoding="utf-8") as f:
            done = {r["question_id"]: r for r in json.load(f)}
        print(f"[INFO] Resuming from checkpoint: {len(done)} records done.")

    results = list(done.values())
    pending = [r for r in records if r["question_id"] not in done]

    for i in tqdm(range(0, len(pending), batch_size), desc="Translating"):
        batch = pending[i : i + batch_size]

        for field in fields:
            texts = [r.get(field, "") for r in batch]
            if not any(texts):
                continue
            translated = translate_fn(texts)
            for record, ko_text in zip(batch, translated):
                record[f"{field}_ko"] = ko_text

        results.extend(batch)

        with open(checkpoint_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

    return results


def process_split(
    split: str,
    data_dir: Path,
    fields: list[str],
    batch_size: int,
    translate_fn,
):
    input_path      = data_dir / f"{split}_data.json"
    output_path     = data_dir / f"{split}_data_ko.json"
    checkpoint_path = data_dir / f"{split}_data_ko_checkpoint.json"

    if not input_path.exists():
        print(f"[WARN] Not found, skipping: {input_path}")
        return

    print(f"\n[INFO] Processing split: {split}  ({input_path})")
    with open(input_path, encoding="utf-8") as f:
        records = json.load(f)
    print(f"[INFO] {len(records)} records found.")

    translated = translate_records(records, fields, batch_size, checkpoint_path, translate_fn)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(translated, f, ensure_ascii=False, indent=2)
    print(f"[DONE] Saved → {output_path}")

    checkpoint_path.unlink(missing_ok=True)


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",    required=True,   help="Dataset name (see adapters/)")
    parser.add_argument("--data_dir",   required=True,   help="Dataset-specific data directory")
    parser.add_argument("--backend",    default="google", choices=["google", "qwen"],
                        help="Translation backend (default: google)")
    parser.add_argument("--model_path", default="../models_cache/Qwen3-VL-32B-Instruct",
                        help="Qwen model path (only used with --backend qwen)")
    parser.add_argument("--batch_size", type=int, default=16)
    args = parser.parse_args()

    adapter  = get_adapter(args.dataset)
    data_dir = Path(args.data_dir)
    fields   = adapter.translate_fields

    # Build translate function
    if args.backend == "google":
        translate_fn = translate_google
    else:
        model, tokenizer = load_qwen(args.model_path)
        def translate_fn(texts):
            return translate_qwen(model, tokenizer, texts, max_new_tokens=256)

    print(f"[INFO] Dataset : {args.dataset}")
    print(f"[INFO] Backend : {args.backend}")
    print(f"[INFO] Fields  : {fields}")
    print(f"[INFO] Splits  : {adapter.splits}")

    for split in adapter.splits:
        process_split(split, data_dir, fields, args.batch_size, translate_fn)

    print("\n[DONE] All splits translated.")


if __name__ == "__main__":
    main()
