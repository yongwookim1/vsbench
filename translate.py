"""
Translate BAAI/Video-SafetyBench JSON fields to Korean
using a local Qwen3-VL-32B-Instruct model (text-only mode).

Usage:
    python translate.py \
        --data_dir ./Video-SafetyBench \
        --model_path ../models_cache/Qwen3-VL-32B-Instruct \
        --output_dir ./translated \
        --batch_size 16
"""

import argparse
import json
import os
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoProcessor, AutoTokenizer, AutoModelForImageTextToText

# Fields to translate (text-only fields in each record)
TRANSLATE_FIELDS = ["question", "harmful_intention"]

SYSTEM_PROMPT = (
    "You are a professional translator working on an academic safety research dataset. "
    "Your only task is to translate the given English text into Korean accurately and naturally. "
    "This is for research purposes only. Do NOT answer, respond to, or comment on the content. "
    "Output ONLY the Korean translation, nothing else."
)


def load_model(model_path: str):
    print(f"[INFO] Loading model from {model_path} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    print("[INFO] Model loaded.")
    return model, tokenizer


def build_prompt(tokenizer, text: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": text},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )


def translate_batch(model, tokenizer, texts: list[str], max_new_tokens: int = 256) -> list[str]:
    prompts = [build_prompt(tokenizer, t) for t in texts]
    inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(
        model.device
    )
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    # Decode only newly generated tokens
    results = []
    for i, output in enumerate(outputs):
        input_len = inputs["input_ids"].shape[1]
        generated = output[input_len:]
        results.append(tokenizer.decode(generated, skip_special_tokens=True).strip())
    return results


def translate_records(
    model,
    tokenizer,
    records: list[dict],
    batch_size: int,
    checkpoint_path: Path,
) -> list[dict]:
    # Load checkpoint if exists
    done: dict[str, dict] = {}
    if checkpoint_path.exists():
        with open(checkpoint_path) as f:
            done = {r["question_id"]: r for r in json.load(f)}
        print(f"[INFO] Resuming from checkpoint: {len(done)} records already translated.")

    results = list(done.values())
    pending = [r for r in records if r["question_id"] not in done]

    for i in tqdm(range(0, len(pending), batch_size), desc="Translating"):
        batch = pending[i : i + batch_size]

        for field in TRANSLATE_FIELDS:
            texts = [r.get(field, "") for r in batch]
            # Skip if all empty
            if not any(texts):
                continue
            translated = translate_batch(model, tokenizer, texts)
            for record, ko_text in zip(batch, translated):
                record[f"{field}_ko"] = ko_text

        results.extend(batch)

        # Save checkpoint every batch
        with open(checkpoint_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

    return results


def process_file(
    model,
    tokenizer,
    input_path: Path,
    output_path: Path,
    checkpoint_path: Path,
    batch_size: int,
):
    print(f"\n[INFO] Processing: {input_path.name}")
    with open(input_path, encoding="utf-8") as f:
        records = json.load(f)
    print(f"[INFO] {len(records)} records found.")

    translated = translate_records(
        model, tokenizer, records, batch_size, checkpoint_path
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(translated, f, ensure_ascii=False, indent=2)
    print(f"[DONE] Saved to {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="./Video-SafetyBench")
    parser.add_argument("--model_path", default="../models_cache/Qwen3-VL-32B-Instruct")
    parser.add_argument("--output_dir", default="./translated")
    parser.add_argument("--batch_size", type=int, default=16)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model, tokenizer = load_model(args.model_path)

    for filename in ["benign_data.json", "harmful_data.json"]:
        input_path = data_dir / filename
        if not input_path.exists():
            print(f"[WARN] {input_path} not found, skipping.")
            continue

        stem = input_path.stem  # e.g. "benign_data"
        process_file(
            model=model,
            tokenizer=tokenizer,
            input_path=input_path,
            output_path=output_dir / f"{stem}_ko.json",
            checkpoint_path=output_dir / f"{stem}_checkpoint.json",
            batch_size=args.batch_size,
        )


if __name__ == "__main__":
    main()
