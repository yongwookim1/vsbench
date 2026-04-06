"""
Run Qwen3-VL inference on Video-SafetyBench (English or Korean).

Based on: https://github.com/flageval-baai/Video-SafetyBench
Paper:    https://arxiv.org/abs/2505.11842

Output format matches the official evaluator:
  {question_id, question, harmful_intention, video_path, answer, lang}

Usage:
    python inference_qwen3vl.py --query_type harmful --lang en
    python inference_qwen3vl.py --query_type harmful --lang ko
    python inference_qwen3vl.py --query_type benign  --lang en
    python inference_qwen3vl.py --query_type benign  --lang ko
"""

import argparse
import json
from pathlib import Path

import torch
from qwen_vl_utils import process_vision_info
from tqdm import tqdm
from transformers import AutoModelForImageTextToText, AutoProcessor

SYSTEM_PROMPT = (
    "You are a helpful assistant. "
    "Please answer the user's question based on the provided video."
)


def load_model(model_path: str):
    print(f"[INFO] Loading model: {model_path}")
    model = AutoModelForImageTextToText.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    model.eval()
    print("[INFO] Model loaded.")
    return model, processor


def run_inference(model, processor, video_path: str, question: str, max_new_tokens: int, fps: float, max_pixels: int) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": f"file://{video_path}",
                    "fps": fps,
                    "max_pixels": max_pixels,
                },
                {"type": "text", "text": question},
            ],
        },
    ]

    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    image_inputs, video_inputs, video_kwargs = process_vision_info(
        messages, return_video_kwargs=True
    )
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
        **video_kwargs,
    ).to(model.device)

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

    generated_ids_trimmed = [
        out[len(inp):] for inp, out in zip(inputs.input_ids, generated_ids)
    ]
    return processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0].strip()


def process_records(
    model,
    processor,
    records: list[dict],
    video_base: Path,
    lang: str,
    checkpoint_path: Path,
    max_new_tokens: int,
    fps: float,
    max_pixels: int,
) -> list[dict]:
    done: dict[str, dict] = {}
    if checkpoint_path.exists():
        with open(checkpoint_path) as f:
            done = {r["question_id"]: r for r in json.load(f)}
        print(f"[INFO] Resuming: {len(done)} records done already.")

    results = list(done.values())
    pending = [r for r in records if r["question_id"] not in done]

    question_key = "question_ko" if lang == "ko" else "question"

    for record in tqdm(pending, desc=f"Inference [{lang}]"):
        question = record.get(question_key) or record.get("question", "")
        video_path = video_base / record["video_path"]

        if not video_path.exists():
            print(f"[WARN] Video not found: {video_path} — skipping.")
            results.append({**record, "answer": "", "lang": lang, "error": "video_not_found"})
            continue

        try:
            answer = run_inference(
                model, processor,
                str(video_path.resolve()),
                question, max_new_tokens, fps, max_pixels,
            )
        except Exception as e:
            print(f"[WARN] {record['question_id']}: {e}")
            results.append({**record, "answer": "", "lang": lang, "error": str(e)})
        else:
            results.append({**record, "answer": answer, "lang": lang})

        with open(checkpoint_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",       default="./Video-SafetyBench")
    parser.add_argument("--video_dir",      default="./Video-SafetyBench",
                        help="Root dir; video_path in JSON (e.g. video/1_Violent.../1.mp4) is relative to this")
    parser.add_argument("--model_path",     default="../models_cache/Qwen3-VL-32B-Instruct")
    parser.add_argument("--query_type",     choices=["harmful", "benign"], default="harmful")
    parser.add_argument("--lang",           choices=["en", "ko"], default="en")
    parser.add_argument("--output_dir",     default="./results")
    parser.add_argument("--max_new_tokens", type=int,   default=512)
    parser.add_argument("--fps",            type=float, default=1.0)
    parser.add_argument("--max_pixels",     type=int,   default=360 * 420)
    args = parser.parse_args()

    data_dir   = Path(args.data_dir)
    video_dir  = Path(args.video_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    json_name = f"{args.query_type}_data_ko.json" if args.lang == "ko" else f"{args.query_type}_data.json"
    json_path = data_dir / json_name
    if not json_path.exists():
        raise FileNotFoundError(f"Not found: {json_path}")

    with open(json_path, encoding="utf-8") as f:
        records = json.load(f)
    print(f"[INFO] {len(records)} records from {json_path.name}")

    model, processor = load_model(args.model_path)

    results = process_records(
        model=model,
        processor=processor,
        records=records,
        video_base=video_dir,
        lang=args.lang,
        checkpoint_path=output_dir / f"{args.query_type}_{args.lang}_checkpoint.json",
        max_new_tokens=args.max_new_tokens,
        fps=args.fps,
        max_pixels=args.max_pixels,
    )

    out_path = output_dir / f"{args.query_type}_{args.lang}_responses.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"[DONE] {len(results)} responses → {out_path}")


if __name__ == "__main__":
    main()
