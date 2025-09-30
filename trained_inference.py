"""Generate answers with the fine-tuned adapter for comparison against the baseline."""
import argparse
import os
from typing import List, Dict

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel

from train_fn import _read_json_any, PROMPT_TPL, answer_question


def _select_samples(rows: List[Dict], title_key: str, content_key: str, limit: int) -> List[Dict]:
    picked = []
    for row in rows:
        if title_key in row and content_key in row:
            title = str(row[title_key]).strip()
            content = str(row[content_key]).strip()
            if title and content:
                picked.append({"title": title, "content": content})
        if len(picked) >= limit:
            break
    return picked


def main():
    parser = argparse.ArgumentParser(description="Run generations with a fine-tuned LoRA adapter.")
    parser.add_argument("--data_path", default="./trn.clean.jsonl", help="Path to the dataset used for training.")
    parser.add_argument("--model_name", default="google/flan-t5-base", help="Foundation model identifier.")
    parser.add_argument("--adapter_dir", default="./models/flan_t5_a10_lora", help="Directory containing the LoRA adapter.")
    parser.add_argument("--title_key", default="title", help="Key for the title field in the dataset.")
    parser.add_argument("--content_key", default="content", help="Key for the target field in the dataset.")
    parser.add_argument("--limit", type=int, default=3, help="Number of samples to generate.")
    parser.add_argument("--use_merged", action="store_true", help="Load the merged full model instead of the adapter.")
    args = parser.parse_args()

    if not os.path.exists(args.data_path):
        raise FileNotFoundError(f"Data file not found: {args.data_path}")
    if not os.path.exists(args.adapter_dir):
        raise FileNotFoundError(f"Adapter/model directory not found: {args.adapter_dir}")

    rows = _read_json_any(args.data_path)
    samples = _select_samples(rows, args.title_key, args.content_key, args.limit)
    if not samples:
        raise ValueError("No valid samples found in the dataset with the specified keys.")

    print(f"[load] base={args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    if args.use_merged:
        model_path = os.path.join(args.adapter_dir, "merged")
        if not os.path.exists(model_path):
            raise FileNotFoundError("Merged model not found inside adapter directory.")
        model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
        print(f"[load] merged model from {model_path}")
    else:
        base = AutoModelForSeq2SeqLM.from_pretrained(args.model_name)
        model = PeftModel.from_pretrained(base, args.adapter_dir)
        print(f"[load] adapter from {args.adapter_dir}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    print(f"[device] Using {device}")
    print(f"[samples] Generating {len(samples)} fine-tuned predictions\n")

    for idx, sample in enumerate(samples, start=1):
        prompt = PROMPT_TPL.format(title=sample["title"])
        prediction = answer_question(model, tokenizer, prompt)
        print(f"=== Sample {idx} ===")
        print("Prompt:", prompt)
        print("Target:", sample["content"])
        print("Fine-tuned:", prediction)
        print()


if __name__ == "__main__":
    main()
