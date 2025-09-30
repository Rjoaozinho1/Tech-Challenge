"""Quick baseline inference with the foundation model before fine-tuning."""
import argparse
import os
from typing import List, Dict

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from train_fn import _read_json_any, PROMPT_TPL, answer_question


def _select_samples(rows: List[Dict], title_key: str, content_key: str, limit: int) -> List[Dict]:
    """Pick up to `limit` rows that include the required keys."""
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
    parser = argparse.ArgumentParser(description="Run baseline generations before training.")
    parser.add_argument("--data_path", default="./trn.clean.jsonl", help="Path to the training data (JSON/JSONL).")
    parser.add_argument("--model_name", default="google/flan-t5-base", help="Name or path of the Hugging Face model.")
    parser.add_argument("--title_key", default="title", help="Key containing the title field in the dataset.")
    parser.add_argument("--content_key", default="content", help="Key containing the description field in the dataset.")
    parser.add_argument("--limit", type=int, default=3, help="Number of samples to generate.")
    args = parser.parse_args()

    if not os.path.exists(args.data_path):
        raise FileNotFoundError(f"Data file not found: {args.data_path}")

    rows = _read_json_any(args.data_path)
    samples = _select_samples(rows, args.title_key, args.content_key, args.limit)
    if not samples:
        raise ValueError("No valid samples found in the dataset with the specified keys.")

    print(f"[load] model={args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    print(f"[device] Using {device}")
    print(f"[samples] Generating {len(samples)} baseline predictions\n")

    for idx, sample in enumerate(samples, start=1):
        prompt = PROMPT_TPL.format(title=sample["title"])
        prediction = answer_question(model, tokenizer, prompt)
        print(f"=== Sample {idx} ===")
        print("Prompt:", prompt)
        print("Target:", sample["content"])
        print("Baseline:", prediction)
        print()


if __name__ == "__main__":
    main()
