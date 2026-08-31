#!/usr/bin/env python3
"""Utility script to run inference with a fine‑tuned transformer model.

Usage:
    python scripts/predict_transformer.py --model-dir <path> "<text>"

Arguments:
    --model-dir   Directory where the fine‑tuned model and tokenizer are saved
    <text>        The input text to classify (harmful vs non‑harmful)

The script loads the model and tokenizer, tokenises the input, runs a forward pass,
and prints a JSON‑like dict with the raw text, the predicted label (0 = non‑harmful,
1 = harmful) and the probability of the harmful class.
"""

import argparse
import json
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def main():
    parser = argparse.ArgumentParser(description="Predict with a fine‑tuned transformer model")
    parser.add_argument("--model-dir", required=True, help="Path to the saved model directory")
    parser.add_argument("text", help="Text to classify")
    args = parser.parse_args()

    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_dir)
    model.eval()
    device = torch.device("cpu")
    model.to(device)

    # Tokenise input
    inputs = tokenizer(args.text, return_tensors="pt", truncation=True, max_length=192)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=-1).squeeze().cpu().numpy()
        pred = int(probs.argmax())
        p_harmful = float(probs[1])  # probability of label 1 (harmful)

    result = {
        "text": args.text,
        "harmful": pred,
        "p_harmful": p_harmful,
    }
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
