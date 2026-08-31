#!/usr/bin/env python3
from pathlib import Path
import argparse
import json
import pandas as pd
from sklearn.model_selection import train_test_split
from webtrace.models.transformer import train_transformer, TransformerConfig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/processed/corpus.csv")
    ap.add_argument("--output", default="results/transformer")
    ap.add_argument("--model", default="distilbert-base-uncased")
    ap.add_argument("--epochs", type=int, default=2)
    args = ap.parse_args()
    df = pd.read_csv(args.data)
    df = df[df.label_harmful.notna()].copy()
    df["label_harmful"] = df.label_harmful.astype(int)
    train, valid = train_test_split(df, test_size=0.2, stratify=df.label_harmful, random_state=42)
    valid = valid.iloc[:3000]   # keep exactly 3k validation posts
    metrics = train_transformer(train, valid, args.output, TransformerConfig(model_name=args.model, epochs=args.epochs))
    Path(args.output).mkdir(parents=True, exist_ok=True)
    (Path(args.output) / "eval_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))

if __name__ == "__main__":
    main()
