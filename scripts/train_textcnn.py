#!/usr/bin/env python3
from pathlib import Path
import argparse
import json
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

from webtrace.models.textcnn import TextCNNConfig, train_textcnn, predict_textcnn


def main():
    ap = argparse.ArgumentParser(description="Train a fully local PyTorch TextCNN harmful-content classifier.")
    ap.add_argument("--data", default="data/processed/corpus.csv")
    ap.add_argument("--output", default="results/textcnn")
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--batch-size", type=int, default=32)
    args = ap.parse_args()

    df = pd.read_csv(args.data)
    df = df[df.label_harmful.notna()].copy()
    df["label_harmful"] = df.label_harmful.astype(int)
    train, valid = train_test_split(
        df, test_size=0.2, stratify=df.label_harmful, random_state=42
    )
    valid = valid.iloc[:3000]   # keep exactly 3k validation posts
    config = TextCNNConfig(epochs=args.epochs, batch_size=args.batch_size)
    model, vocab, metrics = train_textcnn(train, valid, args.output, config)
    preds, probs = predict_textcnn(model, vocab, config, valid.text.astype(str).tolist())

    out = Path(args.output)
    report = classification_report(valid.label_harmful, preds, output_dict=True, zero_division=0)
    cm = confusion_matrix(valid.label_harmful, preds).tolist()
    (out / "eval_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (out / "classification_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (out / "confusion_matrix.json").write_text(json.dumps(cm, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
