#!/usr/bin/env python3
from pathlib import Path
import argparse
import json
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from webtrace.models.baseline import BaselineModel
from webtrace.evaluation import evaluate_binary, error_analysis


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/processed/corpus.csv")
    ap.add_argument("--results", default="results")
    args = ap.parse_args()

    df = pd.read_csv(args.data)
    df = df[df["label_harmful"].notna()].copy()
    df["label_harmful"] = df["label_harmful"].astype(int)
    train, test = train_test_split(df, test_size=0.25, stratify=df.label_harmful, random_state=42)
    test = test.iloc[:3000]   # keep exactly 3k validation posts
    model = BaselineModel.create().fit(train.text, train.label_harmful)
    pred = model.predict(test.text)
    metrics = evaluate_binary(test.label_harmful, pred)

    out = Path(args.results); out.mkdir(parents=True, exist_ok=True)
    joblib.dump(model.pipeline, out / "tfidf_logreg.joblib")
    (out / "baseline_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    error_analysis(test, pred, out / "baseline_errors.csv")
    print(json.dumps({k: v for k, v in metrics.items() if k != "classification_report"}, indent=2))

if __name__ == "__main__":
    main()
