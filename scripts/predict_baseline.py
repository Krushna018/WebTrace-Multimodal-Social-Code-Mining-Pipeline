#!/usr/bin/env python3
"""Predict with the TF‑IDF + Logistic‑Regression baseline model.

Usage:
    python scripts/predict_baseline.py --model-dir <path> "<text>"

Arguments:
    --model-dir   Directory where the saved pipeline (tfidf_logreg.joblib) resides.
    <text>        The input text to classify.

The script loads the joblib‑saved scikit‑learn pipeline, runs prediction
and probability, and prints a JSON object matching the output format of the
other inference scripts.
"""

import argparse
import json
import pathlib
import joblib

def main():
    parser = argparse.ArgumentParser(description="Baseline model inference")
    parser.add_argument("--model-dir", required=True, help="Directory containing tfidf_logreg.joblib")
    parser.add_argument("text", help="Text to classify")
    args = parser.parse_args()

    model_path = pathlib.Path(args.model_dir) / "tfidf_logreg.joblib"
    pipeline = joblib.load(model_path)

    pred = pipeline.predict([args.text])[0]
    prob = pipeline.predict_proba([args.text])[0][1]  # probability of class 1 (harmful)

    result = {"text": args.text, "harmful": int(pred), "p_harmful": float(prob)}
    print(json.dumps(result, ensure_ascii=False))

if __name__ == "__main__":
    main()
