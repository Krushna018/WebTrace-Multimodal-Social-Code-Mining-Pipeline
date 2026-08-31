from __future__ import annotations

import json
from pathlib import Path
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support, accuracy_score


def evaluate_binary(y_true, y_pred) -> dict:
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(p),
        "recall_macro": float(r),
        "f1_macro": float(f1),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "classification_report": classification_report(y_true, y_pred, zero_division=0, output_dict=True),
    }


def error_analysis(df: pd.DataFrame, y_pred, out_csv: str | Path) -> pd.DataFrame:
    out = df.copy()
    out["prediction"] = y_pred
    errors = out[out["prediction"] != out["label_harmful"]].copy()
    cols = [c for c in ["post_id", "platform", "text", "repo_url", "code_share_type", "label_harmful", "prediction"] if c in errors.columns]
    errors[cols].to_csv(out_csv, index=False)
    return errors[cols]
