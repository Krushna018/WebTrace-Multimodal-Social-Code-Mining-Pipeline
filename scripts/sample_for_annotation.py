#!/usr/bin/env python3
from pathlib import Path
import argparse
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/processed/corpus.csv")
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--out", default="data/processed/annotation_sample.csv")
    args = ap.parse_args()
    df = pd.read_csv(args.data)
    groups = []
    if "code_share_type" in df.columns:
        per = max(1, args.n // max(1, df.code_share_type.nunique()))
        for _, g in df.groupby("code_share_type"):
            groups.append(g.sample(min(per, len(g)), random_state=42))
        sample = pd.concat(groups).drop_duplicates().head(args.n)
        if len(sample) < args.n and len(df) > len(sample):
            rem = df.drop(sample.index, errors="ignore").sample(min(args.n-len(sample), len(df)-len(sample)), random_state=43)
            sample = pd.concat([sample, rem])
    else:
        sample = df.sample(min(args.n, len(df)), random_state=42)
    sample = sample.copy()
    sample["manual_label_harmful"] = ""
    sample["review_notes"] = ""
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    sample.to_csv(args.out, index=False)
    print(f"wrote {len(sample):,} records for manual validation -> {args.out}")

if __name__ == "__main__":
    main()
