#!/usr/bin/env python3
from pathlib import Path
import argparse
import pandas as pd
from webtrace.preprocess import normalize_text, extract_code_snippets
from webtrace.collectors.github import extract_github_repo_url
from webtrace.taxonomy import classify_code_sharing


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+")
    ap.add_argument("--out", default="data/processed/corpus.csv")
    args = ap.parse_args()

    frames = [pd.read_csv(p) for p in args.inputs]
    df = pd.concat(frames, ignore_index=True)
    df["text"] = df["text"].fillna("").map(normalize_text)
    if "repo_url" not in df.columns:
        df["repo_url"] = ""
    df["repo_url"] = [r if isinstance(r, str) and r else (extract_github_repo_url(t) or "") for r, t in zip(df.repo_url, df.text)]
    df["code_snippet"] = df.text.map(lambda x: "\n---\n".join(extract_code_snippets(x)))
    if "image_ocr_text" not in df.columns:
        df["image_ocr_text"] = ""
    else:
        df["image_ocr_text"] = df["image_ocr_text"].fillna("")
    df["code_share_type"] = [classify_code_sharing(t, r or None, o or None) for t, r, o in zip(df.text, df.repo_url, df.image_ocr_text)]
    df = df.drop_duplicates(subset=["platform", "text"]).reset_index(drop=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(df.code_share_type.value_counts(dropna=False))
    print(f"prepared {len(df):,} unique records -> {args.out}")

if __name__ == "__main__":
    main()
