#!/usr/bin/env python3
from pathlib import Path
import argparse
from webtrace.synthetic import generate_demo
from webtrace.taxonomy import classify_code_sharing


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=5000)
    ap.add_argument("--out", default="data/raw/demo_posts.csv")
    args = ap.parse_args()
    df = generate_demo(args.n)
    df["code_share_type"] = [classify_code_sharing(t, r) for t, r in zip(df.text, df.repo_url)]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"wrote {len(df):,} demo records to {args.out}")

if __name__ == "__main__":
    main()
