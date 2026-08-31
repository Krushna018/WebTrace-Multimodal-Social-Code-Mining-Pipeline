#!/usr/bin/env python3
from pathlib import Path
import argparse
import pandas as pd
from webtrace.collectors.mastodon import collect_public_timeline
from webtrace.taxonomy import classify_code_sharing


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instance", default="mastodon.social")
    ap.add_argument("--pages", type=int, default=5)
    ap.add_argument("--out", default="data/raw/mastodon_posts.csv")
    args = ap.parse_args()
    records = collect_public_timeline(args.instance, limit=40, max_pages=args.pages)
    rows = []
    for rec in records:
        rec.code_share_type = classify_code_sharing(rec.text, rec.repo_url, rec.image_ocr_text)
        rows.append(rec.to_dict())
    df = pd.DataFrame(rows)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"wrote {len(df):,} public posts to {args.out}")

if __name__ == "__main__":
    main()
