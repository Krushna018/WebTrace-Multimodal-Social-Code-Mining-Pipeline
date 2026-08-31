from __future__ import annotations

from pathlib import Path
import pandas as pd

from ..schema import PostRecord
from ..utils import stable_id
from .github import extract_github_repo_url


def import_csv(path: str | Path, platform: str = "csv") -> list[PostRecord]:
    df = pd.read_csv(path)
    text_col = "text" if "text" in df.columns else df.columns[0]
    records: list[PostRecord] = []
    for i, row in df.iterrows():
        text = str(row.get(text_col, ""))
        url = str(row.get("url", "")) if "url" in df.columns else ""
        records.append(PostRecord(
            post_id=str(row.get("post_id", stable_id(platform, str(i), text))),
            platform=str(row.get("platform", platform)),
            created_at=str(row.get("created_at", "")),
            author=str(row.get("author", "")),
            text=text,
            url=url,
            repo_url=str(row.get("repo_url", "")) or extract_github_repo_url(text),
            code_snippet=str(row.get("code_snippet", "")) or None,
            image_path=str(row.get("image_path", "")) or None,
            label_harmful=int(row["label_harmful"]) if "label_harmful" in df.columns and pd.notna(row["label_harmful"]) else None,
            code_share_type=str(row.get("code_share_type", "")) or None,
        ))
    return records
