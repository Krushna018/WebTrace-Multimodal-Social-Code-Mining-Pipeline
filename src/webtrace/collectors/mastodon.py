from __future__ import annotations

import html
import time
from datetime import datetime, timezone
from typing import Iterable, Optional

import requests
from bs4 import BeautifulSoup

from ..schema import PostRecord
from ..utils import stable_id
from .github import extract_github_repo_url


def strip_html(value: str) -> str:
    soup = BeautifulSoup(html.unescape(value or ""), "html.parser")
    return soup.get_text(" ", strip=True)


def collect_public_timeline(
    instance: str = "mastodon.social",
    limit: int = 100,
    max_pages: int = 10,
    sleep_seconds: float = 0.5,
) -> list[PostRecord]:
    """Collect public Mastodon statuses without authentication.

    Respect the target instance's terms, robots policies, and rate limits.
    The public endpoint can vary by instance.
    """
    base = f"https://{instance}/api/v1/timelines/public"
    records: list[PostRecord] = []
    max_id: Optional[str] = None

    for _ in range(max_pages):
        params = {"limit": min(limit, 20)}
        if max_id:
            params["max_id"] = max_id
        # --- Rate‑limit handling -------------------------------------------------
        while True:
            r = requests.get(base, params=params, timeout=30, headers={"User-Agent": "WebTraceResearch/0.1"})
            if r.status_code == 429:
                # Server asks us to back‑off. Respect `Retry-After` if present.
                retry_after = r.headers.get("Retry-After")
                wait = int(retry_after) if retry_after and retry_after.isdigit() else 60
                print(f"⚠️ 429 Too Many Requests – sleeping {wait}s before retry")
                time.sleep(wait)
                continue  # retry the same request
            r.raise_for_status()
            items = r.json()
            if not items:
                break
            # ---------------------------------------------------------------------
        for item in items:
            text = strip_html(item.get("content", ""))
            url = item.get("url") or item.get("uri") or ""
            record = PostRecord(
                post_id=str(item.get("id") or stable_id(instance, url, text)),
                platform=f"mastodon:{instance}",
                created_at=item.get("created_at") or datetime.now(timezone.utc).isoformat(),
                author=(item.get("account") or {}).get("acct", ""),
                text=text,
                url=url,
                repo_url=extract_github_repo_url(text),
            )
            records.append(record)
        max_id = str(items[-1].get("id"))
        if len(records) >= limit * max_pages:
            break
        time.sleep(sleep_seconds)
    return records
