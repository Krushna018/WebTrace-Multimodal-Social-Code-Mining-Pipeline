from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass
from typing import Optional
from urllib.parse import urlparse

import requests

GITHUB_URL_RE = re.compile(r"https?://(?:www\.)?github\.com/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", re.I)


@dataclass
class GitHubMetadata:
    repo_url: str
    full_name: str
    stars: int
    forks: int
    language: Optional[str]
    description: Optional[str]


def extract_github_repo_url(text: str) -> Optional[str]:
    match = GITHUB_URL_RE.search(text or "")
    if not match:
        return None
    return match.group(0).rstrip(".,);]")


def normalize_repo(repo_url: str) -> Optional[str]:
    parsed = urlparse(repo_url)
    parts = [p for p in parsed.path.split("/") if p]
    if len(parts) < 2:
        return None
    return f"{parts[0]}/{parts[1].removesuffix('.git')}"


def fetch_repo_metadata(repo_url: str, token: Optional[str] = None, timeout: int = 20) -> Optional[GitHubMetadata]:
    repo = normalize_repo(repo_url)
    if not repo:
        return None
    token = token or os.getenv("GITHUB_TOKEN")
    headers = {"Accept": "application/vnd.github+json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    r = requests.get(f"https://api.github.com/repos/{repo}", headers=headers, timeout=timeout)
    if r.status_code == 404:
        return None
    r.raise_for_status()
    data = r.json()
    return GitHubMetadata(
        repo_url=repo_url,
        full_name=data.get("full_name", repo),
        stars=int(data.get("stargazers_count", 0)),
        forks=int(data.get("forks_count", 0)),
        language=data.get("language"),
        description=data.get("description"),
    )
