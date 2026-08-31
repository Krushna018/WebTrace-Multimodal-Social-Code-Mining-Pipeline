from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Optional


@dataclass
class PostRecord:
    post_id: str
    platform: str
    created_at: str
    author: str
    text: str
    url: str
    repo_url: Optional[str] = None
    code_snippet: Optional[str] = None
    image_path: Optional[str] = None
    image_ocr_text: Optional[str] = None
    label_harmful: Optional[int] = None
    code_share_type: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)
