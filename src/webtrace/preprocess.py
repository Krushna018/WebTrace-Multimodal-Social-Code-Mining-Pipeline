from __future__ import annotations

import re
import unicodedata

URL_RE = re.compile(r"https?://\S+")
WS_RE = re.compile(r"\s+")
CODE_FENCE_RE = re.compile(r"```(?:\w+)?\s*(.*?)```", re.S)
INLINE_CODE_RE = re.compile(r"`([^`]{2,})`")


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text or "")
    text = WS_RE.sub(" ", text).strip()
    return text


def text_without_urls(text: str) -> str:
    return normalize_text(URL_RE.sub(" ", text or ""))


def extract_code_snippets(text: str) -> list[str]:
    text = text or ""
    snippets = [m.strip() for m in CODE_FENCE_RE.findall(text)]
    without_fences = CODE_FENCE_RE.sub(" ", text)
    snippets.extend(m.strip() for m in INLINE_CODE_RE.findall(without_fences))
    return [s for s in snippets if s]


def looks_like_code(text: str) -> bool:
    if not text:
        return False
    indicators = ["def ", "class ", "function ", "=>", "{", "}", ";", "import ", "const ", "let ", "var ", "println(", "print("]
    hits = sum(1 for token in indicators if token in text)
    return hits >= 2
