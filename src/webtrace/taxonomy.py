from __future__ import annotations

from .preprocess import extract_code_snippets

TAXONOMY = {
    "repository_link": "Post links to a source-code repository.",
    "inline_code": "Post includes fenced or inline source code.",
    "image_code": "Post includes an image whose OCR output resembles source code.",
    "code_discussion": "Post discusses code/programming without directly sharing code.",
    "none": "No clear code-sharing signal.",
}


def classify_code_sharing(text: str, repo_url: str | None = None, image_ocr_text: str | None = None) -> str:
    if repo_url:
        return "repository_link"
    if extract_code_snippets(text):
        return "inline_code"
    if isinstance(image_ocr_text, str) and any(tok in image_ocr_text for tok in ["def ", "class ", "=>", "import ", "const ", "let "]):
        return "image_code"
    programming_terms = ["code", "coding", "programming", "compiler", "github", "repository", "script", "function"]
    if any(t in (text or "").lower() for t in programming_terms):
        return "code_discussion"
    return "none"
