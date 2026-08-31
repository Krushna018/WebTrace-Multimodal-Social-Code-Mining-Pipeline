from __future__ import annotations

from pathlib import Path


def extract_text_from_image(path: str | Path) -> str:
    """OCR helper. Requires the system `tesseract` executable plus pytesseract."""
    import cv2
    import pytesseract

    image = cv2.imread(str(path))
    if image is None:
        raise FileNotFoundError(path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return pytesseract.image_to_string(thresh, config="--psm 6").strip()
