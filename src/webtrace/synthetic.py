from __future__ import annotations

import random
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd

from .utils import seed_everything, stable_id

NEUTRAL = [
    "I released a small parser today and wrote up the design notes.",
    "New visualization experiment, source is linked below.",
    "Trying a functional approach for this tiny compiler project.",
    "Sharing notes from my data analysis and code repository.",
    "Built a small simulation and compared several heuristics.",
    "This implementation still needs tests but the basic idea works.",
]
HARMFUL = [
    "You are an idiot and your code is garbage.",
    "Go away, nobody wants your useless project here.",
    "What a stupid implementation, you clearly know nothing.",
    "People like you should stop posting this trash.",
]
CODE = [
    "```python\ndef score(x):\n    return x * 2\n```",
    "`val twice = (x: Int) => x * 2`",
    "```javascript\nconst f = x => x + 1;\n```",
]


def generate_demo(n: int = 5000, seed: int = 42) -> pd.DataFrame:
    seed_everything(seed)
    rows = []
    for i in range(n):
        harmful = 1 if random.random() < 0.25 else 0
        base = random.choice(HARMFUL if harmful else NEUTRAL)
        repo = None
        if random.random() < 0.35:
            repo = f"https://github.com/demo-user/repo-{i % 50}"
            base += f" {repo}"
        if random.random() < 0.20:
            base += " " + random.choice(CODE)
        rows.append({
            "post_id": stable_id("demo", str(i), base),
            "platform": random.choice(["mastodon:demo", "forum:demo", "microblog:demo"]),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "author": f"user_{i % 300}",
            "text": base,
            "url": f"https://example.invalid/post/{i}",
            "repo_url": repo,
            "code_snippet": "",
            "image_path": "",
            "image_ocr_text": "",
            "label_harmful": harmful,
        })
    return pd.DataFrame(rows)
