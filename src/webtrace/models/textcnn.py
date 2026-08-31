from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
import json
from collections import Counter
from typing import Iterable

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


PAD = "<pad>"
UNK = "<unk>"


@dataclass
class TextCNNConfig:
    max_vocab: int = 20000
    min_freq: int = 1
    max_length: int = 160
    embedding_dim: int = 96
    channels: int = 64
    kernels: tuple[int, ...] = (3, 4, 5)
    dropout: float = 0.35
    batch_size: int = 32
    epochs: int = 6
    learning_rate: float = 1e-3
    seed: int = 42


def _tokenize(text: str) -> list[str]:
    import re
    return re.findall(r"[A-Za-z0-9_@#'./:-]+", str(text).lower())


def build_vocab(texts: Iterable[str], config: TextCNNConfig) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for text in texts:
        counts.update(_tokenize(text))
    vocab = {PAD: 0, UNK: 1}
    for token, freq in counts.most_common():
        if freq < config.min_freq or len(vocab) >= config.max_vocab:
            break
        vocab[token] = len(vocab)
    return vocab


def encode(text: str, vocab: dict[str, int], max_length: int) -> list[int]:
    ids = [vocab.get(t, vocab[UNK]) for t in _tokenize(text)[:max_length]]
    if len(ids) < max_length:
        ids += [vocab[PAD]] * (max_length - len(ids))
    return ids


class TextDataset(Dataset):
    def __init__(self, texts, labels, vocab, max_length):
        self.x = torch.tensor([encode(t, vocab, max_length) for t in texts], dtype=torch.long)
        self.y = torch.tensor(list(labels), dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


class TextCNN(nn.Module):
    def __init__(self, vocab_size: int, config: TextCNNConfig, num_classes: int = 2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, config.embedding_dim, padding_idx=0)
        self.convs = nn.ModuleList([
            nn.Conv1d(config.embedding_dim, config.channels, k) for k in config.kernels
        ])
        self.dropout = nn.Dropout(config.dropout)
        self.fc = nn.Linear(config.channels * len(config.kernels), num_classes)

    def forward(self, token_ids):
        x = self.embedding(token_ids).transpose(1, 2)
        features = [torch.relu(conv(x)).max(dim=2).values for conv in self.convs]
        x = torch.cat(features, dim=1)
        return self.fc(self.dropout(x))


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(p),
        "recall_macro": float(r),
        "f1_macro": float(f1),
    }


def train_textcnn(train_df, valid_df, output_dir: str, config: TextCNNConfig = TextCNNConfig()):
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    vocab = build_vocab(train_df["text"].astype(str), config)
    train_ds = TextDataset(train_df["text"].astype(str), train_df["label_harmful"].astype(int), vocab, config.max_length)
    valid_ds = TextDataset(valid_df["text"].astype(str), valid_df["label_harmful"].astype(int), vocab, config.max_length)

    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_ds, batch_size=config.batch_size, shuffle=False)

    model = TextCNN(len(vocab), config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    criterion = nn.CrossEntropyLoss()

    best_f1 = -1.0
    best_state = None
    history = []

    for epoch in range(1, config.epochs + 1):
        model.train()
        total_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * len(yb)

        model.eval()
        preds, truths = [], []
        with torch.no_grad():
            for xb, yb in valid_loader:
                logits = model(xb.to(device))
                preds.extend(logits.argmax(dim=1).cpu().tolist())
                truths.extend(yb.tolist())
        metrics = _metrics(np.asarray(truths), np.asarray(preds))
        metrics["epoch"] = epoch
        metrics["train_loss"] = total_loss / max(1, len(train_ds))
        history.append(metrics)
        if metrics["f1_macro"] > best_f1:
            best_f1 = metrics["f1_macro"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out / "model.pt")
    (out / "vocab.json").write_text(json.dumps(vocab, indent=2), encoding="utf-8")
    (out / "config.json").write_text(json.dumps(asdict(config), indent=2), encoding="utf-8")
    (out / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")

    return model, vocab, history[-1] if history else {}


def load_textcnn(model_dir: str):
    path = Path(model_dir)
    config_dict = json.loads((path / "config.json").read_text(encoding="utf-8"))
    config_dict["kernels"] = tuple(config_dict["kernels"])
    config = TextCNNConfig(**config_dict)
    vocab = json.loads((path / "vocab.json").read_text(encoding="utf-8"))
    model = TextCNN(len(vocab), config)
    model.load_state_dict(torch.load(path / "model.pt", map_location="cpu"))
    model.eval()
    return model, vocab, config


def predict_textcnn(model, vocab, config: TextCNNConfig, texts: list[str]) -> tuple[np.ndarray, np.ndarray]:
    x = torch.tensor([encode(t, vocab, config.max_length) for t in texts], dtype=torch.long)
    with torch.no_grad():
        probs = torch.softmax(model(x), dim=1).numpy()
    preds = probs.argmax(axis=1)
    return preds, probs
