from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class TransformerConfig:
    model_name: str = "distilbert-base-uncased"
    max_length: int = 192
    epochs: int = 2
    batch_size: int = 16
    learning_rate: float = 2e-5


def train_transformer(train_df, valid_df, output_dir: str, config: TransformerConfig = TransformerConfig()):
    """Fine-tune a small transformer classifier.

    Input dataframes must have columns `text` and `label_harmful`.
    The function is intentionally isolated so the rest of the project remains runnable
    without downloading a transformer model.
    """
    from datasets import Dataset
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        DataCollatorWithPadding,
        Trainer,
        TrainingArguments,
    )
    import numpy as np
    from sklearn.metrics import precision_recall_fscore_support, accuracy_score

    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(config.model_name, num_labels=2)

    def tok(batch):
        return tokenizer(batch["text"], truncation=True, max_length=config.max_length)

    train_ds = Dataset.from_pandas(train_df[["text", "label_harmful"]].rename(columns={"label_harmful": "label"}), preserve_index=False).map(tok, batched=True)
    valid_ds = Dataset.from_pandas(valid_df[["text", "label_harmful"]].rename(columns={"label_harmful": "label"}), preserve_index=False).map(tok, batched=True)

    def metrics(pred):
        logits, labels = pred
        preds = np.argmax(logits, axis=-1)
        p, r, f1, _ = precision_recall_fscore_support(labels, preds, average="macro", zero_division=0)
        return {"accuracy": accuracy_score(labels, preds), "precision_macro": p, "recall_macro": r, "f1_macro": f1}

    args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=config.epochs,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        learning_rate=config.learning_rate,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        report_to=[],
        seed=42,
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=valid_ds,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=metrics,
    )
    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    return trainer.evaluate()
