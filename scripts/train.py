from __future__ import annotations

import argparse
import inspect
from pathlib import Path

import numpy as np
from datasets import Dataset, DatasetDict, load_dataset
from sklearn.metrics import f1_score, roc_auc_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

LABELS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
CIVIL_COMMENTS_LABELS = [
    "toxicity",
    "severe_toxicity",
    "obscene",
    "threat",
    "insult",
    "identity_attack",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default="distilbert-base-uncased")
    parser.add_argument("--dataset-name", default="civil_comments")
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--output-dir", default="models/distilbert-toxic")
    parser.add_argument("--epochs", type=float, default=2)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--max-length", type=int, default=192)
    parser.add_argument("--max-samples", type=int, default=2000)
    parser.add_argument("--label-threshold", type=float, default=0.5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = load_toxic_dataset(args.dataset_name, args.max_samples)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    def tokenize(batch: dict) -> dict:
        tokenized = tokenizer(
            batch[args.text_column],
            max_length=args.max_length,
            truncation=True,
        )
        tokenized["labels"] = np.array(batch["labels"], dtype=np.float32)
        return tokenized

    encoded = dataset.map(tokenize, batched=True, remove_columns=dataset["train"].column_names)
    id2label = {i: label for i, label in enumerate(LABELS)}
    label2id = {label: i for i, label in id2label.items()}
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=len(LABELS),
        problem_type="multi_label_classification",
        id2label=id2label,
        label2id=label2id,
    )

    training_args = build_training_args(args)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=encoded["train"],
        eval_dataset=encoded["validation"],
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=lambda pred: compute_metrics(pred, args.label_threshold),
    )
    trainer.train()
    output_dir = Path(args.output_dir)
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)


def load_toxic_dataset(dataset_name: str, max_samples: int | None) -> DatasetDict:
    if max_samples:
        return load_streamed_sample(dataset_name, max_samples)

    dataset = load_dataset(dataset_name)
    if "validation" not in dataset:
        split = dataset["train"].train_test_split(test_size=0.1, seed=42)
        dataset = DatasetDict(train=split["train"], validation=split["test"])

    source_labels = CIVIL_COMMENTS_LABELS if dataset_name == "civil_comments" else LABELS

    def add_labels(row: dict) -> dict:
        row["labels"] = [float(row[source]) for source in source_labels]
        return row

    dataset = dataset.map(add_labels)
    return dataset


def load_streamed_sample(dataset_name: str, max_samples: int) -> DatasetDict:
    validation_samples = max(max_samples // 5, 1)
    train_rows = list(load_dataset(dataset_name, split="train", streaming=True).take(max_samples))
    validation_rows = list(
        load_dataset(dataset_name, split="validation", streaming=True).take(validation_samples)
    )
    dataset = DatasetDict(
        train=Dataset.from_list(train_rows),
        validation=Dataset.from_list(validation_rows),
    )
    source_labels = CIVIL_COMMENTS_LABELS if dataset_name == "civil_comments" else LABELS

    def add_labels(row: dict) -> dict:
        row["labels"] = [float(row[source]) for source in source_labels]
        return row

    return dataset.map(add_labels)


def compute_metrics(pred, threshold: float) -> dict:
    logits, labels = pred
    probs = 1.0 / (1.0 + np.exp(-logits))
    predictions = (probs >= threshold).astype(int)
    metrics = {
        "f1_micro": f1_score(labels, predictions, average="micro", zero_division=0),
        "f1_macro": f1_score(labels, predictions, average="macro", zero_division=0),
    }
    try:
        metrics["roc_auc_macro"] = roc_auc_score(labels, probs, average="macro")
    except ValueError:
        metrics["roc_auc_macro"] = 0.0
    return metrics


def build_training_args(args: argparse.Namespace) -> TrainingArguments:
    kwargs = {
        "output_dir": args.output_dir,
        "save_strategy": "epoch",
        "learning_rate": args.learning_rate,
        "per_device_train_batch_size": args.batch_size,
        "per_device_eval_batch_size": args.batch_size,
        "num_train_epochs": args.epochs,
        "weight_decay": 0.01,
        "load_best_model_at_end": True,
        "metric_for_best_model": "f1_macro",
        "greater_is_better": True,
        "report_to": "none",
    }
    signature = inspect.signature(TrainingArguments.__init__)
    strategy_arg = "eval_strategy" if "eval_strategy" in signature.parameters else "evaluation_strategy"
    kwargs[strategy_arg] = "epoch"
    return TrainingArguments(**kwargs)


if __name__ == "__main__":
    main()
