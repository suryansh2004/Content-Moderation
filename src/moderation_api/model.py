from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Protocol

import numpy as np
import onnxruntime as ort
import torch
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer

from moderation_api.labels import DEFAULT_LABELS, normalize_label


class ModerationModel(Protocol):
    backend: str
    labels: list[str]

    def predict(self, texts: list[str], threshold: float) -> tuple[list[dict], float]:
        ...


def sigmoid(logits: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-logits))


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=-1, keepdims=True)


def resolve_labels(model_dir: str | Path) -> list[str]:
    config = AutoConfig.from_pretrained(model_dir)
    if config.id2label:
        labels = [config.id2label.get(i) or config.id2label.get(str(i)) for i in range(len(config.id2label))]
        normalized = [normalize_label(label) for label in labels]
        if normalized == ["label_0", "label_1"]:
            return ["non_toxic", "toxic"]
        return normalized
    return DEFAULT_LABELS


@dataclass
class TorchModerationModel:
    model_dir: str | Path
    max_length: int = 192
    backend: str = "torch"

    def __post_init__(self) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_dir)
        self.model.eval()
        self.labels = resolve_labels(self.model_dir)
        self.problem_type = self.model.config.problem_type

    @torch.inference_mode()
    def predict(self, texts: list[str], threshold: float) -> tuple[list[dict], float]:
        started = perf_counter()
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        outputs = self.model(**encoded)
        scores = scores_from_logits(outputs.logits.detach().cpu().numpy(), self.problem_type)
        return format_results(texts, scores, self.labels, threshold), elapsed_ms(started)


@dataclass
class OnnxModerationModel:
    model_dir: Path
    max_length: int = 192
    provider: str = "CPUExecutionProvider"
    backend: str = "onnx"

    def __post_init__(self) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir)
        self.labels = resolve_labels(self.model_dir)
        self.problem_type = AutoConfig.from_pretrained(self.model_dir).problem_type
        providers = [self.provider] if self.provider in ort.get_available_providers() else None
        self.session = ort.InferenceSession(str(find_onnx_file(self.model_dir)), providers=providers)
        self.input_names = {input_.name for input_ in self.session.get_inputs()}

    def predict(self, texts: list[str], threshold: float) -> tuple[list[dict], float]:
        started = perf_counter()
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="np",
        )
        ort_inputs = {name: value for name, value in encoded.items() if name in self.input_names}
        logits = self.session.run(None, ort_inputs)[0]
        scores = scores_from_logits(logits, self.problem_type)
        return format_results(texts, scores, self.labels, threshold), elapsed_ms(started)


def find_onnx_file(model_dir: str | Path) -> Path:
    model_path = Path(model_dir)
    optimized = sorted(model_path.glob("*optimized*.onnx"))
    if optimized:
        return optimized[0]
    candidates = sorted(model_path.glob("*.onnx"))
    if not candidates:
        raise FileNotFoundError(f"No .onnx model file found in {model_path}")
    return candidates[0]


def format_results(
    texts: list[str],
    scores: np.ndarray,
    labels: list[str],
    threshold: float,
) -> list[dict]:
    results = []
    positive_indices = positive_label_indices(labels)
    for text, row in zip(texts, scores, strict=True):
        label_scores = [
            {
                "label": label,
                "score": float(score),
                "flagged": bool(index in positive_indices and score >= threshold),
            }
            for index, (label, score) in enumerate(zip(labels, row, strict=True))
        ]
        max_score = max(item["score"] for item in label_scores)
        results.append(
            {
                "text": text,
                "flagged": any(item["flagged"] for item in label_scores),
                "max_score": max_score,
                "labels": label_scores,
            }
        )
    return results


def scores_from_logits(logits: np.ndarray, problem_type: str | None) -> np.ndarray:
    if logits.shape[-1] == 1 or problem_type == "multi_label_classification":
        return sigmoid(logits)
    return softmax(logits)


def positive_label_indices(labels: list[str]) -> set[int]:
    negative_labels = {"non_toxic", "not_toxic", "neutral", "clean", "safe", "acceptable", "label_0"}
    positives = {index for index, label in enumerate(labels) if label not in negative_labels}
    if len(labels) == 2 and not positives:
        return {1}
    return positives


def elapsed_ms(started: float) -> float:
    return (perf_counter() - started) * 1000.0


def load_model(
    model_dir: str | Path,
    backend: str,
    max_length: int,
    onnx_provider: str = "CPUExecutionProvider",
) -> ModerationModel:
    if backend == "onnx":
        return OnnxModerationModel(model_dir=model_dir, max_length=max_length, provider=onnx_provider)
    return TorchModerationModel(model_dir=model_dir, max_length=max_length)
