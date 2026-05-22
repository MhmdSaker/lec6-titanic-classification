"""Evaluation metrics (Strategy pattern) and confusion-matrix plotting.

Implements the Strategy pattern from Lecture 6 — each metric is a swappable
strategy with a common .score() interface.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


class MetricStrategy(ABC):
    name: str = "metric"

    @abstractmethod
    def score(self, y_true, y_pred) -> float: ...


class AccuracyStrategy(MetricStrategy):
    name = "accuracy"

    def score(self, y_true, y_pred) -> float:
        return float(accuracy_score(y_true, y_pred))


class PrecisionStrategy(MetricStrategy):
    name = "precision"

    def score(self, y_true, y_pred) -> float:
        return float(precision_score(y_true, y_pred, zero_division=0))


class RecallStrategy(MetricStrategy):
    name = "recall"

    def score(self, y_true, y_pred) -> float:
        return float(recall_score(y_true, y_pred, zero_division=0))


class F1Strategy(MetricStrategy):
    name = "f1"

    def score(self, y_true, y_pred) -> float:
        return float(f1_score(y_true, y_pred, zero_division=0))


_STRATEGY_REGISTRY = {
    "accuracy": AccuracyStrategy,
    "precision": PrecisionStrategy,
    "recall": RecallStrategy,
    "f1": F1Strategy,
}


class Evaluator:
    """Computes a configurable set of metric strategies for one prediction set."""

    def __init__(self, metric_names: Iterable[str]):
        self.strategies = [_STRATEGY_REGISTRY[m]() for m in metric_names]

    def evaluate(self, y_true, y_pred) -> dict[str, float]:
        return {s.name: round(s.score(y_true, y_pred), 4) for s in self.strategies}


def save_confusion_matrix(
    y_true,
    y_pred,
    out_path: str | Path,
    title: str,
    labels: list[str] | None = None,
) -> np.ndarray:
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm, display_labels=labels or ["Died", "Survived"]
    )
    disp.plot(cmap="Blues", ax=ax, colorbar=False)
    ax.set_title(title)
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return cm


def save_metric_comparison(
    baseline_metrics: dict[str, float],
    experiment_metrics: dict[str, float],
    out_path: str | Path,
) -> None:
    names = list(baseline_metrics.keys())
    base_vals = [baseline_metrics[n] for n in names]
    exp_vals = [experiment_metrics[n] for n in names]

    x = np.arange(len(names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(x - width / 2, base_vals, width, label="Baseline")
    ax.bar(x + width / 2, exp_vals, width, label="Experiment (balanced)")
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Baseline vs Experiment — Metric Comparison")
    ax.legend()
    for i, (b, e) in enumerate(zip(base_vals, exp_vals)):
        ax.text(i - width / 2, b + 0.01, f"{b:.2f}", ha="center", fontsize=8)
        ax.text(i + width / 2, e + 0.01, f"{e:.2f}", ha="center", fontsize=8)
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
