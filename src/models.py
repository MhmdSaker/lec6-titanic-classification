"""Model factory pattern — instantiate classifiers from config dicts.

Implements the Factory pattern from Lecture 6 (slide: Factory Pattern in AI Systems).
"""
from __future__ import annotations

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier


class ModelFactory:
    """Centralised classifier creation — swap models via config, not code edits."""

    _REGISTRY = {
        "logistic_regression": LogisticRegression,
        "decision_tree": DecisionTreeClassifier,
        "random_forest": RandomForestClassifier,
    }

    @classmethod
    def create(cls, model_cfg: dict, seed: int):
        cfg = dict(model_cfg)
        model_type = cfg.pop("type")
        if model_type not in cls._REGISTRY:
            raise ValueError(
                f"Unknown model type '{model_type}'. "
                f"Registered: {list(cls._REGISTRY)}"
            )
        klass = cls._REGISTRY[model_type]
        # Inject random_state where the estimator supports it
        if "random_state" in klass().get_params():
            cfg.setdefault("random_state", seed)
        return klass(**cfg)
