"""End-to-end runner for the Lecture 6 assignment.

Loads config -> sets seeds -> builds pipeline -> trains baseline + experiment ->
saves metrics, confusion matrices, error tables, and a JSON run log.
"""
from __future__ import annotations

import json
import platform
import random
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
import yaml
from sklearn.pipeline import Pipeline

from src.data import (
    build_preprocessor,
    describe_dataset,
    load_titanic,
    split_data,
)
from src.error_analysis import build_error_table, save_error_table, summarise_errors
from src.evaluation import Evaluator, save_confusion_matrix, save_metric_comparison
from src.models import ModelFactory


PROJECT_ROOT = Path(__file__).parent


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def load_config(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def package_versions() -> dict[str, str]:
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit-learn": sklearn.__version__,
        "pyyaml": yaml.__version__,
    }


def run_one_configuration(
    label: str,
    model_cfg: dict,
    preprocessor,
    X_train,
    X_test,
    y_train,
    y_test,
    metric_names: list[str],
    seed: int,
    fig_dir: Path,
):
    model = ModelFactory.create(model_cfg, seed=seed)
    pipe = Pipeline([("preprocess", preprocessor), ("model", model)])
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    y_proba = (
        pipe.predict_proba(X_test) if hasattr(pipe.named_steps["model"], "predict_proba") else None
    )
    metrics = Evaluator(metric_names).evaluate(y_test, y_pred)
    cm = save_confusion_matrix(
        y_test,
        y_pred,
        out_path=fig_dir / f"confusion_matrix_{label}.png",
        title=f"Confusion Matrix — {label}",
    )
    return {
        "label": label,
        "model_cfg": model_cfg,
        "metrics": metrics,
        "confusion_matrix": cm.tolist(),
        "y_pred": y_pred,
        "y_proba": y_proba,
    }


def main() -> None:
    cfg = load_config(PROJECT_ROOT / "config.yaml")
    seed = int(cfg["seed"])
    set_global_seed(seed)

    fig_dir = PROJECT_ROOT / cfg["output"]["figures_dir"]
    log_dir = PROJECT_ROOT / cfg["output"]["logs_dir"]
    fig_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    # ---- Task 1: Dataset ------------------------------------------------
    df = load_titanic(drop_columns=cfg["data"]["drop_columns"])
    dataset_summary = describe_dataset(df, target=cfg["data"]["target"])
    print("Dataset summary:")
    print(json.dumps(dataset_summary, indent=2))

    # ---- Task 2: Preprocess + split + baseline --------------------------
    X_train, X_test, y_train, y_test = split_data(
        df,
        target=cfg["data"]["target"],
        test_size=cfg["data"]["test_size"],
        seed=seed,
    )
    preprocessor = build_preprocessor(
        numeric_features=cfg["preprocessing"]["numeric_features"],
        categorical_features=cfg["preprocessing"]["categorical_features"],
        numeric_imputer=cfg["preprocessing"]["numeric_imputer"],
        categorical_imputer=cfg["preprocessing"]["categorical_imputer"],
        scale_numeric=cfg["preprocessing"]["scale_numeric"],
    )

    baseline = run_one_configuration(
        label="baseline",
        model_cfg=cfg["model"]["baseline"],
        preprocessor=preprocessor,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        metric_names=cfg["metrics"],
        seed=seed,
        fig_dir=fig_dir,
    )

    # ---- Task 4: Error analysis on baseline -----------------------------
    error_table = build_error_table(
        X_test=X_test,
        y_true=y_test,
        y_pred=baseline["y_pred"],
        y_proba=baseline["y_proba"],
    )
    error_summary = summarise_errors(error_table)
    save_error_table(error_table, log_dir / "baseline_predictions.csv")
    save_error_table(
        error_table[~error_table["correct"]], log_dir / "baseline_errors.csv"
    )

    # ---- Task 5: Experiment (single-factor change: class_weight) --------
    # Rebuild a fresh preprocessor so we don't leak fitted state across runs.
    preprocessor_exp = build_preprocessor(
        numeric_features=cfg["preprocessing"]["numeric_features"],
        categorical_features=cfg["preprocessing"]["categorical_features"],
        numeric_imputer=cfg["preprocessing"]["numeric_imputer"],
        categorical_imputer=cfg["preprocessing"]["categorical_imputer"],
        scale_numeric=cfg["preprocessing"]["scale_numeric"],
    )
    experiment = run_one_configuration(
        label="experiment_class_balanced",
        model_cfg=cfg["model"]["experiment"],
        preprocessor=preprocessor_exp,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        metric_names=cfg["metrics"],
        seed=seed,
        fig_dir=fig_dir,
    )

    save_metric_comparison(
        baseline["metrics"],
        experiment["metrics"],
        out_path=fig_dir / "metric_comparison.png",
    )

    # ---- Task 6: Reproducibility log ------------------------------------
    run_log = {
        "timestamp_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "seed": seed,
        "config": cfg,
        "package_versions": package_versions(),
        "dataset_summary": dataset_summary,
        "baseline": {
            "model_cfg": baseline["model_cfg"],
            "metrics": baseline["metrics"],
            "confusion_matrix": baseline["confusion_matrix"],
        },
        "experiment": {
            "model_cfg": experiment["model_cfg"],
            "metrics": experiment["metrics"],
            "confusion_matrix": experiment["confusion_matrix"],
        },
        "error_analysis_summary": error_summary,
        "single_factor_changed": "class_weight: null -> balanced",
    }
    with open(log_dir / "run_log.json", "w", encoding="utf-8") as f:
        json.dump(run_log, f, indent=2)

    # ---- Console summary -----------------------------------------------
    print("\n=== Baseline metrics ===")
    print(json.dumps(baseline["metrics"], indent=2))
    print("\n=== Experiment metrics (class_weight=balanced) ===")
    print(json.dumps(experiment["metrics"], indent=2))
    print("\n=== Error analysis summary ===")
    print(json.dumps(error_summary, indent=2))
    print(f"\nArtifacts written to:\n  figures: {fig_dir}\n  logs:    {log_dir}")


if __name__ == "__main__":
    main()
