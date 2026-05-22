"""Qualitative error analysis — Task 4."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def build_error_table(
    X_test: pd.DataFrame,
    y_true,
    y_pred,
    y_proba=None,
) -> pd.DataFrame:
    """Return a per-sample table flagging which predictions were wrong."""
    df = X_test.copy().reset_index(drop=False).rename(columns={"index": "orig_index"})
    df["true_label"] = np.asarray(y_true)
    df["predicted"] = np.asarray(y_pred)
    df["correct"] = df["true_label"] == df["predicted"]
    if y_proba is not None:
        df["confidence"] = np.max(y_proba, axis=1).round(3)
    return df


def summarise_errors(error_table: pd.DataFrame) -> dict:
    """Compute the aggregate stats referenced by the report's Error Analysis section."""
    errors = error_table[~error_table["correct"]]
    total = len(error_table)
    n_err = len(errors)
    fn = errors[(errors["true_label"] == 1) & (errors["predicted"] == 0)]
    fp = errors[(errors["true_label"] == 0) & (errors["predicted"] == 1)]

    by_sex = (
        errors.groupby("sex").size().to_dict() if "sex" in errors.columns else {}
    )
    by_pclass = (
        errors.groupby("pclass").size().to_dict()
        if "pclass" in errors.columns
        else {}
    )

    age_stats = {}
    if "age" in errors.columns and errors["age"].notna().any():
        age_stats = {
            "mean_age_errors": round(float(errors["age"].mean(skipna=True)), 2),
            "mean_age_overall_test": round(
                float(error_table["age"].mean(skipna=True)), 2
            ),
            "missing_age_in_errors": int(errors["age"].isna().sum()),
        }

    return {
        "total_test_samples": total,
        "total_errors": n_err,
        "error_rate": round(n_err / total, 4) if total else 0.0,
        "false_negatives": len(fn),
        "false_positives": len(fp),
        "errors_by_sex": by_sex,
        "errors_by_pclass": by_pclass,
        "age_stats": age_stats,
    }


def save_error_table(error_table: pd.DataFrame, out_path: str | Path) -> None:
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    error_table.to_csv(out_path, index=False)
