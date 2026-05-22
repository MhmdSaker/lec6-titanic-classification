"""Data loading and preprocessing for the Titanic classification task."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


@dataclass
class DataSplit:
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    feature_names: list[str]


def load_titanic(drop_columns: list[str]) -> pd.DataFrame:
    """Load the Titanic dataset from seaborn and drop leakage/redundant columns."""
    df = sns.load_dataset("titanic")
    df = df.drop(columns=[c for c in drop_columns if c in df.columns])
    # Coerce boolean alone -> int (the preprocessor treats it as categorical anyway)
    if "alone" in df.columns:
        df["alone"] = df["alone"].astype(int)
    return df


def build_preprocessor(
    numeric_features: list[str],
    categorical_features: list[str],
    numeric_imputer: str = "median",
    categorical_imputer: str = "most_frequent",
    scale_numeric: bool = True,
) -> ColumnTransformer:
    """Construct an sklearn ColumnTransformer for mixed numeric/categorical data."""
    numeric_steps = [("imputer", SimpleImputer(strategy=numeric_imputer))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))
    numeric_pipe = Pipeline(numeric_steps)

    categorical_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy=categorical_imputer)),
            ("onehot", OneHotEncoder(handle_unknown="ignore", drop="if_binary")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric_features),
            ("cat", categorical_pipe, categorical_features),
        ]
    )


def split_data(
    df: pd.DataFrame, target: str, test_size: float, seed: int
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Stratified train/test split that preserves class balance."""
    X = df.drop(columns=[target])
    y = df[target]
    return train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )


def describe_dataset(df: pd.DataFrame, target: str) -> dict:
    """Return summary statistics required by Task 1."""
    counts = df[target].value_counts().sort_index()
    return {
        "n_samples": int(len(df)),
        "n_features": int(df.shape[1] - 1),
        "feature_columns": [c for c in df.columns if c != target],
        "target": target,
        "class_counts": {int(k): int(v) for k, v in counts.items()},
        "class_distribution_pct": {
            int(k): round(100 * v / len(df), 2) for k, v in counts.items()
        },
        "missing_values": {
            c: int(df[c].isna().sum()) for c in df.columns if df[c].isna().any()
        },
    }
