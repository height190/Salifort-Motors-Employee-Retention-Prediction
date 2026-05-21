"""Feature engineering utilities for employee attrition modeling."""

from __future__ import annotations

import pandas as pd


TARGET = "left"

BASE_FEATURES = [
    "satisfaction_level",
    "last_evaluation",
    "number_project",
    "average_monthly_hours",
    "tenure",
    "work_accident",
    "promotion_last_5years",
    "department",
    "salary",
]

LEAKAGE_AWARE_FEATURES = [
    "last_evaluation",
    "number_project",
    "tenure",
    "work_accident",
    "promotion_last_5years",
    "department",
    "salary",
    "overworked",
]


def add_retention_features(df: pd.DataFrame, overwork_threshold: int = 175) -> pd.DataFrame:
    """Add interpretable features aligned to HR retention actions."""
    featured = df.copy()

    if "average_monthly_hours" in featured.columns:
        featured["overworked"] = (featured["average_monthly_hours"] > overwork_threshold).astype(int)

    if {"number_project", "average_monthly_hours"}.issubset(featured.columns):
        featured["hours_per_project"] = (
            featured["average_monthly_hours"] / featured["number_project"].replace(0, pd.NA)
        ).astype("float")

    if "tenure" in featured.columns:
        featured["early_tenure"] = featured["tenure"].between(2, 4).astype(int)

    return featured


def get_feature_columns(leakage_aware: bool = True) -> list[str]:
    """Return feature columns for the selected modeling scenario."""
    return LEAKAGE_AWARE_FEATURES if leakage_aware else BASE_FEATURES


def split_features_target(
    df: pd.DataFrame,
    leakage_aware: bool = True,
    target: str = TARGET,
) -> tuple[pd.DataFrame, pd.Series]:
    """Split a dataframe into model features and target."""
    features = [column for column in get_feature_columns(leakage_aware) if column in df.columns]
    missing = set(get_feature_columns(leakage_aware)) - set(features)
    if missing:
        raise ValueError(f"Missing expected feature columns: {sorted(missing)}")
    if target not in df.columns:
        raise ValueError(f"Missing target column: {target}")

    return df[features].copy(), df[target].astype(int)
