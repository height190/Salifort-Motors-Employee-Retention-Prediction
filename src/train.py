"""Model training workflow for the Salifort Motors retention project."""

from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

try:
    from feature_engineering import split_features_target
except ImportError:
    from .feature_engineering import split_features_target


RANDOM_STATE = 42


def _one_hot_encoder() -> OneHotEncoder:
    """Create an encoder compatible with recent and older scikit-learn versions."""
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def make_preprocessor(X: pd.DataFrame, scale_numeric: bool = False) -> ColumnTransformer:
    """Build a preprocessing transformer for numeric and categorical columns."""
    categorical_features = X.select_dtypes(include=["object", "category"]).columns.tolist()
    numeric_features = [column for column in X.columns if column not in categorical_features]

    numeric_transformer = StandardScaler() if scale_numeric else "passthrough"

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_transformer, numeric_features),
            ("categorical", _one_hot_encoder(), categorical_features),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def build_models(X: pd.DataFrame) -> dict[str, Pipeline]:
    """Create baseline and tree-based model pipelines."""
    return {
        "Logistic Regression": Pipeline(
            steps=[
                ("preprocess", make_preprocessor(X, scale_numeric=True)),
                (
                    "model",
                    LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE),
                ),
            ]
        ),
        "Decision Tree": Pipeline(
            steps=[
                ("preprocess", make_preprocessor(X)),
                (
                    "model",
                    DecisionTreeClassifier(
                        max_depth=8,
                        min_samples_leaf=25,
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "Random Forest": Pipeline(
            steps=[
                ("preprocess", make_preprocessor(X)),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=300,
                        max_depth=12,
                        min_samples_leaf=10,
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
    }


def prepare_train_test_data(
    df: pd.DataFrame,
    leakage_aware: bool = True,
    test_size: float = 0.2,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Create a stratified train/test split."""
    X, y = split_features_target(df, leakage_aware=leakage_aware)
    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=RANDOM_STATE,
        stratify=y,
    )


def train_candidate_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> dict[str, Pipeline]:
    """Fit all candidate models and return trained pipelines."""
    models = build_models(X_train)
    for model in models.values():
        model.fit(X_train, y_train)
    return models
