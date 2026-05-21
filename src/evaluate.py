"""Model evaluation helpers for classification results."""

from __future__ import annotations

import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def classification_metrics(model, X_test: pd.DataFrame, y_test: pd.Series) -> dict[str, float]:
    """Compute common binary classification metrics."""
    y_pred = model.predict(X_test)
    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_score)
    else:
        auc = None

    return {
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "accuracy": accuracy_score(y_test, y_pred),
        "auc": auc,
    }


def compare_models(models: dict, X_test: pd.DataFrame, y_test: pd.Series) -> pd.DataFrame:
    """Return a sorted model comparison table."""
    rows = []
    for name, model in models.items():
        metrics = classification_metrics(model, X_test, y_test)
        rows.append({"model": name, **metrics})

    return (
        pd.DataFrame(rows)
        .sort_values(["recall", "f1"], ascending=False)
        .reset_index(drop=True)
        .round(3)
    )


def plot_confusion_matrix(model, X_test: pd.DataFrame, y_test: pd.Series, title: str):
    """Plot a confusion matrix for a fitted classifier."""
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_estimator(
        model,
        X_test,
        y_test,
        display_labels=["Stayed", "Left"],
        cmap="Blues",
        values_format="d",
        ax=ax,
    )
    ax.set_title(title)
    return fig, ax


def plot_roc_curve(model, X_test: pd.DataFrame, y_test: pd.Series, title: str):
    """Plot a ROC curve for a fitted classifier."""
    fig, ax = plt.subplots(figsize=(5, 4))
    RocCurveDisplay.from_estimator(model, X_test, y_test, ax=ax)
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
    ax.set_title(title)
    return fig, ax


def get_feature_importance(model) -> pd.DataFrame:
    """Extract feature importances from tree-based sklearn pipelines."""
    if "model" not in model.named_steps or "preprocess" not in model.named_steps:
        raise ValueError("Expected a sklearn Pipeline with preprocess and model steps.")

    estimator = model.named_steps["model"]
    if not hasattr(estimator, "feature_importances_"):
        raise ValueError("The selected estimator does not expose feature_importances_.")

    feature_names = model.named_steps["preprocess"].get_feature_names_out()
    return (
        pd.DataFrame({"feature": feature_names, "importance": estimator.feature_importances_})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )


def plot_feature_importance(importance: pd.DataFrame, top_n: int = 12):
    """Plot the top model features by importance."""
    plot_data = importance.head(top_n).sort_values("importance")
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(plot_data["feature"], plot_data["importance"], color="#2f6f8f")
    ax.set_title("Top Random Forest Feature Importances")
    ax.set_xlabel("Importance")
    ax.set_ylabel("")
    return fig, ax
