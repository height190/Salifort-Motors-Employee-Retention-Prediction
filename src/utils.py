"""Shared utilities for the Salifort Motors retention project."""

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
IMAGES = PROJECT_ROOT / "images"
REPORTS = PROJECT_ROOT / "reports"


def ensure_project_dirs() -> None:
    """Create expected output directories if they do not already exist."""
    for path in [
        DATA_RAW,
        DATA_PROCESSED,
        IMAGES / "eda",
        IMAGES / "modeling",
        IMAGES / "results",
        REPORTS,
    ]:
        path.mkdir(parents=True, exist_ok=True)


def load_raw_data(filename: str = "HR_capstone_dataset.csv") -> pd.DataFrame:
    """Load the capstone CSV from data/raw with a clear error message."""
    path = DATA_RAW / filename
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. Add HR_capstone_dataset.csv to data/raw/ "
            "before running the analysis."
        )
    return pd.read_csv(path)


def save_processed_data(df: pd.DataFrame, filename: str = "salifort_clean.csv") -> Path:
    """Save a processed dataset and return its path."""
    ensure_project_dirs()
    path = DATA_PROCESSED / filename
    df.to_csv(path, index=False)
    return path


def load_processed_data(filename: str = "salifort_clean.csv") -> pd.DataFrame:
    """Load a processed dataset from data/processed."""
    path = DATA_PROCESSED / filename
    if not path.exists():
        raise FileNotFoundError(
            f"Processed dataset not found at {path}. Run notebooks/01_data_cleaning.ipynb first."
        )
    return pd.read_csv(path)


def save_plot(fig, relative_path: str, dpi: int = 150) -> Path:
    """Save a Matplotlib figure under the project root."""
    ensure_project_dirs()
    path = PROJECT_ROOT / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    return path
