"""Data cleaning helpers for the Salifort Motors HR dataset."""

import re

import pandas as pd


COLUMN_RENAME_MAP = {
    "time_spend_company": "tenure",
    "work_accident": "work_accident",
    "department": "department",
}


def to_snake_case(column: str) -> str:
    """Convert a column name to snake_case."""
    value = column.strip()
    value = re.sub(r"(?<!^)(?=[A-Z])", "_", value)
    value = re.sub(r"[^0-9a-zA-Z]+", "_", value)
    value = re.sub(r"_+", "_", value).strip("_").lower()
    return value


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with consistent, concise column names."""
    cleaned = df.copy()
    cleaned.columns = [to_snake_case(col) for col in cleaned.columns]
    cleaned = cleaned.rename(columns=COLUMN_RENAME_MAP)
    return cleaned


def clean_employee_data(df: pd.DataFrame, drop_duplicates: bool = True) -> pd.DataFrame:
    """Clean the raw employee dataset for analysis and modeling."""
    cleaned = standardize_columns(df)

    if drop_duplicates:
        cleaned = cleaned.drop_duplicates().reset_index(drop=True)

    category_columns = ["department", "salary"]
    for column in category_columns:
        if column in cleaned.columns:
            cleaned[column] = cleaned[column].astype(str).str.strip().str.lower()

    integer_columns = ["number_project", "tenure", "work_accident", "left", "promotion_last_5years"]
    for column in integer_columns:
        if column in cleaned.columns:
            cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce").astype("Int64")

    numeric_columns = ["satisfaction_level", "last_evaluation", "average_monthly_hours"]
    for column in numeric_columns:
        if column in cleaned.columns:
            cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")

    return cleaned


def summarize_data_quality(df: pd.DataFrame) -> pd.DataFrame:
    """Create a compact data quality summary for notebook display."""
    return pd.DataFrame(
        {
            "dtype": df.dtypes.astype(str),
            "missing_values": df.isna().sum(),
            "missing_pct": (df.isna().mean() * 100).round(2),
            "unique_values": df.nunique(dropna=False),
        }
    )
