"""Yearly aggregation and analytics."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from .entity_lists import ALL_BIOMARKER_COLUMNS, ALL_ML_METHOD_COLUMNS
from .extract import ENTITY_YEAR_CONSTRAINTS

logger = logging.getLogger(__name__)


def _clip_anachronistic_ml_methods(df: pd.DataFrame, ml_columns: list[str]) -> None:
    """Zeruje wiersze metod ML dla lat przed minimalnym rokiem (zabezpieczenie po agregacji)."""
    if "year" not in df.columns:
        return
    y = pd.to_numeric(df["year"], errors="coerce")
    for col in ml_columns:
        min_y = ENTITY_YEAR_CONSTRAINTS.get(col)
        if min_y is None or col not in df.columns:
            continue
        invalid = y.notna() & (y < min_y)
        if invalid.any():
            df.loc[invalid, col] = 0


def aggregate_by_year(input_path: str) -> pd.DataFrame:
    """Aggregate enriched records by publication year.

    Computes per-year totals, rates (per 1000 abstracts), and
    AI penetration index (ml_rate / biomarker_rate).

    Args:
        input_path: Path to enriched .jsonl file.

    Returns:
        DataFrame with one row per year and columns for all metrics.
    """
    records = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            import json
            records.append(json.loads(line))

    if not records:
        logger.warning("No records found in %s", input_path)
        return pd.DataFrame()

    df = pd.DataFrame(records)

    all_biomarkers = list(ALL_BIOMARKER_COLUMNS)
    all_ml_methods = list(ALL_ML_METHOD_COLUMNS)

    # Expand nested entity dicts into top-level columns for aggregation
    for col in all_biomarkers:
        df[col] = df["biomarkers_found"].apply(lambda d: d.get(col, 0))
    for col in all_ml_methods:
        df[col] = df["ml_methods_found"].apply(lambda d: d.get(col, 0))

    _clip_anachronistic_ml_methods(df, all_ml_methods)
    ml_cols = [c for c in all_ml_methods if c in df.columns]
    df["has_ml"] = df[ml_cols].sum(axis=1) > 0

    # Per-year aggregation
    grouped = df.groupby("year")

    agg_dict: dict = {
        "pmid": "count",
        "has_biomarker": "sum",
        "has_ml": "sum",
    }
    for col in all_biomarkers + all_ml_methods:
        agg_dict[col] = "sum"

    year_df = grouped.agg(agg_dict).rename(columns={"pmid": "total_abstracts"})
    year_df = year_df.rename(
        columns={"has_biomarker": "abstracts_with_any_biomarker",
                 "has_ml": "abstracts_with_any_ml"}
    )

    # Compute rates per 1000 abstracts
    year_df["biomarker_rate"] = (
        year_df["abstracts_with_any_biomarker"] / year_df["total_abstracts"] * 1000
    )
    year_df["ml_rate"] = (
        year_df["abstracts_with_any_ml"] / year_df["total_abstracts"] * 1000
    )

    # AI penetration index (handle div-by-zero)
    year_df["ai_penetration_index"] = year_df["ml_rate"] / year_df["biomarker_rate"].replace(0, float("nan"))

    year_df = year_df.reset_index()
    logger.info("Aggregated %d records into %d year rows", len(df), len(year_df))
    return year_df


def add_trend_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add year-over-year growth and 3-year rolling average.

    Args:
        df: DataFrame from aggregate_by_year.

    Returns:
        Updated DataFrame with new trend columns.
    """
    df = df.copy()

    # Sort by year to ensure correct pct_change
    df = df.sort_values("year").reset_index(drop=True)

    # Year-over-year growth (percentage)
    df["ml_growth_yoy"] = df["ml_rate"].pct_change() * 100
    df["biomarker_growth_yoy"] = df["biomarker_rate"].pct_change() * 100

    # 3-year rolling average (center=True for centered window)
    df["ml_rate_smooth"] = df["ml_rate"].rolling(window=3, center=True, min_periods=1).mean()
    df["biomarker_rate_smooth"] = df["biomarker_rate"].rolling(window=3, center=True, min_periods=1).mean()

    logger.info("Added trend features to DataFrame")
    return df


def save_aggregated(df: pd.DataFrame, output_path: str) -> None:
    """Save aggregated DataFrame to CSV and Parquet.

    Args:
        df: DataFrame to save.
        output_path: Base path (extensions .csv and .parquet will be added).
    """
    base = Path(output_path)
    base.parent.mkdir(parents=True, exist_ok=True)

    csv_path = base.with_suffix(".csv")
    parquet_path = base.with_suffix(".parquet")

    df.to_csv(csv_path, index=False, encoding="utf-8")
    logger.info("Saved CSV -> %s", csv_path)

    df.to_parquet(parquet_path, index=False)
    logger.info("Saved Parquet -> %s", parquet_path)
