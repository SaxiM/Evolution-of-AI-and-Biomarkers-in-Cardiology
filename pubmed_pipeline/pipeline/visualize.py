"""Visualization functions for aggregated PubMed trend data."""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

logger = logging.getLogger(__name__)

sns.set_style("whitegrid")

ML_MILESTONES = {
    2001: "Random Forest",
    2006: "Deep Learning (Hinton)",
    2012: "AlexNet",
    2014: "XGBoost",
    2017: "Transformer (Attention)",
    2018: "BERT / GPT",
    2022: "ChatGPT",
}


def generate_all_plots(df: pd.DataFrame, output_dir: str = "data/figures/") -> None:
    """Generate and save all trend plots.

    Produces five matplotlib figures:
        1. ai_vs_biomarkers.png      — smoothed rate comparison
        2. ai_penetration_index.png  — ML/biomarker rate ratio over time
        3. ml_yoy_growth.png         — year-over-year ML growth bar chart
        4. ml_method_breakdown.png   — stacked area of ML method counts
        5. biomarker_breakdown.png  — stacked area of biomarker counts

    Args:
        df: Aggregated DataFrame (from aggregate_by_year + add_trend_features).
        output_dir: Directory to save figures.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    all_ml = [
        "logistic_regression", "random_forest", "SVM", "neural_network",
        "deep_learning", "transformer", "XGBoost", "naive_bayes",
        "decision_tree", "clustering",
    ]
    all_biomarkers = [
        "CRP", "troponin", "cholesterol", "HbA1c", "IL-6",
        "glucose", "BNP", "ferritin", "albumin", "creatinine",
    ]

    # ------------------------------------------------------------------
    # Plot 1: AI vs Biomarkers smoothed rate comparison
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(12, 6), dpi=150)
    ax.plot(
        df["year"], df["biomarker_rate_smooth"],
        label="Biomarker Rate (smoothed)", color="#2196F3", linewidth=2,
    )
    ax.plot(
        df["year"], df["ml_rate_smooth"],
        label="ML Rate (smoothed)", color="#FF5722", linewidth=2,
    )
    ax.fill_between(
        df["year"], df["ml_rate_smooth"],
        alpha=0.15, color="#FF5722",
    )
    ax.set_xlabel("Year")
    ax.set_ylabel("Mentions per 1,000 abstracts")
    ax.set_title("AI/ML vs Biomarker Mentions in PubMed (per 1000 abstracts)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(output_path / "01_ai_vs_biomarkers.png")
    plt.close(fig)
    logger.info("Saved 01_ai_vs_biomarkers.png")

    # ------------------------------------------------------------------
    # Plot 2: AI Penetration Index
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(12, 6), dpi=150)
    ax.plot(
        df["year"], df["ai_penetration_index"],
        color="#9C27B0", linewidth=2,
    )
    ax.axhline(
        y=1.0, color="gray", linestyle="--", linewidth=1.5,
        label="Parity (y=1)",
    )
    ax.set_xlabel("Year")
    ax.set_ylabel("AI Penetration Index")
    ax.set_title("AI Penetration Index (ML rate / Biomarker rate)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(output_path / "02_ai_penetration_index.png")
    plt.close(fig)
    logger.info("Saved 02_ai_penetration_index.png")

    # ------------------------------------------------------------------
    # Plot 3: YoY growth of ML (bar chart)
    # ------------------------------------------------------------------
    # Drop first row (NaN from pct_change)
    growth_df = df.dropna(subset=["ml_growth_yoy"]).copy()

    fig, ax = plt.subplots(figsize=(12, 6), dpi=150)
    colors = ["#4CAF50" if v >= 0 else "#F44336" for v in growth_df["ml_growth_yoy"]]
    ax.bar(growth_df["year"], growth_df["ml_growth_yoy"], color=colors)
    ax.axhline(y=0, color="black", linewidth=0.8)
    ax.set_xlabel("Year")
    ax.set_ylabel("ML Rate YoY Growth (%)")
    ax.set_title("Year-over-Year Growth of ML Mentions")
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    fig.savefig(output_path / "03_ml_yoy_growth.png")
    plt.close(fig)
    logger.info("Saved 03_ml_yoy_growth.png")

    # ------------------------------------------------------------------
    # Plot 4: ML method breakdown (stacked area)
    # ------------------------------------------------------------------
    ml_counts = df[all_ml].astype(float)
    # Stack method counts: fill NaN with 0 for missing years/methods
    ml_counts = ml_counts.fillna(0)

    fig, ax = plt.subplots(figsize=(12, 6), dpi=150)
    palette = sns.color_palette("tab10", n_colors=len(all_ml))
    ax.stackplot(
        df["year"],
        [ml_counts[col].values for col in all_ml],
        labels=all_ml,
        colors=palette,
        alpha=0.8,
    )
    ax.set_xlabel("Year")
    ax.set_ylabel("Mention Count")
    ax.set_title("ML Method Breakdown Over Time")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(output_path / "04_ml_method_breakdown.png")
    plt.close(fig)
    logger.info("Saved 04_ml_method_breakdown.png")

    # ------------------------------------------------------------------
    # Plot 5: Biomarker breakdown (stacked area)
    # ------------------------------------------------------------------
    bio_counts = df[all_biomarkers].fillna(0)

    fig, ax = plt.subplots(figsize=(12, 6), dpi=150)
    palette = sns.color_palette("Set2", n_colors=len(all_biomarkers))
    ax.stackplot(
        df["year"],
        [bio_counts[col].values for col in all_biomarkers],
        labels=all_biomarkers,
        colors=palette,
        alpha=0.8,
    )
    ax.set_xlabel("Year")
    ax.set_ylabel("Mention Count")
    ax.set_title("Biomarker Breakdown Over Time")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(output_path / "05_biomarker_breakdown.png")
    plt.close(fig)
    logger.info("Saved 05_biomarker_breakdown.png")

    logger.info("All plots saved to %s", output_path)


def plot_single_biomarker_vs_ml(df: pd.DataFrame, biomarker_name: str = "cholesterol", output_dir: str = "data/figures/") -> None:
    """
    Dla wskazanego biomarkera rysuje:
    - Linię jego rocznego wskaźnika wzmianek (per 1000 abstraktów, wygładzona 3-letnią średnią)
    - Nałożoną linię łącznego wskaźnika AI/ML
    - Pionowe adnotacje kluczowych kamieni milowych AI/ML
    Zapisuje jako: 06_{biomarker_name}_vs_ml.png
    """
    fig, ax = plt.subplots(figsize=(14, 6))

    ax.plot(df["year"], df["ml_rate_smooth"],
            label="AI/ML (łącznie)", color="#FF5722", linewidth=2.5)
    ax.fill_between(df["year"], df["ml_rate_smooth"], alpha=0.1, color="#FF5722")

    biomarker_col = f"{biomarker_name}_rate_smooth"
    if biomarker_col in df.columns:
        ax.plot(df["year"], df[biomarker_col],
                label=biomarker_name.title(), color="#2196F3", linewidth=2.5)
        ax.fill_between(df["year"], df[biomarker_col], alpha=0.1, color="#2196F3")

    for year, label in ML_MILESTONES.items():
        ax.axvline(x=year, color="gray", linestyle="--", alpha=0.5, linewidth=1)
        ax.text(year + 0.2, ax.get_ylim()[1] * 0.92, label,
                fontsize=7.5, color="gray", rotation=90, va="top")

    ax.set_title(f"{biomarker_name.title()} vs AI/ML — trendy w czasie (per 1000 abstraktów)")
    ax.set_xlabel("Rok")
    ax.set_ylabel("Wskaźnik wzmianek (per 1000 abstraktów)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    output_path = output_path / f"06_{biomarker_name}_vs_ml.png"
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logging.info(f"Zapisano wykres: {output_path}")
