"""Visualization functions for aggregated PubMed trend data."""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from .entity_lists import ALL_BIOMARKER_COLUMNS, ALL_ML_METHOD_COLUMNS

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

# Grupy metod dla wykresów „dla laików” (suma wzmianek w grupie; jeden abstrakt może liczyć się w wielu).
ML_FAMILY_GROUPS: dict[str, list[str]] = {
    "classic_statistics": ["linear_regression", "cox_model", "logistic_regression"],
    "traditional_ml": [
        "random_forest",
        "SVM",
        "naive_bayes",
        "decision_tree",
        "clustering",
    ],
    "boosting": ["XGBoost", "gradient_boosting", "lightgbm", "catboost"],
    "neural_deep": ["neural_network", "deep_learning"],
    "transformers": ["transformer"],
}

ML_FAMILY_LABELS_PL: dict[str, str] = {
    "classic_statistics": "Statystyka klasyczna (regresja, Cox, logit)",
    "traditional_ml": "ML klasyczne (RF, SVM, drzewa…)",
    "boosting": "Boosting (XGBoost, GBM, LightGBM…)",
    "neural_deep": "Sieci i deep learning",
    "transformers": "Transformery / LLM (słowa w tekście)",
}


def _smoothing_window_years(n_years: int) -> int:
    """Okno wygładzenia zależne od długości szeregu — przy małej próbie mniejsze okno."""
    if n_years <= 6:
        return 3
    if n_years <= 14:
        return 5
    return 7


def _smooth_columns(df: pd.DataFrame, cols: list[str], window: int) -> pd.DataFrame:
    out = df[cols].astype(float).fillna(0).copy()
    w = max(1, min(window, len(out)))
    if w <= 1:
        return out
    return out.rolling(window=w, center=True, min_periods=1).mean()


def _family_sum_matrix(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Roczne sumy wzmianek per grupa (kolumny = klucze ML_FAMILY_GROUPS)."""
    rows = []
    keys_order: list[str] = []
    for key, members in ML_FAMILY_GROUPS.items():
        present = [c for c in members if c in df.columns]
        if not present:
            continue
        keys_order.append(key)
        rows.append(df[present].astype(float).fillna(0).sum(axis=1))
    if not rows:
        return pd.DataFrame(), []
    mat = pd.concat(rows, axis=1)
    mat.columns = keys_order
    return mat, keys_order


def _corpus_caption(df: pd.DataFrame) -> str:
    if "total_abstracts" not in df.columns:
        return ""
    mean_y = df["total_abstracts"].mean()
    tot = df["total_abstracts"].sum()
    return (
        f"Korpus: średnio ~{mean_y:.0f} abstraktów/rok (łącznie {tot:.0f}). "
        f"Mała próba roczna = większe wahania; wykresy używają wygładzenia, by pokazać trend."
    )


def generate_all_plots(df: pd.DataFrame, output_dir: str = "data/figures/") -> None:
    """Generate and save trend plots (PNG).

    m.in. wykresy **uproszczone dla laików**: grupy metad (``04``, ``07``, ``08``)
    z wygładzeniem kroczącym przy małej próbie rocznej; szczegóły per-metoda w ``04b``.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    all_ml = [c for c in ALL_ML_METHOD_COLUMNS if c in df.columns]
    all_biomarkers = [c for c in ALL_BIOMARKER_COLUMNS if c in df.columns]
    if not all_ml:
        all_ml = list(ALL_ML_METHOD_COLUMNS)
    if not all_biomarkers:
        all_biomarkers = list(ALL_BIOMARKER_COLUMNS)

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
    win_ix = _smoothing_window_years(len(df))
    pen_smooth = df["ai_penetration_index"].rolling(
        window=win_ix, center=True, min_periods=1
    ).mean()
    ax.plot(
        df["year"],
        pen_smooth,
        color="#E91E63",
        linewidth=2,
        linestyle="--",
        alpha=0.9,
        label=f"Wygładzony trend ({win_ix}-letnie okno)",
    )
    ax.legend()
    ax.grid(True, alpha=0.3)
    cap = _corpus_caption(df)
    if cap:
        fig.text(0.5, 0.01, cap, ha="center", fontsize=8, color="dimgray")
    plt.tight_layout(rect=(0, 0.06, 1, 1))
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
    ax.set_title("Year-over-Year Growth of ML Mentions (surowe — przy małej próbie bywa chaotyczne)")
    ax.grid(True, alpha=0.3, axis="y")
    cap = _corpus_caption(df)
    if cap:
        fig.text(0.5, 0.01, cap, ha="center", fontsize=8, color="dimgray")
    plt.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(output_path / "03_ml_yoy_growth.png")
    plt.close(fig)
    logger.info("Saved 03_ml_yoy_growth.png")

    win = _smoothing_window_years(len(df))
    fam_mat, fam_keys = _family_sum_matrix(df)
    fam_smooth = _smooth_columns(fam_mat, list(fam_mat.columns), win) if not fam_mat.empty else fam_mat

    # ------------------------------------------------------------------
    # Plot 4: ML — 5 czytelnych grup (wygładzone; trend zamiast „szpilek”)
    # ------------------------------------------------------------------
    if not fam_smooth.empty:
        fig, ax = plt.subplots(figsize=(13, 6.5), dpi=150)
        palette_f = sns.color_palette("Set2", n_colors=len(fam_keys))
        labels_pl = [ML_FAMILY_LABELS_PL.get(k, k) for k in fam_keys]
        ax.stackplot(
            df["year"],
            [fam_smooth[k].values for k in fam_keys],
            labels=labels_pl,
            colors=palette_f,
            alpha=0.88,
        )
        ax.set_xlabel("Rok")
        ax.set_ylabel("Łączna liczba wzmianek w grupie (po wygładzeniu)")
        ax.set_title(
            f"Metody analityczne w czasie — pięć grup (średnia krocząca: {win} lat)\n"
            "Jeden artykuł może zawierać wiele metod; to nie jest „% publikacji”."
        )
        ax.legend(loc="upper left", fontsize=9)
        ax.grid(True, alpha=0.3)
        cap = _corpus_caption(df)
        if cap:
            fig.text(0.5, 0.01, cap, ha="center", fontsize=8, color="dimgray")
        plt.tight_layout(rect=(0, 0.07, 1, 1))
        fig.savefig(output_path / "04_ml_method_breakdown.png")
        plt.close(fig)
        logger.info("Saved 04_ml_method_breakdown.png (family groups, smoothed)")

        # ------------------------------------------------------------------
        # Plot 7: te same grupy — linie (najłatwiejsze do czytania)
        # ------------------------------------------------------------------
        fig, ax = plt.subplots(figsize=(13, 6), dpi=150)
        for i, key in enumerate(fam_keys):
            ax.plot(
                df["year"],
                fam_smooth[key],
                label=ML_FAMILY_LABELS_PL.get(key, key),
                linewidth=2.6,
                color=palette_f[i % len(palette_f)],
            )
        ax.set_xlabel("Rok")
        ax.set_ylabel(f"Liczba wzmianek (średnia krocząca {win} lat)")
        ax.set_title("Trendy metod — uproszczone grupy (dla osób spoza ML)")
        ax.legend(loc="best", fontsize=9)
        ax.grid(True, alpha=0.3)
        if cap:
            fig.text(0.5, 0.01, cap, ha="center", fontsize=8, color="dimgray")
        plt.tight_layout(rect=(0, 0.06, 1, 1))
        fig.savefig(output_path / "07_ml_families_easy_read.png")
        plt.close(fig)
        logger.info("Saved 07_ml_families_easy_read.png")

        # ------------------------------------------------------------------
        # Plot 8: skład % — jaką część wszystkich wzmianek o metodach stanowi każda grupa
        # ------------------------------------------------------------------
        row_tot = fam_smooth.sum(axis=1).replace(0, float("nan"))
        fam_pct = fam_smooth.div(row_tot, axis=0) * 100.0
        fam_pct = fam_pct.fillna(0.0)
        fig, ax = plt.subplots(figsize=(13, 6.5), dpi=150)
        ax.stackplot(
            df["year"],
            [fam_pct[k].values for k in fam_keys],
            labels=labels_pl,
            colors=palette_f,
            alpha=0.88,
        )
        ax.set_xlabel("Rok")
        ax.set_ylabel("Udział wzmianek między grupami (%)")
        ax.set_title(
            "Zmiana „miksu” metod w czasie (%) — suma warstw = 100% w każdym roku\n"
            "Pokazuje np. wzrost deep learningu / boosting względem statystyki klasycznej, "
            "nawet gdy mała próba rozmywa skalę bezwzględną."
        )
        ax.legend(loc="upper left", fontsize=9)
        ax.set_ylim(0, 100)
        ax.grid(True, alpha=0.3)
        if cap:
            fig.text(0.5, 0.01, cap, ha="center", fontsize=8, color="dimgray")
        plt.tight_layout(rect=(0, 0.08, 1, 1))
        fig.savefig(output_path / "08_ml_family_mix_percent.png")
        plt.close(fig)
        logger.info("Saved 08_ml_family_mix_percent.png")

    # ------------------------------------------------------------------
    # Plot 4b: szczegółowy stos (wszystkie metody), też wygładzony — dla ekspertów
    # ------------------------------------------------------------------
    ml_counts = _smooth_columns(df, all_ml, win)
    fig, ax = plt.subplots(figsize=(13, 6.5), dpi=150)
    palette = sns.color_palette("tab10", n_colors=len(all_ml))
    ax.stackplot(
        df["year"],
        [ml_counts[col].values for col in all_ml],
        labels=all_ml,
        colors=palette,
        alpha=0.8,
    )
    ax.set_xlabel("Year")
    ax.set_ylabel("Mention count (smoothed)")
    ax.set_title(f"Wszystkie metody (wygładzenie {win} lat) — mniej szumu niż surowe liczby")
    ax.legend(loc="upper left", fontsize=7)
    ax.grid(True, alpha=0.3)
    cap = _corpus_caption(df)
    if cap:
        fig.text(0.5, 0.01, cap, ha="center", fontsize=8, color="dimgray")
    plt.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(output_path / "04b_ml_methods_detail_smoothed.png")
    plt.close(fig)
    logger.info("Saved 04b_ml_methods_detail_smoothed.png")

    # ------------------------------------------------------------------
    # Plot 5: Biomarker breakdown (stacked area)
    # ------------------------------------------------------------------
    bio_counts = _smooth_columns(df, all_biomarkers, win)

    fig, ax = plt.subplots(figsize=(13, 6.5), dpi=150)
    palette = sns.color_palette("Set2", n_colors=len(all_biomarkers))
    ax.stackplot(
        df["year"],
        [bio_counts[col].values for col in all_biomarkers],
        labels=all_biomarkers,
        colors=palette,
        alpha=0.8,
    )
    ax.set_xlabel("Year")
    ax.set_ylabel("Mention count (smoothed)")
    ax.set_title(f"Biomarkery w czasie (średnia krocząca {win} lat)")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.3)
    cap = _corpus_caption(df)
    if cap:
        fig.text(0.5, 0.01, cap, ha="center", fontsize=8, color="dimgray")
    plt.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(output_path / "05_biomarker_breakdown.png")
    plt.close(fig)
    logger.info("Saved 05_biomarker_breakdown.png")

    logger.info("All plots saved to %s", output_path)
    generate_interactive_plots(df, str(output_path))


def generate_interactive_plots(df: pd.DataFrame, output_dir: str) -> None:
    """Eksport interaktywnych wykresów HTML (Plotly): zoom, hover, legenda."""
    try:
        import plotly.graph_objects as go
    except ImportError:
        logger.warning("plotly nie jest zainstalowane — pomijam wykresy HTML.")
        return

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    all_ml = [c for c in ALL_ML_METHOD_COLUMNS if c in df.columns]
    all_bio = [c for c in ALL_BIOMARKER_COLUMNS if c in df.columns]
    if not all_ml:
        all_ml = list(ALL_ML_METHOD_COLUMNS)
    if not all_bio:
        all_bio = list(ALL_BIOMARKER_COLUMNS)

    fig1 = go.Figure()
    fig1.add_trace(
        go.Scatter(
            x=df["year"],
            y=df["biomarker_rate_smooth"],
            name="Biomarker rate (smoothed)",
            mode="lines",
            line=dict(color="#2196F3", width=2),
        )
    )
    fig1.add_trace(
        go.Scatter(
            x=df["year"],
            y=df["ml_rate_smooth"],
            name="ML / methods rate (smoothed)",
            mode="lines",
            line=dict(color="#FF5722", width=2),
        )
    )
    fig1.update_layout(
        title="AI/ML vs biomarker mentions (per 1000 abstracts, smoothed)",
        xaxis_title="Year",
        yaxis_title="Per 1000 abstracts",
        template="plotly_white",
        hovermode="x unified",
    )
    fig1.write_html(out / "interactive_01_ai_vs_biomarkers.html", include_plotlyjs="cdn")

    fig2 = go.Figure()
    fig2.add_trace(
        go.Scatter(
            x=df["year"],
            y=df["ai_penetration_index"],
            mode="lines",
            line=dict(color="#9C27B0", width=2),
            name="AI penetration index",
        )
    )
    win_i = _smoothing_window_years(len(df))
    fig2.add_trace(
        go.Scatter(
            x=df["year"],
            y=df["ai_penetration_index"]
            .rolling(window=win_i, center=True, min_periods=1)
            .mean(),
            mode="lines",
            line=dict(color="#E91E63", width=2, dash="dash"),
            name=f"Trend wygładzony ({win_i} lat)",
        )
    )
    fig2.add_hline(y=1.0, line_dash="dash", line_color="gray", annotation_text="Parity")
    fig2.update_layout(
        title="AI penetration index (ML rate / biomarker rate)",
        xaxis_title="Year",
        yaxis_title="Index",
        template="plotly_white",
    )
    fig2.write_html(out / "interactive_02_ai_penetration_index.html", include_plotlyjs="cdn")

    growth_df = df.dropna(subset=["ml_growth_yoy"]).copy()
    fig3 = go.Figure(
        data=[
            go.Bar(
                x=growth_df["year"],
                y=growth_df["ml_growth_yoy"],
                marker_color=["#4CAF50" if v >= 0 else "#F44336" for v in growth_df["ml_growth_yoy"]],
            )
        ]
    )
    fig3.update_layout(
        title="Year-over-year growth of ML / method mention rate (%)",
        xaxis_title="Year",
        yaxis_title="YoY %",
        template="plotly_white",
    )
    fig3.write_html(out / "interactive_03_ml_yoy_growth.html", include_plotlyjs="cdn")

    win = _smoothing_window_years(len(df))
    fam_mat_i, fam_keys_i = _family_sum_matrix(df)
    fam_s_i = (
        _smooth_columns(fam_mat_i, list(fam_mat_i.columns), win)
        if not fam_mat_i.empty
        else fam_mat_i
    )
    fig4 = go.Figure()
    if not fam_s_i.empty:
        labels_pl = [ML_FAMILY_LABELS_PL.get(k, k) for k in fam_keys_i]
        for col, lab in zip(fam_keys_i, labels_pl):
            fig4.add_trace(
                go.Scatter(
                    x=df["year"],
                    y=fam_s_i[col],
                    name=lab,
                    stackgroup="one",
                    mode="lines",
                    line=dict(width=0.5),
                )
            )
        fig4.update_layout(
            title=f"Metody — 5 grup (wygładzenie {win} lat), interaktywnie",
            xaxis_title="Rok",
            yaxis_title="Wzmianek w grupie (wygładzone)",
            template="plotly_white",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
    fig4.write_html(out / "interactive_04_ml_method_breakdown.html", include_plotlyjs="cdn")

    if not fam_s_i.empty:
        fig7 = go.Figure()
        for k in fam_keys_i:
            fig7.add_trace(
                go.Scatter(
                    x=df["year"],
                    y=fam_s_i[k],
                    name=ML_FAMILY_LABELS_PL.get(k, k),
                    mode="lines",
                )
            )
        fig7.update_layout(
            title="Trendy — pięć grup metod (wygładzone)",
            xaxis_title="Rok",
            yaxis_title="Liczba wzmianek",
            template="plotly_white",
        )
        fig7.write_html(out / "interactive_07_ml_families.html", include_plotlyjs="cdn")

        row_tot = fam_s_i.sum(axis=1).replace(0, float("nan"))
        fam_pct_i = fam_s_i.div(row_tot, axis=0) * 100.0
        fam_pct_i = fam_pct_i.fillna(0.0)
        fig8 = go.Figure()
        for col, lab in zip(fam_keys_i, labels_pl):
            fig8.add_trace(
                go.Scatter(
                    x=df["year"],
                    y=fam_pct_i[col],
                    name=lab,
                    stackgroup="pct",
                    mode="lines",
                    line=dict(width=0.5),
                )
            )
        fig8.update_layout(
            title="Udział grup w „miksie” wzmianek (%) — 100% na rok",
            xaxis_title="Rok",
            yaxis_title="Procent",
            template="plotly_white",
        )
        fig8.write_html(out / "interactive_08_ml_mix_percent.html", include_plotlyjs="cdn")

    ml_counts = _smooth_columns(df, all_ml, win)
    fig4b = go.Figure()
    for col in all_ml:
        fig4b.add_trace(
            go.Scatter(
                x=df["year"],
                y=ml_counts[col],
                name=col,
                stackgroup="detail",
                mode="lines",
                line=dict(width=0.5),
            )
        )
    fig4b.update_layout(
        title=f"Wszystkie metody (wygładzenie {win} lat)",
        xaxis_title="Year",
        yaxis_title="Count",
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig4b.write_html(out / "interactive_04b_ml_detail_smoothed.html", include_plotlyjs="cdn")

    bio_counts = _smooth_columns(df, all_bio, win)
    fig5 = go.Figure()
    for col in all_bio:
        fig5.add_trace(
            go.Scatter(
                x=df["year"],
                y=bio_counts[col],
                name=col,
                stackgroup="one",
                mode="lines",
                line=dict(width=0.5),
            )
        )
    fig5.update_layout(
        title="Biomarker mentions — stacked counts over time",
        xaxis_title="Year",
        yaxis_title="Count",
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig5.write_html(out / "interactive_05_biomarker_breakdown.html", include_plotlyjs="cdn")

    logger.info("Zapisano interaktywne wykresy Plotly w %s", out)


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
