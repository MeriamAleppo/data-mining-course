"""
Exploratory Data Analysis (EDA) Module
Provides statistical distribution diagnostics, skewness/kurtosis calculators,
target imbalance diagnostics, and standardized publication-quality visualizations.
"""

from typing import List, Tuple, Optional
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns

from src.config import NUMERIC_COLUMNS, TARGET_COLUMN
from src.logger import get_logger

logger = get_logger("EDA")


def calculate_distribution_metrics(df: pd.DataFrame, numeric_cols: List[str] = NUMERIC_COLUMNS) -> pd.DataFrame:
    """
    Computes rigorous distribution parameters including Skewness, Kurtosis,
    and D'Agostino-Pearson normality tests.

    Parameters
    ----------
    df : pd.DataFrame
        Input data.
    numeric_cols : List[str]
        Numeric features to evaluate.

    Returns
    -------
    pd.DataFrame
        Summary table with skewness, kurtosis, and p-values for normality.
    """
    metrics = []
    for col in numeric_cols:
        series = df[col].dropna()
        skew_val = stats.skew(series)
        kurt_val = stats.kurtosis(series)
        norm_stat, p_val = stats.normaltest(series) if len(series) >= 20 else (np.nan, np.nan)

        skew_classification = "Near Normal"
        if skew_val > 1.0:
            skew_classification = "Highly Positively Skewed"
        elif skew_val > 0.5:
            skew_classification = "Moderately Positively Skewed"
        elif skew_val < -1.0:
            skew_classification = "Highly Negatively Skewed"
        elif skew_val < -0.5:
            skew_classification = "Moderately Negatively Skewed"

        metrics.append({
            "Feature": col,
            "Mean": round(series.mean(), 2),
            "Std": round(series.std(), 2),
            "Median": round(series.median(), 2),
            "IQR": round(series.quantile(0.75) - series.quantile(0.25), 2),
            "Skewness": round(skew_val, 4),
            "Kurtosis": round(kurt_val, 4),
            "Normality_p_val": round(p_val, 6),
            "Distribution_Shape": skew_classification
        })

    result_df = pd.DataFrame(metrics)
    logger.info("Calculated distribution metrics across numeric columns.")
    return result_df


def plot_target_distribution(
    df: pd.DataFrame,
    target_col: str = TARGET_COLUMN,
    figsize: Tuple[int, int] = (14, 5)
) -> plt.Figure:
    """
    Visualizes target class distribution with a dual plot:
    1. Seaborn Countplot with absolute counts and percentages
    2. Donut/Pie Chart with slice proportions and imbalance ratio annotation
    """
    counts = df[target_col].value_counts()
    props = df[target_col].value_counts(normalize=True) * 100
    n_neg = counts.get("No", counts.get(0, 0))
    n_pos = counts.get("Yes", counts.get(1, 0))
    imbalance_ratio = max(n_neg, n_pos) / max(min(n_neg, n_pos), 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    palette = {"No": "#2b5c8f", "Yes": "#d9534f", 0: "#2b5c8f", 1: "#d9534f"}

    # Subplot 1: Countplot
    sns.countplot(data=df, x=target_col, palette=palette, ax=ax1, edgecolor="black", linewidth=1.2)
    ax1.set_title(f"Target Distribution: '{target_col}' (N = {len(df):,})", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Retention Status", fontsize=10)
    ax1.set_ylabel("Customer Count", fontsize=10)
    ax1.grid(axis="y", linestyle="--", alpha=0.5)

    for p in ax1.patches:
        height = p.get_height()
        pct = (height / len(df)) * 100
        ax1.annotate(
            f"{int(height):,} ({pct:.1f}%)",
            (p.get_x() + p.get_width() / 2., height),
            ha="center", va="bottom", fontsize=10, fontweight="bold",
            xytext=(0, 4), textcoords="offset points"
        )

    # Subplot 2: Donut Chart
    labels = [str(k) for k in counts.index]
    colors = [palette.get(k, "#2b5c8f") for k in counts.index]
    wedges, texts, autotexts = ax2.pie(
        counts.values,
        labels=labels,
        autopct="%1.1f%%",
        startangle=140,
        colors=colors,
        wedgeprops=dict(width=0.45, edgecolor="black", linewidth=1.2),
        textprops=dict(fontsize=10, fontweight="bold")
    )
    for at in autotexts:
        at.set_color("white")
        at.set_fontsize(11)

    ax2.set_title(
        f"Class Imbalance Ratio: {imbalance_ratio:.2f} : 1\n(Majority / Minority)",
        fontsize=12, fontweight="bold"
    )

    plt.tight_layout()
    return fig


def plot_feature_distributions(
    df: pd.DataFrame,
    numeric_cols: List[str] = NUMERIC_COLUMNS,
    target_col: Optional[str] = TARGET_COLUMN,
    figsize: Tuple[int, int] = (16, 4)
) -> plt.Figure:
    """
    Generates standardized Histograms with overlaid Kernel Density Estimates (KDE)
    and Skewness annotations.
    """
    fig, axes = plt.subplots(1, len(numeric_cols), figsize=figsize)
    if len(numeric_cols) == 1:
        axes = [axes]

    palette = {"No": "#2b5c8f", "Yes": "#d9534f", 0: "#2b5c8f", 1: "#d9534f"} if target_col and target_col in df.columns else None

    for i, col in enumerate(numeric_cols):
        ax = axes[i]
        skew_val = stats.skew(df[col].dropna())

        if target_col and target_col in df.columns:
            sns.histplot(
                data=df, x=col, hue=target_col, kde=True,
                palette=palette, alpha=0.45, ax=ax, bins=30, edgecolor="none"
            )
        else:
            sns.histplot(df[col], kde=True, color="#2b5c8f", ax=ax, bins=30)

        ax.set_title(f"{col} Distribution\n(Skewness = {skew_val:+.3f})", fontsize=11, fontweight="bold")
        ax.set_xlabel(col, fontsize=10)
        ax.set_ylabel("Frequency", fontsize=10)
        ax.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    return fig


def plot_bivariate_continuous_boxplots(
    df: pd.DataFrame,
    numeric_cols: List[str] = NUMERIC_COLUMNS,
    target_col: str = TARGET_COLUMN,
    figsize: Tuple[int, int] = (16, 4)
) -> plt.Figure:
    """
    Renders side-by-side boxplots and violin plots showing separation
    of continuous features across Churn classes.
    """
    fig, axes = plt.subplots(1, len(numeric_cols), figsize=figsize)
    if len(numeric_cols) == 1:
        axes = [axes]

    palette = {"No": "#2b5c8f", "Yes": "#d9534f", 0: "#2b5c8f", 1: "#d9534f"}

    for i, col in enumerate(numeric_cols):
        ax = axes[i]
        sns.boxplot(
            data=df, x=target_col, y=col,
            palette=palette, ax=ax, width=0.45,
            boxprops=dict(alpha=0.8),
            showmeans=True,
            meanprops={"marker": "o", "markerfacecolor": "yellow", "markeredgecolor": "black"}
        )
        ax.set_title(f"{col} by Churn Status", fontsize=11, fontweight="bold")
        ax.set_xlabel("Churn", fontsize=10)
        ax.set_ylabel(col, fontsize=10)
        ax.grid(axis="y", linestyle="--", alpha=0.5)

    plt.tight_layout()
    return fig


def plot_categorical_churn_breakdown(
    df: pd.DataFrame,
    cat_cols: List[str],
    target_col: str = TARGET_COLUMN,
    figsize: Tuple[int, int] = (16, 5)
) -> plt.Figure:
    """
    Visualizes categorical features stratified by Churn status with count and percentage annotations.
    """
    n_cols = len(cat_cols)
    fig, axes = plt.subplots(1, n_cols, figsize=figsize)
    if n_cols == 1:
        axes = [axes]

    palette = {"No": "#2b5c8f", "Yes": "#d9534f", 0: "#2b5c8f", 1: "#d9534f"}

    for i, col in enumerate(cat_cols):
        ax = axes[i]
        sns.countplot(
            data=df, x=col, hue=target_col,
            palette=palette, ax=ax, edgecolor="black", linewidth=0.8
        )
        ax.set_title(f"Churn by {col}", fontsize=11, fontweight="bold")
        ax.set_xlabel(col, fontsize=10)
        ax.set_ylabel("Count", fontsize=10)
        ax.tick_params(axis="x", rotation=25)
        ax.grid(axis="y", linestyle="--", alpha=0.5)

        # Annotate counts
        for p in ax.patches:
            height = p.get_height()
            if height and height > 0:
                ax.annotate(
                    f"{int(height)}",
                    (p.get_x() + p.get_width() / 2., height),
                    ha="center", va="bottom", fontsize=8, xytext=(0, 2),
                    textcoords="offset points"
                )

    plt.tight_layout()
    return fig


def plot_correlation_matrix(
    df: pd.DataFrame,
    numeric_cols: List[str] = NUMERIC_COLUMNS,
    figsize: Tuple[int, int] = (7, 5)
) -> plt.Figure:
    """
    Renders correlation matrix heatmap using Pearson's r with masked upper triangle.
    """
    corr = df[numeric_cols].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        corr, mask=mask, annot=True, fmt=".3f",
        cmap="coolwarm", vmin=-1, vmax=1, square=True,
        linewidths=1, cbar_kws={"shrink": 0.8}, ax=ax
    )
    ax.set_title("Pearson Correlation Matrix (Continuous Features)", fontsize=12, fontweight="bold")
    plt.tight_layout()
    return fig
