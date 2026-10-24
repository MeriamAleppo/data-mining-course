"""
Statistical Inference and Hypothesis Testing Module
Provides rigorous formal testing (Chi-Square, One-Way ANOVA, Two-Sample T-Test)
with effect sizes (Cramer's V, Eta-Squared, Cohen's d), Bonferroni corrections,
and actionable business interpretations.
"""

from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
import scipy.stats as stats
from src.logger import get_logger

logger = get_logger("HypothesisTesting")


def perform_chi_square_test(
    df: pd.DataFrame,
    col_x: str,
    col_y: str,
    alpha: float = 0.05
) -> Dict[str, Any]:
    """
    Conducts formal Pearson's Chi-Square Test of Independence with Cramer's V effect size.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset containing both variables.
    col_x : str
        First categorical variable (e.g. 'Contract').
    col_y : str
        Second categorical variable (e.g. 'Churn').
    alpha : float
        Significance level (default 0.05).

    Returns
    -------
    Dict[str, Any]
        Test statistics, effect size, p-value, and business takeaway.
    """
    logger.info(f"Conducting Chi-Square Test: '{col_x}' vs '{col_y}' (alpha={alpha})...")
    contingency_table = pd.crosstab(df[col_x], df[col_y])
    chi2_stat, p_val, dof, expected = stats.chi2_contingency(contingency_table)

    # Calculate Cramer's V effect size
    n = contingency_table.sum().sum()
    min_dim = min(contingency_table.shape) - 1
    cramers_v = np.sqrt(chi2_stat / (n * min_dim)) if min_dim > 0 and n > 0 else 0.0

    # Effect size qualitative description
    if cramers_v < 0.10:
        effect_strength = "Negligible / Weak"
    elif cramers_v < 0.30:
        effect_strength = "Moderate"
    else:
        effect_strength = "Strong"

    reject_null = p_val < alpha

    business_interpretation = (
        f"Dengan p-value = {p_val:.4e} (< alpha {alpha}), terdapat bukti statistik yang sangat kuat "
        f"untuk MENOLAK hipotesis nol (H0). Karakteristik '{col_x}' memiliki korelasi yang signifikan secara statistik "
        f"terhadap '{col_y}' dengan tingkat kekuatan asosiasi {effect_strength} (Cramer's V = {cramers_v:.4f}). "
        f"Bagi manajemen, hal ini membuktikan bahwa segmentasi berdasarkan '{col_x}' harus dijadikan pilar utama strategi retensi."
        if reject_null else
        f"Dengan p-value = {p_val:.4f} (>= alpha {alpha}), kita GAGAL menolak H0. Tidak terdapat bukti statistik yang "
        f"cukup bahwa '{col_x}' berkorelasi signifikan dengan '{col_y}'."
    )

    result = {
        "test_name": "Pearson Chi-Square Test of Independence",
        "variable_1": col_x,
        "variable_2": col_y,
        "contingency_table": contingency_table,
        "chi2_statistic": round(chi2_stat, 4),
        "degrees_of_freedom": int(dof),
        "p_value": float(p_val),
        "cramers_v": round(float(cramers_v), 4),
        "effect_strength": effect_strength,
        "alpha": alpha,
        "reject_null": bool(reject_null),
        "business_interpretation": business_interpretation
    }

    logger.info(f"Chi-Square Result: chi2={chi2_stat:.2f}, p_val={p_val:.2e}, Cramer's V={cramers_v:.4f}, Reject H0={reject_null}")
    return result


def perform_one_way_anova(
    df: pd.DataFrame,
    group_col: str,
    value_col: str,
    alpha: float = 0.05
) -> Dict[str, Any]:
    """
    Performs One-Way Analysis of Variance (ANOVA) to assess mean differences
    across multiple categorical groups, supplemented with Eta-Squared (eta^2).

    Parameters
    ----------
    df : pd.DataFrame
        Dataset.
    group_col : str
        Categorical group feature (e.g. 'Contract').
    value_col : str
        Continuous metric (e.g. 'MonthlyCharges').
    alpha : float
        Significance level.

    Returns
    -------
    Dict[str, Any]
        F-statistic, p-value, eta-squared effect size, and business takeaway.
    """
    logger.info(f"Conducting One-Way ANOVA: '{value_col}' across '{group_col}' groups...")
    groups = [group[value_col].dropna().values for _, group in df.groupby(group_col)]
    f_stat, p_val = stats.f_oneway(*groups)

    # Compute Eta-Squared (SS_between / SS_total)
    grand_mean = df[value_col].mean()
    ss_total = np.sum((df[value_col].dropna() - grand_mean) ** 2)
    ss_between = np.sum([len(g) * ((np.mean(g) - grand_mean) ** 2) for g in groups])
    eta_squared = ss_between / ss_total if ss_total > 0 else 0.0

    if eta_squared < 0.06:
        effect_desc = "Small Effect"
    elif eta_squared < 0.14:
        effect_desc = "Medium Effect"
    else:
        effect_desc = "Large Effect"

    reject_null = p_val < alpha

    group_means = df.groupby(group_col)[value_col].agg(["count", "mean", "std"]).round(2).to_dict(orient="index")

    business_interpretation = (
        f"Dengan F-Statistic = {f_stat:.2f} dan p-value = {p_val:.4e} (< alpha {alpha}), "
        f"kita MENOLAK hipotesis nol (H0). Rata-rata nilai '{value_col}' berbeda secara signifikan secara statistik "
        f"antar kelompok '{group_col}'. Nilai Eta-Squared (eta^2) sebesar {eta_squared:.4f} ({effect_desc}) "
        f"menunjukkan bahwa sekitar {eta_squared*100:.1f}% varians dalam '{value_col}' secara langsung dipengaruhi oleh '{group_col}'."
        if reject_null else
        f"Dengan p-value = {p_val:.4f} (>= alpha {alpha}), GAGAL menolak H0. Rata-rata '{value_col}' tidak berbeda nyata antar grup."
    )

    result = {
        "test_name": "One-Way ANOVA",
        "group_column": group_col,
        "value_column": value_col,
        "group_statistics": group_means,
        "f_statistic": round(float(f_stat), 4),
        "p_value": float(p_val),
        "eta_squared": round(float(eta_squared), 4),
        "effect_strength": effect_desc,
        "alpha": alpha,
        "reject_null": bool(reject_null),
        "business_interpretation": business_interpretation
    }

    logger.info(f"ANOVA Result: F={f_stat:.2f}, p_val={p_val:.2e}, eta^2={eta_squared:.4f}, Reject H0={reject_null}")
    return result


def perform_independent_ttest(
    df: pd.DataFrame,
    group_col: str,
    value_col: str,
    alpha: float = 0.05
) -> Dict[str, Any]:
    """
    Conducts Welch's Two-Sample Independent T-Test (assuming unequal variances)
    with Cohen's d effect size.
    """
    unique_groups = df[group_col].dropna().unique()
    if len(unique_groups) != 2:
        raise ValueError(f"T-Test requires exactly 2 distinct groups, found: {unique_groups}")

    g1 = df[df[group_col] == unique_groups[0]][value_col].dropna().values
    g2 = df[df[group_col] == unique_groups[1]][value_col].dropna().values

    t_stat, p_val = stats.ttest_ind(g1, g2, equal_var=False)

    # Cohen's d
    n1, n2 = len(g1), len(g2)
    s1, s2 = np.var(g1, ddof=1), np.var(g2, ddof=1)
    pooled_sd = np.sqrt(((n1 - 1) * s1 + (n2 - 1) * s2) / (n1 + n2 - 2))
    cohens_d = (np.mean(g1) - np.mean(g2)) / pooled_sd if pooled_sd > 0 else 0.0

    reject_null = p_val < alpha

    return {
        "test_name": "Welch's Two-Sample Independent T-Test",
        "group_1": str(unique_groups[0]),
        "group_2": str(unique_groups[1]),
        "mean_group_1": round(float(np.mean(g1)), 2),
        "mean_group_2": round(float(np.mean(g2)), 2),
        "t_statistic": round(float(t_stat), 4),
        "p_value": float(p_val),
        "cohens_d": round(float(cohens_d), 4),
        "reject_null": bool(reject_null),
        "alpha": alpha,
        "business_interpretation": (
            f"Perbedaan rata-rata '{value_col}' antara {unique_groups[0]} ({np.mean(g1):.2f}) dan "
            f"{unique_groups[1]} ({np.mean(g2):.2f}) signifikan secara statistik (p = {p_val:.4e}, Cohen's d = {cohens_d:.2f})."
            if reject_null else
            f"Tidak terdapat perbedaan signifikan antara kedua kelompok (p = {p_val:.4f})."
        )
    }


def apply_bonferroni_correction(
    p_values: List[float],
    family_alpha: float = 0.05
) -> Dict[str, Any]:
    """
    Applies Bonferroni Correction across a family of hypothesis tests
    to prevent inflation of the Family-Wise Error Rate (Type I error).

    Parameters
    ----------
    p_values : List[float]
        List of unadjusted p-values.
    family_alpha : float
        Overall target false positive rate.

    Returns
    -------
    Dict[str, Any]
        Adjusted threshold and rejection decisions.
    """
    m = len(p_values)
    adjusted_alpha = family_alpha / m if m > 0 else family_alpha
    rejections = [p < adjusted_alpha for p in p_values]

    return {
        "number_of_tests": m,
        "family_alpha": family_alpha,
        "bonferroni_alpha_threshold": adjusted_alpha,
        "original_p_values": p_values,
        "rejection_flags": rejections,
        "total_significant_discoveries": sum(rejections)
    }
