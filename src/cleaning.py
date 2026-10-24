"""
Data Cleaning and Quality Enhancement Module
Implements advanced MICE imputation, Tukey IQR, OLS Studentized residuals,
and Multivariate Isolation Forest anomaly detection.
"""

from typing import List, Tuple, Dict, Any, Optional
import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer, SimpleImputer, KNNImputer
from sklearn.ensemble import IsolationForest
import statsmodels.api as sm
import statsmodels.formula.api as smf

from src.config import NUMERIC_COLUMNS, RANDOM_STATE, DB_PATH, CLEANED_TABLE, PROCESSED_DATA_DIR
from src.database import save_to_sqlite
from src.caching import save_to_cache, load_from_cache
from src.logger import get_logger

logger = get_logger("DataCleaning")


def compare_imputation_strategies(
    df: pd.DataFrame,
    numeric_cols: List[str] = NUMERIC_COLUMNS,
    target_col: str = "TotalCharges"
) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    """
    Compares Simple Imputation (Median), KNN Imputer (k=5), and MICE (IterativeImputer).
    Evaluates how each technique preserves the distribution parameters (Mean, Std)
    and multivariate correlation with predictor features (tenure & MonthlyCharges).

    Parameters
    ----------
    df : pd.DataFrame
        Input raw dataframe.
    numeric_cols : List[str]
        List of numeric column names.
    target_col : str
        Column with missing values.

    Returns
    -------
    Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]
        - Comparison summary table of statistical metrics.
        - Dictionary of imputed dataframes for each strategy.
    """
    logger.info("Executing comparative imputation benchmark (Simple vs KNN vs MICE)...")
    cache_key = f"imputation_benchmark_{len(df)}_{target_col}"
    cached = load_from_cache(cache_key)
    if cached is not None:
        return cached

    df_num = df[numeric_cols].copy()
    missing_mask = df_num[target_col].isnull()
    num_missing = missing_mask.sum()
    logger.info(f"Detected {num_missing} missing values in '{target_col}'.")

    # 1. Median Imputation
    median_imp = SimpleImputer(strategy="median")
    df_median = df_num.copy()
    df_median[numeric_cols] = median_imp.fit_transform(df_median)

    # 2. KNN Imputation (k=5)
    knn_imp = KNNImputer(n_neighbors=5)
    df_knn = df_num.copy()
    df_knn[numeric_cols] = knn_imp.fit_transform(df_knn)

    # 3. MICE (IterativeImputer with Bayesian Ridge)
    mice_imp = IterativeImputer(max_iter=15, random_state=RANDOM_STATE)
    df_mice = df_num.copy()
    df_mice[numeric_cols] = mice_imp.fit_transform(df_mice)

    # Baseline before imputation (dropna)
    df_clean_base = df_num.dropna()

    def get_metrics(data: pd.DataFrame, label: str) -> Dict[str, Any]:
        corr_tenure = data[target_col].corr(data["tenure"])
        corr_monthly = data[target_col].corr(data["MonthlyCharges"])
        return {
            "Strategy": label,
            "Mean": round(data[target_col].mean(), 2),
            "Std": round(data[target_col].std(), 2),
            "Median": round(data[target_col].median(), 2),
            "Corr_with_tenure": round(corr_tenure, 4),
            "Corr_with_MonthlyCharges": round(corr_monthly, 4),
        }

    metrics = [
        get_metrics(df_clean_base, "Complete Case (Drop NA)"),
        get_metrics(df_median, "Simple Imputer (Median)"),
        get_metrics(df_knn, "KNN Imputer (k=5)"),
        get_metrics(df_mice, "MICE (IterativeImputer)")
    ]

    summary_df = pd.DataFrame(metrics)
    imputed_dfs = {
        "median": df_median,
        "knn": df_knn,
        "mice": df_mice
    }

    result = (summary_df, imputed_dfs)
    save_to_cache(cache_key, result)
    logger.info("Imputation benchmark completed successfully.")
    return result


def apply_mice_imputation(
    df: pd.DataFrame,
    numeric_cols: List[str] = NUMERIC_COLUMNS,
    random_state: int = RANDOM_STATE
) -> pd.DataFrame:
    """
    Applies production-grade MICE (IterativeImputer) to numeric features.

    Parameters
    ----------
    df : pd.DataFrame
        Source DataFrame.
    numeric_cols : List[str]
        Columns to impute jointly.
    random_state : int
        Seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        DataFrame with imputed numeric values.
    """
    df_clean = df.copy()
    imputer = IterativeImputer(max_iter=20, random_state=random_state)
    imputed_values = imputer.fit_transform(df_clean[numeric_cols])
    df_clean[numeric_cols] = imputed_values
    logger.info("Applied MICE imputation across numeric columns.")
    return df_clean


def detect_outliers_iqr(
    df: pd.DataFrame,
    column: str,
    multiplier: float = 1.5
) -> Tuple[pd.Series, float, float]:
    """
    Calculates classical Tukey IQR fences and identifies univariate outliers.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    column : str
        Target column.
    multiplier : float
        Fence multiplier (default 1.5 for mild outliers).

    Returns
    -------
    Tuple[pd.Series, float, float]
        Boolean series (True if outlier), lower bound, upper bound.
    """
    q1 = df[column].quantile(0.25)
    q3 = df[column].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - multiplier * iqr
    upper_bound = q3 + multiplier * iqr
    outlier_mask = (df[column] < lower_bound) | (df[column] > upper_bound)
    logger.info(f"Tukey IQR on '{column}': Bounds=[{lower_bound:.2f}, {upper_bound:.2f}], "
                f"Outliers={outlier_mask.sum()} ({outlier_mask.mean():.2%})")
    return outlier_mask, lower_bound, upper_bound


def detect_outliers_ols_residuals(
    df: pd.DataFrame,
    formula: str = "TotalCharges ~ tenure + MonthlyCharges",
    threshold: float = 2.5
) -> Tuple[pd.Series, pd.Series]:
    """
    Detects multivariate regression anomalies using Studentized Residuals.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    formula : str
        Statsmodels formula string.
    threshold : float
        Absolute threshold on Studentized residuals (typically 2.0 or 2.5).

    Returns
    -------
    Tuple[pd.Series, pd.Series]
        Boolean mask of outliers, and the continuous Studentized residuals.
    """
    model = smf.ols(formula=formula, data=df).fit()
    influence = model.get_influence()
    studentized_residuals = influence.resid_studentized_internal
    outlier_mask = pd.Series(np.abs(studentized_residuals) > threshold, index=df.index)
    logger.info(f"OLS Residuals Outlier Detection: {outlier_mask.sum()} anomalies flagged (|resid| > {threshold}).")
    return outlier_mask, pd.Series(studentized_residuals, index=df.index)


def detect_outliers_isolation_forest(
    df: pd.DataFrame,
    feature_cols: List[str] = NUMERIC_COLUMNS,
    contamination: float = 0.03,
    random_state: int = RANDOM_STATE
) -> Tuple[pd.Series, pd.Series]:
    """
    Identifies multivariate anomalies using Scikit-Learn's Isolation Forest.
    Unlike univariate IQR, Isolation Forest isolates anomalous observations in high-dimensional feature space.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    feature_cols : List[str]
        Numeric features to evaluate.
    contamination : float
        Expected proportion of outliers in the dataset.
    random_state : int
        Seed for reproducibility.

    Returns
    -------
    Tuple[pd.Series, pd.Series]
        Boolean mask (True if outlier), and anomaly scores.
    """
    iso = IsolationForest(
        n_estimators=150,
        contamination=contamination,
        random_state=random_state,
        n_jobs=-1
    )
    preds = iso.fit_predict(df[feature_cols])
    scores = iso.decision_function(df[feature_cols])

    outlier_mask = pd.Series(preds == -1, index=df.index)
    anomaly_scores = pd.Series(scores, index=df.index)

    logger.info(f"Isolation Forest: Contamination={contamination}, "
                f"Flagged Outliers={outlier_mask.sum()} ({outlier_mask.mean():.2%})")
    return outlier_mask, anomaly_scores


def run_complete_cleaning_pipeline(
    df_raw: pd.DataFrame,
    db_path: str = DB_PATH,
    table_name: str = CLEANED_TABLE
) -> pd.DataFrame:
    """
    Executes end-to-end cleaning:
    1. Removes duplicate customer IDs
    2. Imputes numeric missing values using MICE
    3. Analyzes anomalies using Isolation Forest & OLS residuals
    4. Persists the clean dataset to SQLite and Parquet/CSV

    Returns
    -------
    pd.DataFrame
        Cleaned, high-quality DataFrame ready for EDA and Feature Engineering.
    """
    logger.info("Starting complete Data Cleaning Pipeline...")
    df = df_raw.copy()

    # Deduplication
    initial_len = len(df)
    df = df.drop_duplicates(subset=["customerID"]).reset_index(drop=True)
    logger.info(f"Deduplication complete: {initial_len - len(df)} duplicate records removed.")

    # MICE Imputation
    df = apply_mice_imputation(df, numeric_cols=NUMERIC_COLUMNS)

    # Anomaly Detection tag
    _, anomaly_scores = detect_outliers_isolation_forest(df, feature_cols=NUMERIC_COLUMNS)
    df["anomaly_score"] = anomaly_scores

    # Persist to SQLite
    save_to_sqlite(df, table_name=table_name, db_path=db_path, if_exists="replace")

    # Persist processed CSV backup
    csv_out = PROCESSED_DATA_DIR / "telco_cleaned.csv"
    df.to_csv(csv_out, index=False)
    logger.info(f"Cleaned dataset saved to SQLite '{table_name}' and CSV '{csv_out}'.")

    return df
