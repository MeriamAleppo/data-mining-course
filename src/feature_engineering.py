"""
Feature Engineering and Selection Module
Implements automated power transformations (Yeo-Johnson vs Log1p),
domain-specific business feature creation, comparative categorical encoding,
Variance Inflation Factor (VIF) collinearity elimination,
and information-theoretic feature selection.
"""

from typing import List, Dict, Tuple, Optional, Any
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import PowerTransformer, OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.feature_selection import mutual_info_classif
from sklearn.ensemble import RandomForestClassifier
from statsmodels.stats.outliers_influence import variance_inflation_factor

from src.config import (
    NUMERIC_COLUMNS, NOMINAL_COLUMNS, ORDINAL_COLUMNS,
    ORDINAL_MAPPINGS, TARGET_COLUMN, RANDOM_STATE, DB_PATH,
    FEATURED_TABLE, PROCESSED_DATA_DIR
)
from src.database import save_to_sqlite
from src.logger import get_logger

logger = get_logger("FeatureEngineering")


def compare_skewness_remediations(
    df: pd.DataFrame,
    numeric_cols: List[str] = NUMERIC_COLUMNS
) -> pd.DataFrame:
    """
    Compares original skewness with Log1p and Yeo-Johnson transformations
    across continuous numerical features.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    numeric_cols : List[str]
        Continuous numerical columns.

    Returns
    -------
    pd.DataFrame
        Comparison summary table.
    """
    records = []
    for col in numeric_cols:
        series = df[col].dropna()
        orig_skew = stats.skew(series)

        # Log1p Transformation
        log_series = np.log1p(np.maximum(series, 0))
        log_skew = stats.skew(log_series)

        # Yeo-Johnson Power Transformation
        pt = PowerTransformer(method="yeo-johnson", standardize=False)
        yj_vals = pt.fit_transform(df[[col]].dropna()).flatten()
        yj_skew = stats.skew(yj_vals)

        # Determine winner based on distance from zero
        best_method = "Original"
        min_dist = abs(orig_skew)
        if abs(yj_skew) < min_dist:
            min_dist = abs(yj_skew)
            best_method = "Yeo-Johnson"
        if abs(log_skew) < min_dist:
            min_dist = abs(log_skew)
            best_method = "Log1p"

        records.append({
            "Feature": col,
            "Skewness_Original": round(orig_skew, 4),
            "Skewness_Log1p": round(log_skew, 4),
            "Skewness_YeoJohnson": round(yj_skew, 4),
            "Optimal_Method": best_method,
            "Absolute_Reduction_YJ": round(abs(orig_skew) - abs(yj_skew), 4)
        })

    return pd.DataFrame(records)


def auto_skewness_transform(
    df: pd.DataFrame,
    numeric_cols: List[str] = NUMERIC_COLUMNS,
    threshold: float = 0.5,
    method: str = "yeo-johnson"
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, PowerTransformer]]:
    """
    Automatically detects skewed continuous variables (|skewness| > threshold)
    and applies optimal Power Transformations (Yeo-Johnson or Box-Cox).

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    numeric_cols : List[str]
        Continuous columns to inspect.
    threshold : float
        Absolute skewness threshold above which transformation is triggered.
    method : str
        'yeo-johnson' or 'box-cox'. Yeo-Johnson supports zero and negative values.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame, Dict[str, PowerTransformer]]
        - DataFrame with transformed features.
        - Comparison summary table of Skewness Before & After.
        - Dictionary of fitted PowerTransformer objects.
    """
    logger.info(f"Checking continuous features for skewness remediation (threshold={threshold}, method='{method}')...")
    df_transformed = df.copy()
    transformers = {}
    records = []

    for col in numeric_cols:
        orig_skew = stats.skew(df[col].dropna())
        records_entry = {
            "Feature": col,
            "Skewness_Original": round(orig_skew, 4),
            "Exceeds_Threshold": abs(orig_skew) > threshold,
            "Method_Applied": "None",
            "Skewness_Transformed": round(orig_skew, 4),
            "Remediation_Status": "Preserved Original"
        }

        if abs(orig_skew) > threshold:
            transformer = PowerTransformer(method=method, standardize=False)
            transformed_vals = transformer.fit_transform(df[[col]])
            new_col = f"{col}_transformed"
            df_transformed[new_col] = transformed_vals.flatten()
            transformers[col] = transformer

            new_skew = stats.skew(transformed_vals.flatten())
            records_entry["Method_Applied"] = method
            records_entry["Skewness_Transformed"] = round(new_skew, 4)
            records_entry["Remediation_Status"] = (
                f"Skewness reduced by {abs(orig_skew) - abs(new_skew):.3f} points"
            )
            logger.info(f"Remediated '{col}': Skewness went from {orig_skew:+.3f} to {new_skew:+.3f}")

        records.append(records_entry)

    summary_df = pd.DataFrame(records)
    return df_transformed, summary_df, transformers


def create_business_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generates domain-specific derived business features:
    1. Tenure_Cohort & Tenure_Group (0: 0-12m, 1: 13-24m, 2: 25-48m, 3: 49-72m)
    2. Average_Monthly_Charges (TotalCharges / max(tenure, 1))
    3. Ratio_Total_to_Monthly_Charge (TotalCharges / MonthlyCharges)
    4. Flag_High_Risk_Contract (Month-to-month contract & Electronic check payment)
    5. Num_Support_Services (Count of active support/security add-ons)

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.

    Returns
    -------
    pd.DataFrame
        Enriched DataFrame with newly engineered features.
    """
    logger.info("Engineering domain-specific business features...")
    df_feat = df.copy()

    # 1. Tenure Cohorts & Ordinal Grouping
    df_feat["Tenure_Group"] = pd.cut(
        df_feat["tenure"],
        bins=[0, 12, 24, 48, 72],
        labels=[0, 1, 2, 3],
        include_lowest=True
    ).astype(int)

    # 2. Average Historical Monthly Charge
    df_feat["Average_Monthly_Charges"] = (
        df_feat["TotalCharges"] / np.maximum(df_feat["tenure"], 1)
    ).round(2)

    # 3. Ratio of Total Charges to Current Monthly Charge
    df_feat["Ratio_Total_to_Monthly_Charge"] = (
        df_feat["TotalCharges"] / (df_feat["MonthlyCharges"] + 1e-5)
    ).round(2)

    # 4. Flag High Risk Contract (Month-to-month + Electronic check)
    df_feat["Flag_High_Risk_Contract"] = (
        (df_feat["Contract"] == "Month-to-month") &
        (df_feat["PaymentMethod"] == "Electronic check")
    ).astype(int)

    # 5. Active Support & Protection Services
    support_sec = (df_feat["OnlineSecurity"] == "Yes").astype(int)
    support_tech = (df_feat["TechSupport"] == "Yes").astype(int)
    df_feat["Num_Support_Services"] = support_sec + support_tech

    logger.info("Successfully engineered 5 domain business features.")
    return df_feat


def encode_categorical_features(
    df: pd.DataFrame,
    nominal_cols: List[str] = NOMINAL_COLUMNS,
    ordinal_mappings: Dict[str, Dict[str, int]] = ORDINAL_MAPPINGS,
    drop_first: bool = True
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Executes comparative structured categorical encoding:
    - Binary Encoding: Direct 0/1 mapping for strictly binary features
      (gender, Partner, Dependents, PhoneService, PaperlessBilling).
    - Ordinal Encoding: Preserves explicit hierarchy for Contract
      (Month-to-month: 0, One year: 1, Two year: 2).
    - One-Hot Encoding: Maps multi-class nominal features with drop='first'
      to guarantee full rank and prevent dummy variable trap.
    - Target Binarization: Maps Churn ('Yes' -> 1, 'No' -> 0).

    Parameters
    ----------
    df : pd.DataFrame
        Input data.
    nominal_cols : List[str]
        Nominal categorical columns.
    ordinal_mappings : Dict[str, Dict[str, int]]
        Explicit value-to-rank dictionaries.
    drop_first : bool
        Whether to drop first level in dummy variables.

    Returns
    -------
    Tuple[pd.DataFrame, List[str]]
        Encoded DataFrame and list of all feature column names.
    """
    logger.info("Encoding categorical features (Binary + Ordinal + One-Hot with drop='first')...")
    df_encoded = df.copy()

    # 1. Binary Features (0/1 mapping)
    binary_cols = ["gender", "Partner", "Dependents", "PhoneService", "PaperlessBilling"]
    binary_map = {
        "Female": 0, "Male": 1,
        "No": 0, "Yes": 1,
        "no": 0, "yes": 1,
        0: 0, 1: 1
    }
    for col in binary_cols:
        if col in df_encoded.columns:
            df_encoded[col] = df_encoded[col].map(binary_map).fillna(0).astype(int)
            logger.info(f"Applied Binary Encoding to '{col}'")

    # 2. Ordinal Encoding for Contract
    for col, mapping in ordinal_mappings.items():
        if col in df_encoded.columns:
            # If values match keys, map them; if already mapped/numeric, keep
            if df_encoded[col].dtype == object or str(df_encoded[col].dtype).startswith("str") or df_encoded[col].dtype == "string":
                df_encoded[col] = df_encoded[col].map(mapping).fillna(0).astype(int)
                logger.info(f"Applied Ordinal Encoding to '{col}' with mapping: {mapping}")
            elif df_encoded[col].dtype in [np.int64, np.int32, int]:
                logger.info(f"Column '{col}' already integer encoded.")

    # 3. Target Column Binarization
    if TARGET_COLUMN in df_encoded.columns:
        if df_encoded[TARGET_COLUMN].dtype == object or str(df_encoded[TARGET_COLUMN].dtype).startswith("str") or df_encoded[TARGET_COLUMN].dtype == "string":
            df_encoded[TARGET_COLUMN] = df_encoded[TARGET_COLUMN].map(
                {"Yes": 1, "No": 0, "yes": 1, "no": 0, 1: 1, 0: 0}
            ).fillna(0).astype(int)
            logger.info("Binarized target 'Churn' to integer {0, 1}")

    # 4. Multi-class Nominal One-Hot Encoding
    # Filter out columns that are already binary/ordinal
    multi_class_nominals = [
        c for c in nominal_cols
        if c in df_encoded.columns and c not in binary_cols and c not in ordinal_mappings
    ]
    if multi_class_nominals:
        df_encoded = pd.get_dummies(
            df_encoded,
            columns=multi_class_nominals,
            drop_first=drop_first,
            dtype=int
        )
        logger.info(f"Applied One-Hot Encoding across {len(multi_class_nominals)} nominal columns: {multi_class_nominals}")

    feature_cols = [c for c in df_encoded.columns if c not in ["customerID", TARGET_COLUMN, "anomaly_score"]]
    return df_encoded, feature_cols


def calculate_vif(
    df: pd.DataFrame,
    feature_cols: List[str]
) -> pd.DataFrame:
    """
    Computes Variance Inflation Factor (VIF) to detect severe multicollinearity.
    Rule of thumb:
    - VIF < 5: Low multicollinearity (safe)
    - 5 <= VIF < 10: Moderate multicollinearity (acceptable)
    - VIF >= 10: Severe multicollinearity (redundant feature / near-linear combination)

    Parameters
    ----------
    df : pd.DataFrame
        Feature dataset.
    feature_cols : List[str]
        Features to inspect.

    Returns
    -------
    pd.DataFrame
        Ranked VIF summary table.
    """
    logger.info(f"Calculating Variance Inflation Factor (VIF) across {len(feature_cols)} features...")
    X = df[feature_cols].dropna().astype(float)
    X_const = X.copy()
    X_const["_intercept"] = 1.0

    vif_records = []
    for i, col in enumerate(feature_cols):
        idx = X_const.columns.get_loc(col)
        vif_val = variance_inflation_factor(X_const.values, idx)
        status = "Acceptable (<5)"
        if vif_val >= 10:
            status = "High Collinearity (>=10)"
        elif vif_val >= 5:
            status = "Moderate (5-10)"

        vif_records.append({
            "Feature": col,
            "VIF": round(vif_val, 2),
            "Collinearity_Status": status
        })

    vif_df = pd.DataFrame(vif_records).sort_values("VIF", ascending=False).reset_index(drop=True)
    return vif_df


def compute_feature_importance_ranking(
    X: pd.DataFrame,
    y: pd.Series,
    top_k: int = 15
) -> pd.DataFrame:
    """
    Calculates dual-perspective feature importance:
    1. Information Theory: Mutual Information (`mutual_info_classif`)
    2. Tree Ensemble Gini Importance: Random Forest Classifier

    Parameters
    ----------
    X : pd.DataFrame
        Predictor matrix.
    y : pd.Series
        Target vector.
    top_k : int
        Number of top features to report.

    Returns
    -------
    pd.DataFrame
        Combined feature ranking table.
    """
    logger.info("Computing Mutual Information and Random Forest feature rankings...")
    mi_scores = mutual_info_classif(X, y, random_state=RANDOM_STATE)
    mi_series = pd.Series(mi_scores, index=X.columns, name="Mutual_Information")

    rf = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=RANDOM_STATE)
    rf.fit(X, y)
    rf_series = pd.Series(rf.feature_importances_, index=X.columns, name="RF_Gini_Importance")

    ranking_df = pd.DataFrame({
        "Mutual_Information": mi_series,
        "RF_Gini_Importance": rf_series
    })

    ranking_df["Norm_MI"] = ranking_df["Mutual_Information"] / ranking_df["Mutual_Information"].max()
    ranking_df["Norm_RF"] = ranking_df["RF_Gini_Importance"] / ranking_df["RF_Gini_Importance"].max()
    ranking_df["Composite_Score"] = 0.5 * ranking_df["Norm_MI"] + 0.5 * ranking_df["Norm_RF"]

    ranking_df = ranking_df.sort_values("Composite_Score", ascending=False).reset_index()
    ranking_df.rename(columns={"index": "Feature"}, inplace=True)

    logger.info(f"Top 5 Ranked Features: {ranking_df['Feature'].head(5).tolist()}")
    return ranking_df.head(top_k)


def run_complete_feature_engineering_pipeline(
    df_clean: pd.DataFrame,
    db_path: str = DB_PATH,
    table_name: str = FEATURED_TABLE
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Executes full modular pipeline:
    1. Skewness check & Yeo-Johnson power transformation
    2. Feature engineering mandiri (business domain features)
    3. Structured categorical encoding (Binary, Ordinal, OHE drop='first')
    4. Multicollinearity elimination (dropping raw TotalCharges, Ratio, Average_Monthly)
    5. Persists dataset to:
       - 'data/processed/telco_churn_featured.csv' (primary deliverable)
       - 'data/processed/telco_featured.csv' (pipeline compatibility)
       - SQLite table 'featured_customers'

    Returns
    -------
    Tuple[pd.DataFrame, List[str]]
        Final featured DataFrame and list of feature column names.
    """
    logger.info("Running complete Feature Engineering Pipeline...")

    # 1. Skewness Transformation
    df_skew, _, _ = auto_skewness_transform(df_clean, numeric_cols=NUMERIC_COLUMNS)

    # 2. Business Features
    df_domain = create_business_features(df_skew)

    # 3. Categorical Encoding
    df_encoded, all_features = encode_categorical_features(df_domain)

    # 4. Eliminasi Multikolinearitas Terarah:
    # Drop TotalCharges (untransformed), Ratio_Total_to_Monthly_Charge (redundant with tenure),
    # and Average_Monthly_Charges (redundant with MonthlyCharges)
    redundant_collinear = [
        "TotalCharges",
        "Ratio_Total_to_Monthly_Charge",
        "Average_Monthly_Charges"
    ]
    cols_to_drop = [c for c in redundant_collinear if c in df_encoded.columns]
    df_final = df_encoded.drop(columns=cols_to_drop)
    logger.info(f"Eliminated redundant collinear features: {cols_to_drop}")

    final_feature_cols = [
        c for c in df_final.columns
        if c not in ["customerID", TARGET_COLUMN, "anomaly_score"]
    ]

    # 5. Persist to SQLite
    save_to_sqlite(df_final, table_name=table_name, db_path=db_path, if_exists="replace")

    # 6. Save to CSVs
    primary_csv = PROCESSED_DATA_DIR / "telco_churn_featured.csv"
    compat_csv = PROCESSED_DATA_DIR / "telco_featured.csv"
    df_final.to_csv(primary_csv, index=False)
    df_final.to_csv(compat_csv, index=False)

    logger.info(f"Featured dataset persisted to SQLite '{table_name}' and CSV '{primary_csv}'.")
    return df_final, final_feature_cols
