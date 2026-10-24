"""
Predictive Modeling and Machine Learning Benchmarking Module
Implements Stratified K-Fold Cross Validation, Hyperparameter Tuning (GridSearch),
Baseline Logistic Regression, Random Forest, XGBoost, Voting Ensemble,
PR-AUC & ROC-AUC curves, and Cost-Sensitive Matrix threshold optimization.
"""

from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import StratifiedKFold, GridSearchCV, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, roc_curve, precision_recall_curve,
    confusion_matrix, classification_report
)
from xgboost import XGBClassifier

from src.config import (
    RANDOM_STATE, TARGET_COLUMN, COST_FALSE_NEGATIVE,
    COST_FALSE_POSITIVE, BENEFIT_TRUE_POSITIVE, DB_PATH
)
from src.database import log_model_metric
from src.logger import get_logger

logger = get_logger("PredictiveModeling")


def prepare_modeling_data(
    df: pd.DataFrame,
    target_col: str = TARGET_COLUMN,
    test_size: float = 0.20,
    random_state: int = RANDOM_STATE
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, List[str]]:
    """
    Partitions dataset into Stratified Train and Test sets.

    Parameters
    ----------
    df : pd.DataFrame
        Featured dataset.
    target_col : str
        Target binary column.
    test_size : float
        Proportion of holdout test set.
    random_state : int
        Seed for reproducibility.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, List[str]]
        X_train, X_test, y_train, y_test, and feature_names.
    """
    logger.info(f"Preparing train/test split (test_size={test_size}, stratify by '{target_col}')...")
    drop_cols = [target_col]
    if "customerID" in df.columns:
        drop_cols.append("customerID")
    if "anomaly_score" in df.columns:
        drop_cols.append("anomaly_score")

    X = df.drop(columns=drop_cols)
    y_raw = df[target_col]
    if y_raw.dtype == object or str(y_raw.dtype).startswith("str") or y_raw.dtype == bool:
        y = y_raw.map({"Yes": 1, "No": 0, "yes": 1, "no": 0, 1: 1, 0: 0, "1": 1, "0": 0}).fillna(0).astype(int)
    else:
        y = y_raw.astype(int)
    feature_names = X.columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    logger.info(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}, Class balance y_train: {dict(y_train.value_counts(normalize=True).round(3))}")
    return X_train, X_test, y_train, y_test, feature_names


def build_and_tune_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv_splits: int = 5
) -> Dict[str, Any]:
    """
    Executes Hyperparameter Tuning using Stratified 5-Fold Cross Validation
    across three core architectures:
    1. Baseline: Logistic Regression (L2 regularization)
    2. Non-Linear Ensemble: Random Forest
    3. State-of-the-Art Gradient Boosting: XGBoost
    plus a Soft-Voting Ensemble.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training features.
    y_train : pd.Series
        Training labels.
    cv_splits : int
        Number of stratified cross-validation folds.

    Returns
    -------
    Dict[str, Any]
        Dictionary containing best estimators, tuning results, and grid searches.
    """
    logger.info(f"Initiating Model Benchmark & Hyperparameter Tuning with Stratified {cv_splits}-Fold CV...")
    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=RANDOM_STATE)
    models = {}

    # 1. Baseline: Logistic Regression Pipeline (with Standard Scaling)
    logger.info("Tuning Model 1: Baseline Logistic Regression...")
    lr_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(class_weight="balanced", max_iter=1000, random_state=RANDOM_STATE))
    ])
    lr_param_grid = {
        "classifier__C": [0.01, 0.1, 1.0, 10.0]
    }
    lr_grid = GridSearchCV(lr_pipe, lr_param_grid, cv=cv, scoring="roc_auc", n_jobs=-1)
    lr_grid.fit(X_train, y_train)
    models["Logistic Regression"] = lr_grid.best_estimator_
    logger.info(f"Best Logistic Regression ROC-AUC (CV): {lr_grid.best_score_:.4f} with params: {lr_grid.best_params_}")

    # 2. Random Forest Classifier
    logger.info("Tuning Model 2: Random Forest Classifier...")
    rf = RandomForestClassifier(class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1)
    rf_param_grid = {
        "n_estimators": [100, 150],
        "max_depth": [4, 6, 8],
        "min_samples_split": [5, 10]
    }
    rf_grid = GridSearchCV(rf, rf_param_grid, cv=cv, scoring="roc_auc", n_jobs=-1)
    rf_grid.fit(X_train, y_train)
    models["Random Forest"] = rf_grid.best_estimator_
    logger.info(f"Best Random Forest ROC-AUC (CV): {rf_grid.best_score_:.4f} with params: {rf_grid.best_params_}")

    # 3. XGBoost Classifier
    logger.info("Tuning Model 3: XGBoost Classifier...")
    # Calculate scale_pos_weight = count(negative) / count(positive)
    neg_count = (y_train == 0).sum()
    pos_count = (y_train == 1).sum()
    scale_pos = neg_count / pos_count if pos_count > 0 else 1.0

    xgb = XGBClassifier(
        eval_metric="logloss",
        random_state=RANDOM_STATE,
        scale_pos_weight=scale_pos,
        n_jobs=-1
    )
    xgb_param_grid = {
        "n_estimators": [100, 150],
        "max_depth": [3, 4, 5],
        "learning_rate": [0.03, 0.1],
        "subsample": [0.8, 1.0]
    }
    xgb_grid = GridSearchCV(xgb, xgb_param_grid, cv=cv, scoring="roc_auc", n_jobs=-1)
    xgb_grid.fit(X_train, y_train)
    models["XGBoost"] = xgb_grid.best_estimator_
    logger.info(f"Best XGBoost ROC-AUC (CV): {xgb_grid.best_score_:.4f} with params: {xgb_grid.best_params_}")

    # 4. Soft Voting Ensemble
    logger.info("Constructing Model 4: Soft-Voting Ensemble (Random Forest + XGBoost)...")
    voting_clf = VotingClassifier(
        estimators=[
            ("rf", models["Random Forest"]),
            ("xgb", models["XGBoost"])
        ],
        voting="soft"
    )
    voting_clf.fit(X_train, y_train)
    models["Voting Ensemble"] = voting_clf

    return {
        "models": models,
        "grid_searches": {
            "Logistic Regression": lr_grid,
            "Random Forest": rf_grid,
            "XGBoost": xgb_grid
        }
    }


def evaluate_models_on_test(
    models: Dict[str, Any],
    X_test: pd.DataFrame,
    y_test: pd.Series,
    cost_fn: float = COST_FALSE_NEGATIVE,
    cost_fp: float = COST_FALSE_POSITIVE
) -> pd.DataFrame:
    """
    Evaluates all benchmark models on holdout test set using ROC-AUC, PR-AUC,
    Accuracy, Precision, Recall, F1-Score, and Financial Business Cost.
    Persists evaluation metrics directly to SQLite.
    """
    logger.info(f"Evaluating {len(models)} models on holdout test set ({len(y_test)} records)...")
    results = []

    for name, model in models.items():
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        roc_auc = roc_auc_score(y_test, y_prob)
        pr_auc = average_precision_score(y_test, y_prob)

        # Cost matrix at default 0.5 threshold
        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()
        business_cost = (fn * cost_fn) + (fp * cost_fp)

        # Optimize threshold
        opt_thresh, min_cost = optimize_decision_threshold(y_test, y_prob, cost_fn, cost_fp)

        metric_dict = {
            "Model": name,
            "Accuracy": round(acc, 4),
            "Precision": round(prec, 4),
            "Recall": round(rec, 4),
            "F1-Score": round(f1, 4),
            "ROC-AUC": round(roc_auc, 4),
            "PR-AUC": round(pr_auc, 4),
            "Default_Cost_USD": round(business_cost, 2),
            "Optimal_Threshold": round(opt_thresh, 2),
            "Optimal_Cost_USD": round(min_cost, 2),
            "Cost_Savings_USD": round(business_cost - min_cost, 2)
        }
        results.append(metric_dict)

        # Persist to SQLite
        try:
            log_model_metric(name, {
                "accuracy": acc, "precision": prec, "recall": rec, "f1_score": f1,
                "roc_auc": roc_auc, "pr_auc": pr_auc, "optimal_threshold": opt_thresh,
                "total_business_cost": min_cost
            })
        except Exception as e:
            logger.warning(f"Could not log metric to SQLite for {name}: {e}")

    results_df = pd.DataFrame(results).sort_values("ROC-AUC", ascending=False).reset_index(drop=True)
    logger.info("Model evaluation completed.")
    return results_df


def optimize_decision_threshold(
    y_true: pd.Series,
    y_prob: np.ndarray,
    cost_fn: float = COST_FALSE_NEGATIVE,
    cost_fp: float = COST_FALSE_POSITIVE
) -> Tuple[float, float]:
    """
    Finds classification probability threshold tau* that minimizes total business cost:
    Total Cost = (FN * Cost_FN) + (FP * Cost_FP)

    Returns
    -------
    Tuple[float, float]
        Optimal threshold and minimum cost.
    """
    thresholds = np.linspace(0.05, 0.95, 91)
    costs = []

    for t in thresholds:
        preds = (y_prob >= t).astype(int)
        cm = confusion_matrix(y_true, preds)
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
        else:
            tn, fp, fn, tp = len(y_true), 0, 0, 0
        total_cost = (fn * cost_fn) + (fp * cost_fp)
        costs.append(total_cost)

    min_idx = np.argmin(costs)
    return thresholds[min_idx], costs[min_idx]


def plot_combined_roc_and_pr_curves(
    models: Dict[str, Any],
    X_test: pd.DataFrame,
    y_test: pd.Series,
    figsize: Tuple[int, int] = (15, 6)
) -> plt.Figure:
    """
    Renders ROC and Precision-Recall Curves side-by-side for all benchmark models.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    colors = ["#2b5c8f", "#e67e22", "#27ae60", "#8e44ad"]

    # No-skill baseline for PR curve
    no_skill_pr = y_test.sum() / len(y_test)

    for (name, model), color in zip(models.items(), colors):
        y_prob = model.predict_proba(X_test)[:, 1]

        # 1. ROC Curve
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = roc_auc_score(y_test, y_prob)
        ax1.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc:.4f})", color=color, lw=2)

        # 2. PR Curve
        prec, rec, _ = precision_recall_curve(y_test, y_prob)
        pr_auc = average_precision_score(y_test, y_prob)
        ax2.plot(rec, prec, label=f"{name} (PR-AUC = {pr_auc:.4f})", color=color, lw=2)

    # Styling ROC
    ax1.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random Chance (0.50)")
    ax1.set_title("Receiver Operating Characteristic (ROC) Curve", fontsize=12, fontweight="bold")
    ax1.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=10)
    ax1.set_ylabel("True Positive Rate (Recall / Sensitivity)", fontsize=10)
    ax1.legend(loc="lower right", fontsize=9)
    ax1.grid(True, linestyle="--", alpha=0.5)

    # Styling PR
    ax2.axhline(no_skill_pr, linestyle="--", color="gray", label=f"No Skill Baseline ({no_skill_pr:.2f})")
    ax2.set_title("Precision-Recall (PR) Curve (Crucial for Imbalance)", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Recall", fontsize=10)
    ax2.set_ylabel("Precision", fontsize=10)
    ax2.legend(loc="lower left", fontsize=9)
    ax2.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    return fig


def plot_cost_threshold_curve(
    y_test: pd.Series,
    y_prob: np.ndarray,
    model_name: str = "XGBoost",
    cost_fn: float = COST_FALSE_NEGATIVE,
    cost_fp: float = COST_FALSE_POSITIVE,
    figsize: Tuple[int, int] = (8, 5)
) -> plt.Figure:
    """
    Visualizes business cost trajectory as decision threshold varies,
    highlighting default 0.50 vs optimal cost-minimizing threshold tau*.
    """
    thresholds = np.linspace(0.05, 0.95, 91)
    costs = []

    for t in thresholds:
        preds = (y_prob >= t).astype(int)
        cm = confusion_matrix(y_test, preds)
        tn, fp, fn, tp = cm.ravel()
        costs.append((fn * cost_fn) + (fp * cost_fp))

    costs = np.array(costs)
    opt_idx = np.argmin(costs)
    opt_t = thresholds[opt_idx]
    opt_c = costs[opt_idx]

    # Default 0.50 index
    def_idx = np.abs(thresholds - 0.50).argmin()
    def_c = costs[def_idx]

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(thresholds, costs, color="#2b5c8f", lw=2.5, label="Total Expected Business Cost ($)")
    ax.scatter([0.50], [def_c], color="#d9534f", s=90, zorder=5, label=f"Default Threshold (0.50) -> ${def_c:,.0f}")
    ax.scatter([opt_t], [opt_c], color="#27ae60", s=110, zorder=5, label=f"Cost-Optimal Threshold ({opt_t:.2f}) -> ${opt_c:,.0f}")

    # Annotate savings
    savings = def_c - opt_c
    ax.annotate(
        f"Savings: ${savings:,.0f}\n({(savings/def_c)*100:.1f}%)",
        (opt_t, opt_c),
        xytext=(opt_t + 0.10, opt_c + (def_c - opt_c) * 0.4),
        arrowprops=dict(facecolor="#27ae60", shrink=0.08, width=1.5, headwidth=6),
        fontweight="bold", color="#1b5e20", fontsize=10
    )

    ax.set_title(f"Cost-Sensitive Threshold Optimization ({model_name})\n(CFN=${cost_fn:.0f}, CFP=${cost_fp:.0f})", fontsize=11, fontweight="bold")
    ax.set_xlabel("Decision Classification Threshold (tau)", fontsize=10)
    ax.set_ylabel("Total Financial Loss ($)", fontsize=10)
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    return fig


def plot_confusion_matrices(
    models: Dict[str, Any],
    X_test: pd.DataFrame,
    y_test: pd.Series,
    figsize: Tuple[int, int] = (16, 4)
) -> plt.Figure:
    """
    Renders heatmaps of Confusion Matrices across benchmark models.
    """
    fig, axes = plt.subplots(1, len(models), figsize=figsize)
    if len(models) == 1:
        axes = [axes]

    for (name, model), ax in zip(models.items(), axes):
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
            xticklabels=["Stay (0)", "Churn (1)"],
            yticklabels=["Stay (0)", "Churn (1)"]
        )
        ax.set_title(f"{name}\nConfusion Matrix", fontsize=11, fontweight="bold")
        ax.set_xlabel("Predicted Label", fontsize=10)
        ax.set_ylabel("Actual Label", fontsize=10)

    plt.tight_layout()
    return fig


def plot_feature_importance_top(
    model: Any,
    feature_names: List[str],
    top_n: int = 10,
    figsize: Tuple[int, int] = (9, 5)
) -> plt.Figure:
    """
    Visualizes top feature importances from tree-based model (XGBoost or Random Forest).
    """
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "named_steps") and hasattr(model.named_steps["classifier"], "coef_"):
        importances = np.abs(model.named_steps["classifier"].coef_[0])
    else:
        raise ValueError("Model does not provide feature_importances_ or coef_")

    feat_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importances
    }).sort_values("Importance", ascending=True).tail(top_n)

    fig, ax = plt.subplots(figsize=figsize)
    ax.barh(feat_df["Feature"], feat_df["Importance"], color="#2b5c8f", edgecolor="black", alpha=0.8)
    ax.set_title(f"Top {top_n} Predictive Feature Importances", fontsize=11, fontweight="bold")
    ax.set_xlabel("Relative Importance Weight", fontsize=10)
    ax.grid(axis="x", linestyle="--", alpha=0.5)

    plt.tight_layout()
    return fig
