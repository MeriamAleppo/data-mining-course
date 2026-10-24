"""
Configuration Module for Data Mining Pipeline
Centralizes all directories, database paths, random seeds, and domain constants.
"""

from pathlib import Path
from typing import List, Dict

# Base project paths
BASE_DIR: Path = Path(__file__).resolve().parent.parent
DATA_DIR: Path = BASE_DIR / "data"
LOGS_DIR: Path = BASE_DIR / "logs"
CACHE_DIR: Path = DATA_DIR / "cache"
PROCESSED_DATA_DIR: Path = DATA_DIR / "processed"

# File paths
RAW_CSV_PATH: Path = DATA_DIR / "telco_customer_churn.csv"
RAW_JSON_PATH: Path = DATA_DIR / "churn_raw.json"
IRIS_CSV_PATH: Path = DATA_DIR / "iris_data.csv"
LEGACY_DB_PATH: Path = DATA_DIR / "sample_database.db"

# SQLite Central Database Path
DB_PATH: Path = DATA_DIR / "telco_churn.db"

# SQLite Table Names
RAW_TABLE: str = "raw_customers"
CLEANED_TABLE: str = "cleaned_customers"
FEATURED_TABLE: str = "featured_customers"
METRICS_TABLE: str = "model_metrics"

# Global reproducibility seed
RANDOM_STATE: int = 42

# Feature Definitions
ID_COLUMN: str = "customerID"
TARGET_COLUMN: str = "Churn"

NUMERIC_COLUMNS: List[str] = [
    "tenure",
    "MonthlyCharges",
    "TotalCharges"
]

NOMINAL_COLUMNS: List[str] = [
    "gender",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "TechSupport",
    "PaperlessBilling",
    "PaymentMethod"
]

ORDINAL_COLUMNS: List[str] = ["Contract"]

ORDINAL_MAPPINGS: Dict[str, Dict[str, int]] = {
    "Contract": {
        "Month-to-month": 0,
        "One year": 1,
        "Two year": 2
    }
}

# Cost-Sensitive Matrix Parameters (USD / IDR equivalent units)
# False Negative (Customer churns unnoticed): Lost Lifetime Value (~$500)
# False Positive (False alarm, unnecessary retention promo): Promo cost (~$50)
# True Positive (Successfully retained): Prevented loss minus promo cost ($500 - $50 = $450)
COST_FALSE_NEGATIVE: float = 500.0
COST_FALSE_POSITIVE: float = 50.0
BENEFIT_TRUE_POSITIVE: float = 450.0

# Ensure directories exist
LOGS_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
