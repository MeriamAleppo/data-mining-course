"""
Data Ingestion Module
Handles robust data retrieval from Flat Files (CSV), Semi-Structured (JSON), and Relational Databases (SQLite).
"""

import json
import sqlite3
from pathlib import Path
from typing import Dict, Any, Union, Optional
import pandas as pd
from src.config import (
    RAW_CSV_PATH, RAW_JSON_PATH, IRIS_CSV_PATH, LEGACY_DB_PATH,
    DB_PATH, RAW_TABLE, ID_COLUMN
)
from src.database import save_to_sqlite, init_db
from src.logger import get_logger

logger = get_logger("DataIngestion")


def load_csv_data(
    file_path: Union[str, Path] = RAW_CSV_PATH,
    delimiter: str = ",",
    na_values: Optional[list] = None
) -> pd.DataFrame:
    """
    Loads raw tabular data from CSV with strict error handling and whitespace cleaning.

    Parameters
    ----------
    file_path : Union[str, Path]
        Path to the CSV file.
    delimiter : str
        CSV delimiter.
    na_values : Optional[list]
        Custom list of representations for NA/NaN values.

    Returns
    -------
    pd.DataFrame
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"CSV file not found at: {path}")

    custom_na = ["", " ", "NA", "N/A", "null", "NaN"]
    if na_values:
        custom_na.extend(na_values)

    logger.info(f"Reading CSV from {path.name}...")
    df = pd.read_csv(path, sep=delimiter, na_values=custom_na)

    # Convert numeric fields that might be object due to spaces (e.g. TotalCharges)
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    logger.info(f"Loaded {len(df)} rows and {len(df.columns)} columns from CSV.")
    return df


def load_json_data(file_path: Union[str, Path] = RAW_JSON_PATH) -> pd.DataFrame:
    """
    Reads semi-structured JSON records into a DataFrame.

    Parameters
    ----------
    file_path : Union[str, Path]
        Path to JSON file.

    Returns
    -------
    pd.DataFrame
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"JSON file not found at: {path}")

    logger.info(f"Reading JSON from {path.name}...")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    df = pd.DataFrame(data)
    logger.info(f"Loaded {len(df)} JSON records.")
    return df


def query_legacy_sqlite(
    db_path: Union[str, Path] = LEGACY_DB_PATH,
    query: str = "SELECT * FROM customer_transactions LIMIT 10"
) -> pd.DataFrame:
    """
    Retrieves data directly from a legacy or source SQLite database.

    Parameters
    ----------
    db_path : Union[str, Path]
        Database path.
    query : str
        SQL query.

    Returns
    -------
    pd.DataFrame
    """
    path = Path(db_path)
    if not path.exists():
        logger.warning(f"Legacy database not found at {path}. Returning empty DataFrame.")
        return pd.DataFrame()

    with sqlite3.connect(str(path)) as conn:
        df = pd.read_sql_query(query, conn)
    logger.info(f"Retrieved {len(df)} records from legacy SQLite query.")
    return df


def validate_data_quality(df: pd.DataFrame, id_col: str = ID_COLUMN) -> Dict[str, Any]:
    """
    Audits the input DataFrame for schema completeness, nulls, and duplicate records.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    id_col : str
        Primary key column for checking uniqueness.

    Returns
    -------
    Dict[str, Any]
        Audit summary metrics.
    """
    total_rows = len(df)
    total_cols = len(df.columns)
    duplicate_ids = int(df.duplicated(subset=[id_col]).sum()) if id_col in df.columns else 0
    null_counts = df.isnull().sum()
    null_summary = null_counts[null_counts > 0].to_dict()

    audit = {
        "total_records": total_rows,
        "total_features": total_cols,
        "duplicate_ids": duplicate_ids,
        "features_with_nulls": null_summary,
        "memory_usage_mb": round(df.memory_usage(deep=True).sum() / (1024 * 1024), 3)
    }

    logger.info(f"Data Quality Audit: {total_rows} rows, {duplicate_ids} duplicate IDs, "
                f"{len(null_summary)} columns with missing values.")
    return audit


def ingest_telco_to_database(
    csv_path: Union[str, Path] = RAW_CSV_PATH,
    db_path: Union[str, Path] = DB_PATH,
    table_name: str = RAW_TABLE
) -> pd.DataFrame:
    """
    End-to-End ingestion pipeline: Reads CSV, audits quality, initializes DB,
    and persists raw data to SQLite table.

    Returns
    -------
    pd.DataFrame
        The loaded and audited DataFrame.
    """
    init_db(db_path)
    df = load_csv_data(csv_path)

    audit = validate_data_quality(df)
    logger.info(f"Audit Results: {audit}")

    # Persist to SQLite
    save_to_sqlite(df, table_name=table_name, db_path=db_path, if_exists="replace")
    return df
