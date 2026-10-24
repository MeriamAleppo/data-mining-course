"""
Database Management Module
Handles SQLite operations, table schema initialization, data persistence, and querying.
"""

import sqlite3
from pathlib import Path
from typing import Optional, Union, List, Dict, Any
import pandas as pd
from src.config import DB_PATH, RAW_TABLE, CLEANED_TABLE, FEATURED_TABLE, METRICS_TABLE
from src.logger import get_logger

logger = get_logger("DatabaseManager")


def get_connection(db_path: Union[str, Path] = DB_PATH) -> sqlite3.Connection:
    """
    Creates and returns a connection to the SQLite database.

    Parameters
    ----------
    db_path : Union[str, Path]
        Path to SQLite database file.

    Returns
    -------
    sqlite3.Connection
    """
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


def init_db(db_path: Union[str, Path] = DB_PATH) -> None:
    """
    Initializes metadata and core schema structure in the SQLite database.
    """
    logger.info(f"Initializing SQLite database schema at: {db_path}")
    with get_connection(db_path) as conn:
        cursor = conn.cursor()

        # Metrics history table
        cursor.execute(f"""
        CREATE TABLE IF NOT EXISTS {METRICS_TABLE} (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            model_name TEXT NOT NULL,
            run_timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            accuracy REAL,
            precision REAL,
            recall REAL,
            f1_score REAL,
            roc_auc REAL,
            pr_auc REAL,
            optimal_threshold REAL,
            total_business_cost REAL,
            hyperparameters TEXT
        );
        """)

        # Cache registry table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS pipeline_cache (
            cache_key TEXT PRIMARY KEY,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            description TEXT
        );
        """)

        conn.commit()
    logger.info("Database schema initialized successfully.")


def save_to_sqlite(
    df: pd.DataFrame,
    table_name: str,
    db_path: Union[str, Path] = DB_PATH,
    if_exists: str = "replace",
    index: bool = False
) -> int:
    """
    Persists a pandas DataFrame into SQLite table.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to persist.
    table_name : str
        Target table name.
    db_path : Union[str, Path]
        Path to SQLite database.
    if_exists : str
        'replace', 'append', or 'fail'.
    index : bool
        Whether to write DataFrame index as a column.

    Returns
    -------
    int
        Number of rows persisted.
    """
    logger.info(f"Saving {len(df)} rows to SQLite table '{table_name}' ({if_exists})...")
    with get_connection(db_path) as conn:
        df.to_sql(table_name, conn, if_exists=if_exists, index=index)
    logger.info(f"Successfully saved to table '{table_name}'.")
    return len(df)


def read_from_sqlite(
    table_or_query: str,
    db_path: Union[str, Path] = DB_PATH,
    params: Optional[Union[tuple, dict]] = None
) -> pd.DataFrame:
    """
    Reads data from SQLite table or direct SQL query into a pandas DataFrame.

    Parameters
    ----------
    table_or_query : str
        Table name or raw SELECT query.
    db_path : Union[str, Path]
        Path to SQLite database.
    params : Optional[Union[tuple, dict]]
        Query parameters if table_or_query is a query.

    Returns
    -------
    pd.DataFrame
    """
    query = table_or_query.strip()
    if not query.lower().startswith("select"):
        query = f"SELECT * FROM {table_or_query}"

    logger.info(f"Executing SQLite query: {query[:80]}...")
    with get_connection(db_path) as conn:
        df = pd.read_sql_query(query, conn, params=params)
    logger.info(f"Query returned {len(df)} rows and {len(df.columns)} columns.")
    return df


def list_tables(db_path: Union[str, Path] = DB_PATH) -> List[str]:
    """
    Returns a list of all user tables in the SQLite database.
    """
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name;")
        tables = [row[0] for row in cursor.fetchall()]
    return tables


def log_model_metric(
    model_name: str,
    metrics: Dict[str, Any],
    hyperparameters: Optional[Dict[str, Any]] = None,
    db_path: Union[str, Path] = DB_PATH
) -> None:
    """
    Records benchmark model evaluation metrics into the SQLite METRICS_TABLE.
    """
    import json
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(f"""
        INSERT INTO {METRICS_TABLE} (
            model_name, accuracy, precision, recall, f1_score,
            roc_auc, pr_auc, optimal_threshold, total_business_cost, hyperparameters
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
        """, (
            model_name,
            metrics.get("accuracy"),
            metrics.get("precision"),
            metrics.get("recall"),
            metrics.get("f1_score"),
            metrics.get("roc_auc"),
            metrics.get("pr_auc"),
            metrics.get("optimal_threshold", 0.5),
            metrics.get("total_business_cost", 0.0),
            json.dumps(hyperparameters) if hyperparameters else None
        ))
        conn.commit()
    logger.info(f"Logged evaluation metrics for model '{model_name}' to SQLite.")
