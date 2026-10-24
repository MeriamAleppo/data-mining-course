"""
Caching Module for Data Mining Pipeline
Provides disk-based serialization and retrieval for computationally heavy steps.
"""

import functools
import hashlib
import pickle
from pathlib import Path
from typing import Any, Callable, Optional
from src.config import CACHE_DIR
from src.logger import get_logger

logger = get_logger("CacheManager")


def get_cache_path(key: str) -> Path:
    """Generates a unique cache file path based on a hash of the key."""
    hashed = hashlib.md5(key.encode("utf-8")).hexdigest()
    return CACHE_DIR / f"{hashed}.pkl"


def save_to_cache(key: str, data: Any) -> None:
    """Saves arbitrary python object to disk cache."""
    cache_path = get_cache_path(key)
    try:
        with open(cache_path, "wb") as f:
            pickle.dump(data, f)
        logger.debug(f"Saved cache for key='{key}' to {cache_path.name}")
    except Exception as e:
        logger.warning(f"Failed to cache key='{key}': {e}")


def load_from_cache(key: str) -> Optional[Any]:
    """Loads cached object if present, else returns None."""
    cache_path = get_cache_path(key)
    if cache_path.exists():
        try:
            with open(cache_path, "rb") as f:
                data = pickle.load(f)
            logger.info(f"Loaded cached artifact for key='{key}'")
            return data
        except Exception as e:
            logger.warning(f"Corrupted cache file for key='{key}', ignoring: {e}")
            return None
    return None


def clear_cache() -> int:
    """Clears all cached files in the CACHE_DIR."""
    count = 0
    for pkl_file in CACHE_DIR.glob("*.pkl"):
        pkl_file.unlink()
        count += 1
    logger.info(f"Cleared {count} cached files.")
    return count
