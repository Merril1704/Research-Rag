"""
Centralized logging for ResearchRAG.

Every module should use this logger instead of print() or ad-hoc logging.
Logs are written to both console and a rotating log file for post-mortem debugging.

Usage in any module:
    from researchrag.logger import get_logger
    logger = get_logger(__name__)
    logger.info("Something happened")
    logger.error("Something broke", exc_info=True)
"""

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

from researchrag.config import LOG_DIR

# ---------------------------------------------------------------------------
# Log format
# ---------------------------------------------------------------------------
LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)-30s | %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# ---------------------------------------------------------------------------
# Log file settings
# ---------------------------------------------------------------------------
LOG_FILE = LOG_DIR / "researchrag.log"
MAX_LOG_BYTES = 5 * 1024 * 1024   # 5 MB per file
BACKUP_COUNT = 3                   # keep 3 rotated files


def get_logger(name: str, level: int = logging.DEBUG) -> logging.Logger:
    """
    Get a configured logger for a module.

    Args:
        name: Module name (typically __name__).
        level: Logging level. DEBUG by default so file captures everything;
               console handler filters to INFO+.

    Returns:
        Configured logger instance.
    """
    logger = logging.getLogger(name)

    # Avoid adding duplicate handlers if called multiple times
    if logger.handlers:
        return logger

    logger.setLevel(level)
    formatter = logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT)

    # --- Console handler (INFO and above) ---
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # --- File handler (DEBUG and above, rotating) ---
    file_handler = RotatingFileHandler(
        LOG_FILE,
        maxBytes=MAX_LOG_BYTES,
        backupCount=BACKUP_COUNT,
        encoding="utf-8",
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger
