"""
Structured logging setup for production RAG pipeline.
Supports plain-text (default) and JSON (LOG_FORMAT=json) output formats.
"""

from __future__ import annotations

import json as _json
import logging
import logging.handlers
import os
import sys
from pathlib import Path
from typing import Optional

from .config import get_config


class ProductionFormatter(logging.Formatter):
    """Custom formatter with context and timestamps."""

    def format(self, record: logging.LogRecord) -> str:
        if record.levelno >= logging.ERROR:
            fmt = '[%(asctime)s] %(levelname)-8s [%(name)s.%(funcName)s:%(lineno)d] %(message)s'
        else:
            fmt = '[%(asctime)s] %(levelname)-8s %(message)s'
        
        formatter = logging.Formatter(fmt, datefmt='%Y-%m-%d %H:%M:%S')
        return formatter.format(record)


class ConsoleSafeFormatter(ProductionFormatter):
    """Formatter that replaces problematic unicode glyphs on non-UTF8 consoles."""

    _REPLACEMENTS = {
        "✓": "[OK]",
        "✗": "[ERR]",
        "→": "->",
    }

    @staticmethod
    def _is_utf8_console() -> bool:
        encoding = (getattr(sys.stderr, "encoding", None) or "").lower()
        if "utf" in encoding:
            return True
        env_encoding = (os.environ.get("PYTHONIOENCODING") or "").lower()
        return "utf" in env_encoding

    def format(self, record: logging.LogRecord) -> str:
        message = super().format(record)
        if self._is_utf8_console():
            return message
        for old, new in self._REPLACEMENTS.items():
            message = message.replace(old, new)
        return message


class JsonFormatter(logging.Formatter):
    """Structured JSON log formatter for log aggregation pipelines.

    Enable with environment variable: LOG_FORMAT=json
    Each line is a valid JSON object with timestamp, level, logger, message, module, line.
    """

    def format(self, record: logging.LogRecord) -> str:
        return _json.dumps(
            {
                "timestamp": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
                "level": record.levelname,
                "logger": record.name,
                "message": record.getMessage(),
                "module": record.module,
                "line": record.lineno,
            },
            ensure_ascii=False,
        )


class RAGLogger:
    """Centralized logger for RAG pipeline."""

    _instance: Optional[RAGLogger] = None

    def __new__(cls) -> RAGLogger:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        """Initialize logger with file and console handlers."""
        if self._initialized:
            return

        config = get_config()
        self.logger = logging.getLogger("pramana.rag")
        self.logger.setLevel(getattr(logging, config.logging.log_level))
        self.logger.propagate = False

        # Reset existing handlers to avoid stale formatter duplication across imports.
        if self.logger.handlers:
            for handler in list(self.logger.handlers):
                self.logger.removeHandler(handler)

        # File handler with rotation
        log_path = Path(config.logging.log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.handlers.RotatingFileHandler(
            log_path,
            maxBytes=config.logging.max_log_size,
            backupCount=config.logging.backup_count,
            encoding="utf-8",
        )
        file_handler.setLevel(logging.DEBUG)
        file_fmt_cls = JsonFormatter if config.logging.log_format == "json" else ProductionFormatter
        file_handler.setFormatter(file_fmt_cls())
        self.logger.addHandler(file_handler)

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(ConsoleSafeFormatter())
        self.logger.addHandler(console_handler)

        self._initialized = True

    def get_logger(self, name: str) -> logging.Logger:
        """Get a child logger for a specific module."""
        return self.logger.getChild(name)


def get_logger(module_name: str) -> logging.Logger:
    """Get logger for a module."""
    rag_logger = RAGLogger()
    return rag_logger.get_logger(module_name)


# Module-level loggers
logger_embeddings = get_logger("embeddings")
logger_vector_store = get_logger("vector_store")
logger_retrieval = get_logger("retrieval")
logger_llm = get_logger("llm")
logger_pipeline = get_logger("pipeline")
logger_api = get_logger("api")
