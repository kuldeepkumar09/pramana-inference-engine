"""
FAISS Vector Store persistence and management.
Handles saving/loading indices and metadata for faster initialization.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Optional

import faiss
import numpy as np

from .config import get_config
from .logging_setup import logger_vector_store


class VectorStorePersistence:
    """Manages FAISS index persistence and recovery."""

    def __init__(self):
        """Initialize persistence manager."""
        self.config = get_config()
        self.cache_dir = Path(self.config.vector_store.persist_dir)
        self.index_path = self.cache_dir / "faiss.index"
        self.metadata_path = self.cache_dir / "metadata.pkl"
        self.config_path = self.cache_dir / "config.json"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def has_cache(self) -> bool:
        """Check if a cached index exists."""
        return self.index_path.exists() and self.metadata_path.exists()

    def save_index(self, index: faiss.Index, metadata: list, config_info: dict) -> None:
        """
        Save FAISS index and metadata to disk.
        
        Args:
            index: FAISS index object
            metadata: List of metadata dicts for each indexed chunk
            config_info: Configuration info (embedding dim, etc.)
        """
        try:
            # Save FAISS index
            faiss.write_index(index, str(self.index_path))
            logger_vector_store.info(f"✓ Saved FAISS index: {self.index_path}")

            # Save metadata with pickle (handles complex objects)
            with open(self.metadata_path, 'wb') as f:
                pickle.dump(metadata, f)
            logger_vector_store.info(f"✓ Saved metadata: {self.metadata_path}")

            # Save config
            with open(self.config_path, 'w') as f:
                json.dump(config_info, f, indent=2)
            logger_vector_store.info(f"✓ Saved config: {self.config_path}")

        except Exception as e:
            logger_vector_store.error(f"✗ Failed to save index: {e}", exc_info=True)
            raise

    def load_index(self) -> tuple[faiss.Index, list, dict] | None:
        """
        Load FAISS index and metadata from disk.
        
        Returns:
            Tuple of (index, metadata, config) or None if not found
        """
        if not self.has_cache():
            logger_vector_store.warning("No cached index found")
            return None

        try:
            # Load FAISS index
            index = faiss.read_index(str(self.index_path))
            logger_vector_store.info(f"✓ Loaded FAISS index with {index.ntotal} vectors")

            # Load metadata
            with open(self.metadata_path, 'rb') as f:
                metadata = pickle.load(f)
            logger_vector_store.info(f"✓ Loaded {len(metadata)} metadata items")

            # Load config
            with open(self.config_path, 'r') as f:
                config_info = json.load(f)
            logger_vector_store.info(f"✓ Loaded config")

            return index, metadata, config_info

        except Exception as e:
            logger_vector_store.error(f"✗ Failed to load index: {e}", exc_info=True)
            return None

    def clear_cache(self) -> None:
        """Clear the cached index (useful for updates)."""
        try:
            for path in [self.index_path, self.metadata_path, self.config_path]:
                if path.exists():
                    path.unlink()
            logger_vector_store.info("✓ Cleared cache")
        except Exception as e:
            logger_vector_store.error(f"✗ Failed to clear cache: {e}", exc_info=True)

    def get_cache_stats(self) -> dict:
        """Get statistics about cached index."""
        if not self.has_cache():
            return {"cached": False}

        try:
            index = faiss.read_index(str(self.index_path))
            index_size = self.index_path.stat().st_size / (1024 * 1024)  # MB

            return {
                "cached": True,
                "vectors": index.ntotal,
                "index_size_mb": round(index_size, 2),
                "index_path": str(self.index_path),
            }
        except Exception as e:
            logger_vector_store.error(f"✗ Failed to get cache stats: {e}")
            return {"cached": False, "error": str(e)}


# Global instance
_persistence: Optional[VectorStorePersistence] = None


def get_persistence() -> VectorStorePersistence:
    """Get or create global persistence manager."""
    global _persistence
    if _persistence is None:
        _persistence = VectorStorePersistence()
    return _persistence
