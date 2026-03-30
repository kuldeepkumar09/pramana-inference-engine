"""
E5-Small Embeddings Module for RAG.
Handles semantic embedding generation for knowledge base chunks.
Supports GPU acceleration and error handling.
"""

from __future__ import annotations

import time
from collections import OrderedDict
from threading import RLock
import numpy as np
from typing import List, Optional
from sentence_transformers import SentenceTransformer

from .config import get_config
from .logging_setup import logger_embeddings


class EmbeddingEngine:
    """Manages semantic embeddings using E5-Small model with GPU support."""

    def __init__(self, model_name: Optional[str] = None, device: Optional[str] = None):
        """
        Initialize embedding engine with GPU support.
        
        Args:
            model_name: HuggingFace model identifier. Uses config if None.
            device: "cuda" or "cpu". Auto-detects GPU if None.
        """
        config = get_config()
        self.model_name = model_name or config.embeddings.model_name
        self.device = device or config.embeddings.device
        self._query_cache_enabled = bool(config.embeddings.cache_embeddings)
        self._query_cache_max = max(16, int(config.embeddings.query_cache_size))
        self._query_cache_ttl_seconds = max(1, int(config.embeddings.query_cache_ttl_seconds))
        self._query_cache: "OrderedDict[str, tuple[float, np.ndarray]]" = OrderedDict()
        self._query_cache_lock = RLock()

        try:
            logger_embeddings.info(f"Loading {self.model_name} on device: {self.device}")
            self.model = SentenceTransformer(self.model_name, device=self.device)
            self.embedding_dim = self.model.get_sentence_embedding_dimension()
            logger_embeddings.info(f"✓ Embeddings loaded. Dim: {self.embedding_dim}, Device: {self.device}")
        except Exception as e:
            logger_embeddings.error(f"✗ Failed to load model: {e}", exc_info=True)
            raise

    def embed_text(self, text: str) -> np.ndarray:
        """
        Embed a single text string with error handling.
        
        Args:
            text: Text to embed
            
        Returns:
            1D numpy array of embeddings
            
        Raises:
            ValueError: If text is empty or embedding fails
        """
        if not text or not isinstance(text, str):
            raise ValueError("Text must be a non-empty string")

        try:
            prefixed_text = f"passage: {text}"
            embedding = self.model.encode(prefixed_text, convert_to_numpy=True)
            return embedding.astype(np.float32)
        except Exception as e:
            logger_embeddings.error(f"✗ Embedding failed: {e}", exc_info=True)
            raise

    def embed_query(self, query: str) -> np.ndarray:
        """
        Embed a query with error handling and GPU optimization.
        
        Args:
            query: Query text to embed
            
        Returns:
            1D numpy array of query embedding
            
        Raises:
            ValueError: If query is empty or embedding fails
        """
        if not query or not isinstance(query, str):
            raise ValueError("Query must be a non-empty string")

        try:
            normalized_query = " ".join(query.strip().lower().split())
            if self._query_cache_enabled:
                with self._query_cache_lock:
                    payload = self._query_cache.get(normalized_query)
                    if payload is not None:
                        created_at, cached_embedding = payload
                        if (time.monotonic() - created_at) <= self._query_cache_ttl_seconds:
                            self._query_cache.move_to_end(normalized_query)
                            return cached_embedding.copy()
                        del self._query_cache[normalized_query]

            # E5 models require 'query: ' prefix for query inputs
            prefixed_query = f"query: {query}"
            
            # GPU optimization: set to eval mode
            self.model.eval()
            
            embedding = self.model.encode(prefixed_query, convert_to_numpy=True)
            logger_embeddings.debug(f"✓ Query embedded (device: {self.device}, dim: {len(embedding)})")
            embedding_np = embedding.astype(np.float32)

            if self._query_cache_enabled:
                with self._query_cache_lock:
                    self._query_cache[normalized_query] = (time.monotonic(), embedding_np.copy())
                    self._query_cache.move_to_end(normalized_query)
                    if len(self._query_cache) > self._query_cache_max:
                        self._query_cache.popitem(last=False)

            return embedding_np
        except Exception as e:
            logger_embeddings.error(f"✗ Query embedding failed: {e}", exc_info=True)
            raise

    def clear_query_cache(self) -> None:
        """Clear query embedding cache for profiling/testing."""
        with self._query_cache_lock:
            self._query_cache.clear()

    def embed_batch(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """
        Embed multiple texts efficiently with error handling and GPU optimization.
        
        Args:
            texts: List of texts to embed
            batch_size: Batch size for processing. Auto-adjusted for GPU constraints.
            
        Returns:
            2D numpy array of embeddings (N x embedding_dim)
            
        Raises:
            ValueError: If texts is empty or embedding fails
        """
        if not texts or not isinstance(texts, list):
            raise ValueError("Texts must be a non-empty list")

        try:
            config = get_config()
            
            # Auto-adjust batch size for device constraints
            if self.device == "cpu":
                batch_size = min(batch_size, config.embeddings.cpu_batch_size)
            else:
                batch_size = min(batch_size, config.embeddings.gpu_batch_size)
            
            prefixed_texts = [f"passage: {text}" for text in texts]
            
            # GPU optimization: set to eval mode
            self.model.eval()
            
            logger_embeddings.info(f"Embedding {len(texts)} chunks (batch_size={batch_size}, device={self.device})")
            embeddings = self.model.encode(
                prefixed_texts,
                batch_size=batch_size,
                convert_to_numpy=True,
                show_progress_bar=False
            )
            
            logger_embeddings.info(f"✓ Batch embedding complete. Shape: {embeddings.shape}")
            return embeddings.astype(np.float32)
        except Exception as e:
            logger_embeddings.error(f"✗ Batch embedding failed: {e}", exc_info=True)
            raise

    def get_embedding_dimension(self) -> int:
        """Get the dimensionality of embeddings."""
        return self.embedding_dim


# Singleton instance for reuse
_engine_instance: EmbeddingEngine | None = None


def get_embedding_engine() -> EmbeddingEngine:
    """Get or create the global embedding engine instance."""
    global _engine_instance
    if _engine_instance is None:
        _engine_instance = EmbeddingEngine()
    return _engine_instance
