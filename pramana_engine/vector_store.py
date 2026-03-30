"""
FAISS Vector Store for semantic similarity search.
Manages indexing and retrieval of knowledge base chunks with production error handling and persistence.
"""

from __future__ import annotations

import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Any, Optional

import faiss

from .rag_embeddings import get_embedding_engine
from .config import get_config
from .logging_setup import logger_vector_store
from .rag_persistence import VectorStorePersistence


class FAISSVectorStore:
    """FAISS-based vector store for fast semantic search with production error handling."""

    def __init__(self, embedding_dim: int = 384, load_cache: bool = True):
        """
        Initialize FAISS vector store with optional persistence.
        
        Args:
            embedding_dim: Dimensionality of embeddings (E5-Small = 384)
            load_cache: Try loading from cache if True
            
        Raises:
            RuntimeError: If vector store initialization fails
        """
        try:
            config = get_config()
            self.embedding_dim = embedding_dim
            self.index = faiss.IndexFlatL2(embedding_dim)
            self.metadata: List[Dict[str, Any]] = []
            self.persistence = VectorStorePersistence()
            
            logger_vector_store.info(f"Vector store initialized (dim={embedding_dim}, device={config.embeddings.device})")
            
            # Try loading from cache
            if load_cache:
                self._load_from_cache()
        except Exception as e:
            logger_vector_store.error(f"✗ Vector store initialization failed: {e}", exc_info=True)
            raise RuntimeError(f"Failed to initialize vector store: {e}")

    def add_chunks(self, chunks: List[Dict[str, Any]]) -> int:
        """
        Add knowledge base chunks to the vector store with error handling.
        
        Args:
            chunks: List of chunk dicts with 'id', 'text', 'source', 'tags', 'supports'
            
        Returns:
            Number of chunks successfully added
            
        Raises:
            ValueError: If chunks format is invalid
            RuntimeError: If embedding fails
        """
        if not chunks:
            logger_vector_store.debug("No chunks to add")
            return 0

        try:
            # Validate chunk format
            for chunk in chunks:
                if not isinstance(chunk, dict) or "text" not in chunk:
                    raise ValueError(f"Invalid chunk format: {chunk}")

            texts = [chunk["text"] for chunk in chunks]
            logger_vector_store.info(f"Embedding {len(texts)} chunks...")
            
            engine = get_embedding_engine()
            embeddings = engine.embed_batch(texts)

            # Check embedding validity
            if embeddings.shape[0] != len(chunks):
                raise RuntimeError(f"Embedding mismatch: got {embeddings.shape[0]}, expected {len(chunks)}")

            # Add to FAISS index
            self.index.add(embeddings)

            # Store metadata for retrieval
            self.metadata.extend(chunks)
            
            logger_vector_store.info(f"✓ Added {len(chunks)} chunks. Total: {len(self.metadata)}")
            
            # Auto-save to cache
            self.save_to_cache()
            
            return len(chunks)
        except Exception as e:
            logger_vector_store.error(f"✗ Failed to add chunks: {e}", exc_info=True)
            raise

    def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        Search for top-k similar chunks with error handling.
        
        Args:
            query: Query text
            k: Number of results to return
            
        Returns:
            List of dicts with 'id', 'text', 'source', 'distance', 'score'
            
        Raises:
            ValueError: If query is empty or k is invalid
            RuntimeError: If search fails
        """
        try:
            if not query or not isinstance(query, str):
                raise ValueError("Query must be a non-empty string")
            if k <= 0:
                raise ValueError("k must be > 0")
            
            if len(self.metadata) == 0:
                logger_vector_store.warning("Vector store is empty, no results")
                return []
            
            k = min(k, len(self.metadata))  # Ensure k doesn't exceed metadata
            engine = get_embedding_engine()
            query_embedding = engine.embed_query(query)

            # FAISS search returns (distances, indices)
            distances, indices = self.index.search(
                np.array([query_embedding]), k
            )

            results = []
            for idx, distance in zip(indices[0], distances[0]):
                if idx == -1:  # Invalid result
                    continue

                metadata = self.metadata[int(idx)]
                # Convert L2 distance to similarity score (0-1)
                # Lower distance = higher similarity
                similarity_score = 1.0 / (1.0 + distance)

                results.append({
                    "id": metadata["id"],
                    "text": metadata["text"],
                    "source": metadata.get("source", "unknown"),
                    "tags": metadata.get("tags", []),
                    "supports": metadata.get("supports", []),
                    "distance": float(distance),
                    "score": float(similarity_score),
                })

            logger_vector_store.debug(f"✓ Search returned {len(results)} results (k={k})")
            return results
        except Exception as e:
            logger_vector_store.error(f"✗ Search failed: {e}", exc_info=True)
            raise RuntimeError(f"Search failed: {e}")

    def batch_search(self, queries: List[str], k: int = 5) -> List[List[Dict[str, Any]]]:
        """
        Search for multiple queries with error handling.
        
        Args:
            queries: List of query texts
            k: Number of results per query
            
        Returns:
            List of result lists
            
        Raises:
            ValueError: If queries format is invalid
            RuntimeError: If batch search fails
        """
        try:
            if not queries or not isinstance(queries, list):
                raise ValueError("Queries must be a non-empty list")
            
            logger_vector_store.info(f"Batch searching {len(queries)} queries (k={k})")
            
            engine = get_embedding_engine()
            query_embeddings = np.array([
                engine.embed_query(q) for q in queries
            ])

            distances, indices = self.index.search(query_embeddings, k)

            all_results = []
            for dist_row, idx_row in zip(distances, indices):
                results = []
                for idx, distance in zip(idx_row, dist_row):
                    if idx == -1:
                        continue

                    metadata = self.metadata[int(idx)]
                    similarity_score = 1.0 / (1.0 + distance)

                    results.append({
                        "id": metadata["id"],
                        "text": metadata["text"],
                        "source": metadata.get("source", "unknown"),
                        "tags": metadata.get("tags", []),
                        "supports": metadata.get("supports", []),
                        "distance": float(distance),
                        "score": float(similarity_score),
                    })

                all_results.append(results)

            logger_vector_store.debug(f"✓ Batch search complete. Results: {len(all_results)}")
            return all_results
        except Exception as e:
            logger_vector_store.error(f"✗ Batch search failed: {e}", exc_info=True)
            raise RuntimeError(f"Batch search failed: {e}")

    def size(self) -> int:
        """Get number of indexed chunks with logging."""
        count = self.index.ntotal
        logger_vector_store.debug(f"Vector store size: {count} chunks")
        return count

    def save(self, path: str) -> bool:
        """
        Save index and metadata to disk with error handling.
        
        Args:
            path: Directory path to save to
            
        Returns:
            Success status
            
        Raises:
            RuntimeError: If save fails
        """
        try:
            path_obj = Path(path)
            path_obj.mkdir(parents=True, exist_ok=True)

            # Save FAISS index
            faiss.write_index(
                self.index,
                str(path_obj / "faiss.index")
            )

            # Save metadata
            with open(path_obj / "metadata.json", "w") as f:
                json.dump(self.metadata, f, indent=2)
            
            logger_vector_store.info(f"✓ Vector store saved to {path}")
            return True
        except Exception as e:
            logger_vector_store.error(f"✗ Failed to save vector store: {e}", exc_info=True)
            raise RuntimeError(f"Save failed: {e}")

    @staticmethod
    def load(path: str) -> Optional[FAISSVectorStore]:
        """
        Load index and metadata from disk with error handling.
        
        Args:
            path: Directory path to load from
            
        Returns:
            FAISSVectorStore instance or None if load fails
            
        Raises:
            RuntimeError: If load fails
        """
        try:
            path_obj = Path(path)
            
            if not path_obj.exists():
                logger_vector_store.warning(f"Cache path does not exist: {path}")
                return None

            # Load FAISS index
            index = faiss.read_index(str(path_obj / "faiss.index"))

            # Load metadata
            with open(path_obj / "metadata.json") as f:
                metadata = json.load(f)

            # Reconstruct store
            store = FAISSVectorStore(embedding_dim=index.d, load_cache=False)
            store.index = index
            store.metadata = metadata
            
            logger_vector_store.info(f"✓ Vector store loaded from {path} ({len(metadata)} chunks)")
            return store
        except Exception as e:
            logger_vector_store.warning(f"Failed to load vector store from cache: {e}")
            return None


    def save_to_cache(self) -> None:
        """Save vector store to cache directory."""
        try:
            config = get_config()
            self.persistence.save_index(
                self.index,
                self.metadata,
                {
                    "embedding_dim": self.embedding_dim,
                    "index_type": config.vector_store.index_type,
                },
            )
            logger_vector_store.debug("✓ Vector store cached")
        except Exception as e:
            logger_vector_store.warning(f"Failed to cache vector store: {e}")

    def _load_from_cache(self) -> bool:
        """
        Load vector store from cache if available.
        
        Returns:
            True if successfully loaded, False otherwise
        """
        try:
            result = self.persistence.load_index()
            if result:
                self.index, self.metadata, _config_info = result
                logger_vector_store.info(f"✓ Loaded cache ({len(self.metadata)} chunks)")
                return True
            return False
        except Exception as e:
            logger_vector_store.debug(f"Cache load failed (expected if first run): {e}")
            return False


# Singleton instance
_vector_store_instance: FAISSVectorStore | None = None


def get_vector_store() -> FAISSVectorStore:
    """Get or create the global vector store instance with error handling."""
    global _vector_store_instance
    try:
        if _vector_store_instance is None:
            logger_vector_store.info("Initializing vector store...")
            _vector_store_instance = FAISSVectorStore()
        return _vector_store_instance
    except Exception as e:
        logger_vector_store.error(f"✗ Failed to get vector store: {e}", exc_info=True)
        raise
    return _vector_store_instance


def initialize_vector_store(chunks: List[Dict[str, Any]]) -> FAISSVectorStore:
    """
    Initialize vector store with knowledge base chunks.
    
    Args:
        chunks: Knowledge base chunks
        
    Returns:
        Initialized vector store
    """
    store = get_vector_store()
    if store.size() == 0:
        store.add_chunks(chunks)
    return store
