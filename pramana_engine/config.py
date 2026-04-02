"""
Configuration management for RAG+LLM pipeline.
Centralized settings for production deployment.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Literal
from pathlib import Path


@dataclass
class EmbeddingConfig:
    """Embedding model configuration."""
    model_name: str = "intfloat/e5-small-v2"
    batch_size: int = 32
    cpu_batch_size: int = 8
    gpu_batch_size: int = 32
    device: str = "cuda"  # "cuda" or "cpu" - auto-detects GPU
    cache_embeddings: bool = True
    query_cache_size: int = 512
    query_cache_ttl_seconds: int = 300
    normalize_embeddings: bool = True


_PROJECT_ROOT = Path(__file__).parent.parent


@dataclass
class VectorStoreConfig:
    """FAISS vector store configuration."""
    persist_dir: str = str(_PROJECT_ROOT / "vector_store_cache")
    auto_save: bool = True
    load_on_startup: bool = True
    index_type: str = "flatl2"  # or "ivf" for large indices


@dataclass
class LLMConfig:
    """Mistral 7B LLM configuration."""
    model_name: str = "mistral:7b"
    ollama_host: str = field(
        default_factory=lambda: os.environ.get("OLLAMA_HOST", "http://localhost:11434")
    )
    temperature: float = 0.2  # Lower = more deterministic for logic
    top_p: float = 0.9
    max_tokens: int = 512
    timeout: int = 120
    use_quantized: bool = True  # mistral:7b-q4 for lower VRAM
    device_map: str = "auto"  # Auto GPU/CPU


@dataclass
class OpenAIConfig:
    """OpenAI API configuration for LLM fallback."""
    api_key: str = field(default_factory=lambda: os.environ.get("OPENAI_API_KEY", ""))
    model_name: str = "gpt-4o-mini"
    temperature: float = 0.0
    max_tokens: int = 512
    timeout: int = 60


@dataclass
class RetrievalConfig:
    """Hybrid retrieval configuration."""
    top_k: int = 10
    reciprocal_rank_k: float = 60.0
    semantic_weight: float = 0.6
    keyword_weight: float = 0.4
    min_score_threshold: float = 0.1
    cache_max_entries: int = 128
    cache_ttl_seconds: int = 300


@dataclass
class PramanaConfig:
    """Pramana verification configuration."""
    min_retrieval_support: float = 0.15
    min_rule_consistency: float = 0.6
    max_contradiction_strength: float = 0.45
    apply_belief_revision: bool = True


@dataclass
class LoggingConfig:
    """Logging configuration."""
    log_level: str = "INFO"
    log_file: str = "pramana_rag.log"
    max_log_size: int = 10_000_000  # 10MB
    backup_count: int = 5
    # Set to "json" (via LOG_FORMAT env var) to emit structured JSON to the log file.
    log_format: str = field(default_factory=lambda: os.environ.get("LOG_FORMAT", "text"))


@dataclass
class APIConfig:
    """Flask API configuration."""
    host: str = "127.0.0.1"
    port: int = 5000
    debug: bool = False
    max_request_size: int = 16 * 1024 * 1024  # 16MB
    request_timeout: int = 180
    demo_mode: bool = field(
        default_factory=lambda: os.environ.get("PRAMANA_DEMO_MODE", "0").lower() in ("1", "true", "yes")
    )


class ProductionConfig:
    """Production configuration (all components)."""

    def __init__(self):
        """Initialize production config with optimizations."""
        self.embeddings = EmbeddingConfig(
            device=self._detect_device(),
            batch_size=16 if self._has_gpu() else 8,
            cpu_batch_size=8,
            gpu_batch_size=16 if self._has_gpu() else 8,
        )
        self.vector_store = VectorStoreConfig()
        self.llm = LLMConfig(
            model_name=self._select_model(),
            temperature=0.0,  # Fully deterministic: same question always same answer
        )
        self.retrieval = RetrievalConfig()
        self.pramana = PramanaConfig()
        self.logging = LoggingConfig()
        self.api = APIConfig()

        # Create vector store cache directory (persist_dir is always absolute)
        Path(self.vector_store.persist_dir).mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _detect_device() -> str:
        """Auto-detect GPU availability."""
        try:
            import torch
            if torch.cuda.is_available():
                return "cuda"
        except ImportError:
            pass
        return "cpu"

    @staticmethod
    def _has_gpu() -> bool:
        """Check if CUDA GPU is available."""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False

    @staticmethod
    def _has_low_vram() -> bool:
        """Check if GPU has < 6GB VRAM (use quantized model)."""
        try:
            import torch
            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    vram_gb = torch.cuda.get_device_properties(i).total_memory / 1e9
                    if vram_gb < 6:
                        return True
        except ImportError:
            pass
        return False

    @staticmethod
    def _get_vram_gb() -> float:
        """Return total VRAM in GB of the first CUDA device, or 0 if no GPU."""
        try:
            import torch
            if torch.cuda.is_available():
                return torch.cuda.get_device_properties(0).total_memory / 1e9
        except ImportError:
            pass
        return 0.0

    def _select_model(self) -> str:
        """Select the best Ollama model based on available VRAM."""
        vram = self._get_vram_gb()
        if vram <= 0:
            # No GPU — use smallest model for CPU
            return "tinyllama"
        elif vram < 4.0:
            # Very low VRAM (e.g. 2-3 GB)
            return "qwen2:1.5b"
        elif vram < 6.0:
            # Low VRAM (e.g. 4 GB) — phi3:mini needs ~2.3 GB
            return "phi3:mini"
        else:
            # 6 GB+ — mistral fits
            return "mistral:7b"

    def to_dict(self) -> dict:
        """Convert config to dict for logging."""
        return {
            "embeddings": {
                "model": self.embeddings.model_name,
                "device": self.embeddings.device,
                "batch_size": self.embeddings.batch_size,
            },
            "llm": {
                "model": self.llm.model_name,
                "temperature": self.llm.temperature,
            },
            "vector_store": {
                "persist_dir": self.vector_store.persist_dir,
                "auto_save": self.vector_store.auto_save,
            },
        }


# Global instance
_config: ProductionConfig | None = None


def get_config() -> ProductionConfig:
    """Get or create global config instance."""
    global _config
    if _config is None:
        _config = ProductionConfig()
    return _config
