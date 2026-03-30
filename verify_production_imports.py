#!/usr/bin/env python3
"""
Verify that all production-refactored modules import correctly.
Tests:
- Config system loads
- Logging infrastructure initializes
- RAG modules import without errors
- Singleton instances created
"""

import sys
import traceback
from pathlib import Path

# Add project to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_imports():
    """Test all imports work correctly."""
    tests = [
        ("config", "from pramana_engine.config import get_config"),
        ("logging_setup", "from pramana_engine.logging_setup import logger_embeddings, logger_vector_store, logger_retrieval, logger_llm, logger_pipeline"),
        ("rag_prompts", "from pramana_engine.rag_prompts import get_system_prompt, get_cot_prompt, get_simple_prompt"),
        ("rag_persistence", "from pramana_engine.rag_persistence import VectorStorePersistence"),
        ("rag_embeddings", "from pramana_engine.rag_embeddings import EmbeddingEngine, get_embedding_engine"),
        ("vector_store", "from pramana_engine.vector_store import FAISSVectorStore, get_vector_store"),
        ("llm_integration", "from pramana_engine.llm_integration import MistralLLMEngine, get_llm_engine"),
        ("hybrid_retrieval", "from pramana_engine.hybrid_retrieval import hybrid_search, reciprocal_rank_fusion"),
        ("rag_pipeline", "from pramana_engine.rag_pipeline import RAGPipeline, get_rag_pipeline, rag_answer_question"),
    ]
    
    results = []
    for name, import_stmt in tests:
        try:
            exec(import_stmt)
            print(f"[PASS] {name:20} - OK")
            results.append((name, True, None))
        except Exception as e:
            print(f"[FAIL] {name:20} - {str(e)[:80]}")
            results.append((name, False, str(e)))
            traceback.print_exc()
    
    return results

def test_config():
    """Test config system."""
    try:
        from pramana_engine.config import get_config
        config = get_config()
        print(f"[PASS] Config loaded - device: {config.embeddings.device}")
        return True
    except Exception as e:
        print(f"[FAIL] Config failed: {e}")
        traceback.print_exc()
        return False

def test_logging():
    """Test logging system."""
    try:
        from pramana_engine.logging_setup import RAGLogger, logger_embeddings
        print("[PASS] Logging initialized")
        logger_embeddings.info("Test log message")
        return True
    except Exception as e:
        print(f"[FAIL] Logging failed: {e}")
        traceback.print_exc()
        return False

def test_prompts():
    """Test prompts system."""
    try:
        from pramana_engine.rag_prompts import get_system_prompt, get_cot_prompt
        system = get_system_prompt()
        cot = get_cot_prompt("test question", "test context")
        if system and cot:
            print(f"[PASS] Prompts valid - system_len={len(system)}, cot_len={len(cot)}")
            return True
        else:
            print("[FAIL] Prompts empty")
            return False
    except Exception as e:
        print(f"[FAIL] Prompts failed: {e}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("=" * 80)
    print("PRODUCTION IMPORT VERIFICATION")
    print("=" * 80)
    print()
    
    print("1. Testing module imports...")
    print("-" * 80)
    import_results = test_imports()
    
    print()
    print("2. Testing config system...")
    print("-" * 80)
    config_ok = test_config()
    
    print()
    print("3. Testing logging system...")
    print("-" * 80)
    logging_ok = test_logging()
    
    print()
    print("4. Testing prompts system...")
    print("-" * 80)
    prompts_ok = test_prompts()
    
    print()
    print("=" * 80)
    passed_imports = sum(1 for _, ok, _ in import_results if ok)
    total_imports = len(import_results)
    
    summary = [
        f"Import tests: {passed_imports}/{total_imports}",
        f"Config loaded: {'Yes' if config_ok else 'No'}",
        f"Logging ready: {'Yes' if logging_ok else 'No'}",
        f"Prompts valid: {'Yes' if prompts_ok else 'No'}",
    ]
    
    for line in summary:
        print(f"  {line}")
    
    print("=" * 80)
    
    if passed_imports == total_imports and config_ok and logging_ok and prompts_ok:
        print("\nALL PRODUCTION SYSTEMS VERIFIED!")
        sys.exit(0)
    else:
        print("\nSOME SYSTEMS FAILED VERIFICATION")
        sys.exit(1)
