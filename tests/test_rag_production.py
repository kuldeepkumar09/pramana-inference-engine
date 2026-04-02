from types import SimpleNamespace
import logging
from collections import OrderedDict
from threading import RLock

import numpy as np

import pramana_engine.hybrid_retrieval as hybrid_mod
from pramana_engine.rag_pipeline import RAGPipeline
from pramana_engine.hybrid_retrieval import (
    clear_hybrid_retrieval_cache,
    filter_by_pramana,
    reciprocal_rank_fusion,
    rerank_by_target_pramana,
)
from pramana_engine.llm_integration import MistralLLMEngine
from pramana_engine.logging_setup import ConsoleSafeFormatter
from pramana_engine.rag_embeddings import EmbeddingEngine
from pramana_engine.web import create_app


def test_rrf_fuses_keyword_and_semantic_scores():
    keyword = [
        {"id": "A", "text": "k1"},
        {"id": "B", "text": "k2"},
    ]
    semantic = [
        {"id": "B", "text": "s1", "distance": 0.3},
        {"id": "C", "text": "s2", "distance": 0.7},
    ]

    fused = reciprocal_rank_fusion(keyword, semantic, k=60.0)

    assert len(fused) == 3
    # B appears in both sources, so it should rank highest.
    assert fused[0]["id"] == "B"
    assert fused[0]["sources"] == "keyword+semantic"


def test_filter_and_rerank_by_pramana_behaves_consistently():
    chunks = [
        {"id": "1", "supports": ["perception"], "fused_score": 0.2},
        {"id": "2", "supports": ["inference"], "fused_score": 0.2},
        {"id": "3", "supports": ["testimony"], "fused_score": 0.4},
    ]

    filtered = filter_by_pramana(chunks, ["inference", "testimony"])
    assert [c["id"] for c in filtered] == ["2", "3"]

    reranked = rerank_by_target_pramana(filtered, "inference")
    assert reranked[0]["id"] == "2"
    assert reranked[0]["fused_score"] > reranked[1]["fused_score"]


def test_extract_confidence_clamps_to_unit_interval():
    # Avoid full engine initialization; this method has no dependency on client/model state.
    engine = MistralLLMEngine.__new__(MistralLLMEngine)

    assert engine._extract_confidence("0.87") == 0.87
    assert engine._extract_confidence("1.7") == 1.0
    assert engine._extract_confidence("-2") == 0.5


def test_dashboard_route_loads():
    app = create_app()
    app.testing = True
    client = app.test_client()

    response = client.get("/dashboard")

    assert response.status_code == 200
    assert b"RAG Operations Dashboard" in response.data


def test_rag_status_endpoint_works_with_mocked_pipeline(monkeypatch):
    app = create_app()
    app.testing = True
    client = app.test_client()

    class FakeVectorStore:
        def size(self):
            return 42

    class FakeLLM:
        def health_check(self):
            return True

    class FakePipeline:
        def __init__(self):
            self._initialized = True
            self.llm_engine = FakeLLM()
            self.vector_store = FakeVectorStore()

        def initialize(self):
            self._initialized = True

    monkeypatch.setattr("pramana_engine.web.get_rag_pipeline", lambda: FakePipeline())

    response = client.get("/api/rag/status")
    assert response.status_code == 200

    data = response.get_json()
    assert data["pipeline_initialized"] is True
    assert data["vector_store_size"] == 42
    assert data["llm_healthy"] is True


def test_rag_cache_clear_endpoint_clears_all_scopes(monkeypatch):
    app = create_app()
    app.testing = True
    client = app.test_client()

    calls = {"pipeline": 0, "hybrid": 0, "embedding": 0}

    class FakePipeline:
        _initialized = True
        llm_engine = SimpleNamespace(health_check=lambda: True)
        vector_store = SimpleNamespace(size=lambda: 0)

        def clear_runtime_caches(self):
            calls["pipeline"] += 1

    class FakeEmbeddingEngine:
        def clear_query_cache(self):
            calls["embedding"] += 1

    monkeypatch.setattr("pramana_engine.web.get_rag_pipeline", lambda: FakePipeline())
    monkeypatch.setattr("pramana_engine.web.clear_hybrid_retrieval_cache", lambda: calls.__setitem__("hybrid", calls["hybrid"] + 1))
    monkeypatch.setattr("pramana_engine.web.get_embedding_engine", lambda: FakeEmbeddingEngine())

    response = client.post("/api/rag/cache/clear", json={"scope": "all"})

    assert response.status_code == 200
    data = response.get_json()
    assert data["ok"] is True
    assert set(data["cleared"]) == {"pipeline_retrieval_cache", "hybrid_retrieval_cache", "query_embedding_cache"}
    assert calls == {"pipeline": 1, "hybrid": 1, "embedding": 1}


def test_rag_cache_clear_endpoint_rejects_invalid_scope():
    app = create_app()
    app.testing = True
    client = app.test_client()

    response = client.post("/api/rag/cache/clear", json={"scope": "invalid"})

    assert response.status_code == 400
    assert "error" in response.get_json()


def test_rag_cache_clear_endpoint_blocks_non_local_without_token(monkeypatch):
    app = create_app()
    app.testing = True
    client = app.test_client()

    monkeypatch.delenv("PRAMANA_ADMIN_TOKEN", raising=False)
    response = client.post(
        "/api/rag/cache/clear",
        json={"scope": "all"},
        environ_overrides={"REMOTE_ADDR": "203.0.113.10"},
    )

    assert response.status_code == 403
    assert "error" in response.get_json()


def test_rag_cache_clear_endpoint_requires_header_when_token_is_set(monkeypatch):
    app = create_app()
    app.testing = True
    client = app.test_client()

    calls = {"pipeline": 0, "hybrid": 0, "embedding": 0}

    class FakePipeline:
        _initialized = True
        llm_engine = SimpleNamespace(health_check=lambda: True)
        vector_store = SimpleNamespace(size=lambda: 0)

        def clear_runtime_caches(self):
            calls["pipeline"] += 1

    class FakeEmbeddingEngine:
        def clear_query_cache(self):
            calls["embedding"] += 1

    monkeypatch.setenv("PRAMANA_ADMIN_TOKEN", "secret-token")
    monkeypatch.setattr("pramana_engine.web.get_rag_pipeline", lambda: FakePipeline())
    monkeypatch.setattr("pramana_engine.web.clear_hybrid_retrieval_cache", lambda: calls.__setitem__("hybrid", calls["hybrid"] + 1))
    monkeypatch.setattr("pramana_engine.web.get_embedding_engine", lambda: FakeEmbeddingEngine())

    unauthorized = client.post(
        "/api/rag/cache/clear",
        json={"scope": "all"},
        environ_overrides={"REMOTE_ADDR": "203.0.113.10"},
    )
    assert unauthorized.status_code == 401

    authorized = client.post(
        "/api/rag/cache/clear",
        json={"scope": "all"},
        headers={"X-Pramana-Admin-Token": "secret-token"},
        environ_overrides={"REMOTE_ADDR": "203.0.113.10"},
    )
    assert authorized.status_code == 200
    assert calls == {"pipeline": 1, "hybrid": 1, "embedding": 1}


def test_rag_answer_endpoint_uses_pipeline_result(monkeypatch):
    app = create_app()
    app.testing = True
    client = app.test_client()

    class FakePipeline:
        def __init__(self):
            self._initialized = True
            self.llm_engine = SimpleNamespace(health_check=lambda: True)
            self.vector_store = SimpleNamespace(size=lambda: 3)

        def initialize(self):
            self._initialized = True

        def answer_question(self, question, pramana_types=None, use_llm=True, use_reasoning_chain=False):
            return {
                "question": question,
                "answer": "Vyapti is the invariable relation.",
                "answer_source": "llm_reasoning",
                "confidence": 0.91,
                "epistemic_status": "justified",
                "rag_chunks": [{"id": "K1", "text": "chunk", "score": 0.8}],
            }

    monkeypatch.setattr("pramana_engine.web.get_rag_pipeline", lambda: FakePipeline())

    response = client.post(
        "/api/rag/answer",
        json={"question": "Why is vyapti required?", "useLLM": True, "useReasoningChain": False},
    )

    assert response.status_code == 200
    data = response.get_json()
    assert data["answer_source"] == "llm_reasoning"
    assert data["confidence"] == 0.91
    assert data["epistemic_status"] == "justified"


def test_hybrid_search_uses_cache_on_repeat(monkeypatch):
    clear_hybrid_retrieval_cache()

    calls = {"semantic": 0}

    class FakeVectorStore:
        def search(self, query, k=10):
            calls["semantic"] += 1
            return [
                {
                    "id": "S1",
                    "text": "Hetu and vyapti relation",
                    "source": "mock",
                    "supports": ["inference"],
                    "tags": ["hetu", "vyapti"],
                    "distance": 0.2,
                    "score": 0.83,
                }
            ]

    monkeypatch.setattr(hybrid_mod, "get_vector_store", lambda: FakeVectorStore())
    monkeypatch.setattr(
        hybrid_mod,
        "_get_knowledge_index",
        lambda: [
            {
                "chunk": {
                    "id": "K1",
                    "text": "Anumana depends on vyapti.",
                    "source": "mock",
                    "tags": ["anumana", "vyapti"],
                    "supports": ["inference"],
                },
                "token_set": {"anumana", "depends", "on", "vyapti"},
                "tag_set": {"anumana", "vyapti"},
                "supports_set": {"inference"},
            }
        ],
    )

    first = hybrid_mod.hybrid_search("What is vyapti in anumana?", ["inference"], k=2)
    second = hybrid_mod.hybrid_search("What is vyapti in anumana?", ["inference"], k=2)

    assert first == second
    assert calls["semantic"] == 1


def test_pipeline_retrieval_delegates_to_hybrid_search(monkeypatch):
    # Pipeline-level cache was removed (M10). Caching is now handled entirely
    # by hybrid_retrieval._HYBRID_CACHE. Each pipeline call passes through to
    # hybrid_search; deduplication happens inside hybrid_retrieval's own LRU cache.
    pipeline = RAGPipeline()
    pipeline._initialized = True

    calls = {"hybrid": 0}

    def fake_hybrid_search(question, pramana_types=None, k=10):
        calls["hybrid"] += 1
        return [
            {
                "id": "X1",
                "text": "Mock retrieval chunk",
                "source": "mock",
                "supports": ["inference"],
                "tags": ["mock"],
                "fused_score": 0.75,
            }
        ]

    monkeypatch.setattr("pramana_engine.rag_pipeline.hybrid_search", fake_hybrid_search)

    a = pipeline.search_only("Why is vyapti needed?", ["inference"], k=5)
    b = pipeline.search_only("Why is vyapti needed?", ["inference"], k=5)

    assert a == b
    # Pipeline delegates every call to hybrid_search (caching is inside hybrid_retrieval)
    assert calls["hybrid"] == 2


def test_hybrid_search_cache_invalidates_when_corpus_version_changes(monkeypatch):
    clear_hybrid_retrieval_cache()

    calls = {"semantic": 0}

    class FakeVectorStore:
        def search(self, query, k=10):
            calls["semantic"] += 1
            return [{"id": "S1", "text": "x", "supports": ["inference"], "distance": 0.2}]

    versions = iter(["v1", "v2"])
    monkeypatch.setattr(hybrid_mod, "get_external_corpus_version", lambda: next(versions))
    monkeypatch.setattr(hybrid_mod, "get_vector_store", lambda: FakeVectorStore())
    monkeypatch.setattr(
        hybrid_mod,
        "_get_knowledge_index",
        lambda: [
            {
                "chunk": {"id": "K1", "text": "vyapti", "source": "mock", "tags": [], "supports": ["inference"]},
                "token_set": {"vyapti"},
                "tag_set": set(),
                "supports_set": {"inference"},
            }
        ],
    )

    hybrid_mod.hybrid_search("vyapti?", ["inference"], k=1)
    hybrid_mod.hybrid_search("vyapti?", ["inference"], k=1)

    assert calls["semantic"] == 2


def test_pipeline_always_delegates_to_hybrid_search(monkeypatch):
    # Pipeline-level cache was removed (M10). Corpus-version-aware caching is
    # now entirely inside hybrid_retrieval._HYBRID_CACHE. Every pipeline call
    # passes straight to hybrid_search regardless of corpus version.
    pipeline = RAGPipeline()
    pipeline._initialized = True

    calls = {"hybrid": 0}

    def fake_hybrid_search(question, pramana_types=None, k=10):
        calls["hybrid"] += 1
        return [{"id": "X1", "text": "chunk", "supports": ["inference"], "fused_score": 0.7}]

    monkeypatch.setattr("pramana_engine.rag_pipeline.hybrid_search", fake_hybrid_search)

    pipeline.search_only("why vyapti", ["inference"], k=3)
    pipeline.search_only("why vyapti", ["inference"], k=3)

    assert calls["hybrid"] == 2


def test_console_formatter_replaces_unicode_markers_for_non_utf_stream(monkeypatch):
    monkeypatch.delenv("PYTHONIOENCODING", raising=False)
    monkeypatch.setattr(ConsoleSafeFormatter, "_is_utf8_console", staticmethod(lambda: False))

    formatter = ConsoleSafeFormatter()
    record = logging.LogRecord(
        name="pramana.rag.test",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="✓ done -> ✗ fail",
        args=(),
        exc_info=None,
    )

    out = formatter.format(record)
    assert "[OK]" in out
    assert "[ERR]" in out


def test_query_embedding_cache_reuses_previous_embedding():
    class FakeModel:
        def __init__(self):
            self.calls = 0

        def eval(self):
            return None

        def encode(self, text, convert_to_numpy=True):
            self.calls += 1
            return np.array([1.0, 2.0, 3.0], dtype=np.float32)

    engine = EmbeddingEngine.__new__(EmbeddingEngine)
    engine.device = "cpu"
    engine.model = FakeModel()
    engine._query_cache_enabled = True
    engine._query_cache_max = 8
    engine._query_cache_ttl_seconds = 300
    engine._query_cache = OrderedDict()
    engine._query_cache_lock = RLock()

    first = engine.embed_query("What is vyapti?")
    second = engine.embed_query("What is vyapti?")

    assert engine.model.calls == 1
    assert np.array_equal(first, second)


def test_fallacy_heuristic_detects_hetvabhasa_label():
    chunks = [
        {
            "id": "K1",
            "text": "According to Naiyayikas there are five fallacies: savyabhicara, viruddha, satpratipaksha, asiddha and badhita.",
            "source": "mock",
            "fused_score": 0.9,
        }
    ]

    answer = RAGPipeline._heuristic_answer_from_chunks(
        "Which specific type of logical fallacy (Hetvabhasa) is committed here according to Nyaya?",
        chunks,
    )

    assert answer is not None
    assert "Hetvabhasa" in answer
    assert "Savyabhicara" in answer


def test_pipeline_uses_symbolic_fallback_when_llm_answer_is_unsupported(monkeypatch):
    pipeline = RAGPipeline()
    pipeline._initialized = True

    class WeakLLM:
        def generate_answer(self, question, chunks):
            return "This is definitely correct without textual evidence.", "mock"

    pipeline.llm_engine = WeakLLM()
    pipeline._get_cached_retrieval = lambda question, pramana_types, k: [
        {
            "id": "K1",
            "text": "Nyaya describes five major hetvabhasa types: savyabhicara, viruddha, satpratipaksha, asiddha and badhita.",
            "source": "mock",
            "supports": ["inference"],
            "tags": ["hetvabhasa"],
            "fused_score": 0.88,
        }
    ]

    monkeypatch.setattr(
        "pramana_engine.rag_pipeline._run_symbolic_verifier",
        lambda **kwargs: {
            "confidence_decomposition": {"final_confidence": 0.84},
            "belief_revision": {"final_status": "justified"},
            "constraints": [],
            "violated_constraints": [],
        },
    )

    result = pipeline.answer_question(
        question="Which specific type of logical fallacy (Hetvabhasa) is committed here according to Nyaya?",
        pramana_types=["inference"],
        use_llm=True,
        use_reasoning_chain=False,
    )

    assert result["answer_source"] == "symbolic_fallback"
    assert "Hetvabhasa" in result["answer"]


def test_pipeline_rejects_llm_fallacy_answer_without_canonical_labels(monkeypatch):
    pipeline = RAGPipeline()
    pipeline._initialized = True

    class WeakLLM:
        def generate_answer(self, question, chunks):
            bad = "The five categories are Pratyaksahetu, Upamanahetu, Sabdahetu, Arthapattihetu, and Anupalabdhihetu."
            return bad, "mock"

    pipeline.llm_engine = WeakLLM()
    pipeline._get_cached_retrieval = lambda question, pramana_types, k: [
        {
            "id": "K1",
            "text": "Nyaya lists five hetvabhasa: savyabhicara, viruddha, satpratipaksha, asiddha and badhita.",
            "source": "mock",
            "supports": ["inference"],
            "tags": ["hetvabhasa"],
            "fused_score": 0.91,
        }
    ]

    monkeypatch.setattr(
        "pramana_engine.rag_pipeline._run_symbolic_verifier",
        lambda **kwargs: {
            "confidence_decomposition": {"final_confidence": 0.84},
            "belief_revision": {"final_status": "justified"},
            "constraints": [],
            "violated_constraints": [],
        },
    )

    result = pipeline.answer_question(
        question="In Nyaya, what are the five major Hetvabhasa categories?",
        pramana_types=["inference"],
        use_llm=True,
        use_reasoning_chain=False,
    )

    assert result["answer_source"] == "symbolic_fallback"
    assert "Savyabhicara" in result["answer"]


def test_pipeline_fallacy_inference_question_returns_specific_label(monkeypatch):
    pipeline = RAGPipeline()
    pipeline._initialized = True

    class WeakLLM:
        def generate_answer(self, question, chunks):
            return "Unreliable generic answer.", "mock"

    pipeline.llm_engine = WeakLLM()
    pipeline._get_cached_retrieval = lambda question, pramana_types, k: [
        {
            "id": "K1",
            "text": "A hetu that deviates across instances is called savyabhicara (anaikantika).",
            "source": "mock",
            "supports": ["inference"],
            "tags": ["hetvabhasa"],
            "fused_score": 0.9,
        }
    ]

    monkeypatch.setattr(
        "pramana_engine.rag_pipeline._run_symbolic_verifier",
        lambda **kwargs: {
            "confidence_decomposition": {"final_confidence": 0.84},
            "belief_revision": {"final_status": "justified"},
            "constraints": [],
            "violated_constraints": [],
        },
    )

    result = pipeline.answer_question(
        question="The reason proves both presence and absence because the hetu is inconsistent. Which hetvabhasa is this?",
        pramana_types=["inference"],
        use_llm=True,
        use_reasoning_chain=False,
    )

    assert result["answer_source"] == "symbolic_fallback"
    assert "Savyabhicara" in result["answer"]
    assert "likely classification" in result["answer"]


def test_pipeline_fallacy_question_uses_question_clues_when_chunks_are_generic(monkeypatch):
    pipeline = RAGPipeline()
    pipeline._initialized = True

    class WeakLLM:
        def generate_answer(self, question, chunks):
            return "Generic and ungrounded output.", "mock"

    pipeline.llm_engine = WeakLLM()
    pipeline._get_cached_retrieval = lambda question, pramana_types, k: [
        {
            "id": "K1",
            "text": "Nyaya discusses inferential defects in detail.",
            "source": "mock",
            "supports": ["inference"],
            "tags": ["hetvabhasa"],
            "fused_score": 0.7,
        }
    ]

    monkeypatch.setattr(
        "pramana_engine.rag_pipeline._run_symbolic_verifier",
        lambda **kwargs: {
            "confidence_decomposition": {"final_confidence": 0.84},
            "belief_revision": {"final_status": "justified"},
            "constraints": [],
            "violated_constraints": [],
        },
    )

    result = pipeline.answer_question(
        question="The hetu is inconsistent because it supports both presence and absence. Which hetvabhasa is this?",
        pramana_types=["inference"],
        use_llm=True,
        use_reasoning_chain=False,
    )

    assert result["answer_source"] == "symbolic_fallback"
    assert "Savyabhicara" in result["answer"]


def test_pipeline_debate_fault_question_uses_symbolic_fallback_label(monkeypatch):
    pipeline = RAGPipeline()
    pipeline._initialized = True

    class WeakLLM:
        def generate_answer(self, question, chunks):
            return "Ungrounded answer without Nyaya debate labels.", "mock"

    pipeline.llm_engine = WeakLLM()
    pipeline._get_cached_retrieval = lambda question, pramana_types, k: [
        {
            "id": "K1",
            "text": "Chala is a debate defect involving quibbling and twisting the intended meaning.",
            "source": "mock",
            "supports": ["inference"],
            "tags": ["nyaya", "debate"],
            "fused_score": 0.9,
        }
    ]

    monkeypatch.setattr(
        "pramana_engine.rag_pipeline._run_symbolic_verifier",
        lambda **kwargs: {
            "confidence_decomposition": {"final_confidence": 0.81},
            "belief_revision": {"final_status": "justified"},
            "constraints": [],
            "violated_constraints": [],
        },
    )

    result = pipeline.answer_question(
        question="Which Nyaya debate fault applies when someone intentionally twists the opponent's words?",
        pramana_types=["inference"],
        use_llm=True,
        use_reasoning_chain=False,
    )

    assert result["answer_source"] == "symbolic_fallback"
    assert "Chala" in result["answer"]


def test_pipeline_debate_fault_question_uses_question_clues_when_chunks_generic(monkeypatch):
    pipeline = RAGPipeline()
    pipeline._initialized = True

    class WeakLLM:
        def generate_answer(self, question, chunks):
            return "Ungrounded answer without debate labels.", "mock"

    pipeline.llm_engine = WeakLLM()
    pipeline._get_cached_retrieval = lambda question, pramana_types, k: [
        {
            "id": "K1",
            "text": "Nyaya discusses multiple debate defects and refutation techniques.",
            "source": "mock",
            "supports": ["inference"],
            "tags": ["nyaya", "debate"],
            "fused_score": 0.6,
        }
    ]

    monkeypatch.setattr(
        "pramana_engine.rag_pipeline._run_symbolic_verifier",
        lambda **kwargs: {
            "confidence_decomposition": {"final_confidence": 0.81},
            "belief_revision": {"final_status": "justified"},
            "constraints": [],
            "violated_constraints": [],
        },
    )

    result = pipeline.answer_question(
        question="In debate, this is pure refutation with no own thesis. Which Nyaya defect is this?",
        pramana_types=["inference"],
        use_llm=True,
        use_reasoning_chain=False,
    )

    assert result["answer_source"] == "symbolic_fallback"
    assert "Vitanda" in result["answer"]


def test_pipeline_rejects_debate_fault_llm_answer_that_conflicts_with_cue(monkeypatch):
    pipeline = RAGPipeline()
    pipeline._initialized = True

    class WeakLLM:
        def generate_answer(self, question, chunks):
            wrong = "This is Jati due to misrepresentation."
            return wrong, "mock"

    pipeline.llm_engine = WeakLLM()
    pipeline._get_cached_retrieval = lambda question, pramana_types, k: [
        {
            "id": "K1",
            "text": "Nyaya debate defects include chala, jati, nigrahasthana, and vitanda.",
            "source": "mock",
            "supports": ["inference"],
            "tags": ["nyaya", "debate"],
            "fused_score": 0.8,
        }
    ]

    monkeypatch.setattr(
        "pramana_engine.rag_pipeline._run_symbolic_verifier",
        lambda **kwargs: {
            "confidence_decomposition": {"final_confidence": 0.81},
            "belief_revision": {"final_status": "justified"},
            "constraints": [],
            "violated_constraints": [],
        },
    )

    result = pipeline.answer_question(
        question="Which Nyaya debate fault applies when someone intentionally twists the opponent's words?",
        pramana_types=["inference"],
        use_llm=True,
        use_reasoning_chain=False,
    )

    assert result["answer_source"] == "symbolic_fallback"
    assert "Chala" in result["answer"]


def test_pipeline_debate_mode_question_uses_symbolic_fallback(monkeypatch):
    pipeline = RAGPipeline()
    pipeline._initialized = True

    class WeakLLM:
        def generate_answer(self, question, chunks):
            return "Generic debate answer without mode labels.", "mock"

    pipeline.llm_engine = WeakLLM()
    pipeline._get_cached_retrieval = lambda question, pramana_types, k: [
        {
            "id": "K1",
            "text": "Vada is an honest debate aimed at discovering truth through valid reasoning.",
            "source": "mock",
            "supports": ["inference"],
            "tags": ["nyaya", "debate"],
            "fused_score": 0.9,
        }
    ]

    monkeypatch.setattr(
        "pramana_engine.rag_pipeline._run_symbolic_verifier",
        lambda **kwargs: {
            "confidence_decomposition": {"final_confidence": 0.82},
            "belief_revision": {"final_status": "justified"},
            "constraints": [],
            "violated_constraints": [],
        },
    )

    result = pipeline.answer_question(
        question="In a truth-seeking debate, what is this type of Nyaya engagement called?",
        pramana_types=["inference"],
        use_llm=True,
        use_reasoning_chain=False,
    )

    assert result["answer_source"] == "symbolic_fallback"
    assert "Vada" in result["answer"]


def test_pipeline_debate_mode_uses_question_clues_for_jalpa(monkeypatch):
    pipeline = RAGPipeline()
    pipeline._initialized = True

    class WeakLLM:
        def generate_answer(self, question, chunks):
            return "Generic debate without mode label.", "mock"

    pipeline.llm_engine = WeakLLM()
    pipeline._get_cached_retrieval = lambda question, pramana_types, k: [
        {
            "id": "K1",
            "text": "Nyaya distinguishes debate types by their goals and allowed techniques.",
            "source": "mock",
            "supports": ["inference"],
            "tags": ["nyaya", "debate"],
            "fused_score": 0.7,
        }
    ]

    monkeypatch.setattr(
        "pramana_engine.rag_pipeline._run_symbolic_verifier",
        lambda **kwargs: {
            "confidence_decomposition": {"final_confidence": 0.82},
            "belief_revision": {"final_status": "justified"},
            "constraints": [],
            "violated_constraints": [],
        },
    )

    result = pipeline.answer_question(
        question="What is the Nyaya debate mode aimed at winning an argument at any cost?",
        pramana_types=["inference"],
        use_llm=True,
        use_reasoning_chain=False,
    )

    assert result["answer_source"] == "symbolic_fallback"
    assert "Jalpa" in result["answer"]


def test_pipeline_rejects_debate_mode_llm_answer_with_wrong_mode(monkeypatch):
    pipeline = RAGPipeline()
    pipeline._initialized = True

    class WeakLLM:
        def generate_answer(self, question, chunks):
            return "This refers to Jalpa, the winning debate.", "mock"

    pipeline.llm_engine = WeakLLM()
    pipeline._get_cached_retrieval = lambda question, pramana_types, k: [
        {
            "id": "K1",
            "text": "Vada is truth-seeking. Jalpa is competitive. Vitanda is refutation-only.",
            "source": "mock",
            "supports": ["inference"],
            "tags": ["nyaya", "debate"],
            "fused_score": 0.88,
        }
    ]

    monkeypatch.setattr(
        "pramana_engine.rag_pipeline._run_symbolic_verifier",
        lambda **kwargs: {
            "confidence_decomposition": {"final_confidence": 0.82},
            "belief_revision": {"final_status": "justified"},
            "constraints": [],
            "violated_constraints": [],
        },
    )

    result = pipeline.answer_question(
        question="In a truth-seeking debate, what is the Nyaya debate type?",
        pramana_types=["inference"],
        use_llm=True,
        use_reasoning_chain=False,
    )

    assert result["answer_source"] == "symbolic_fallback"
    assert "Vada" in result["answer"]


def test_hybrid_search_keeps_semantic_only_results(monkeypatch):
    """FAISS semantic results without a 'supports' field must survive the post-fusion filter."""
    clear_hybrid_retrieval_cache()

    class FakeVectorStore:
        def search(self, query, k=10):
            # Semantic result has no 'supports' key — simulates raw FAISS output.
            return [{"id": "SEM1", "text": "Anumana is inference-based pramana.", "distance": 0.1}]

    monkeypatch.setattr(hybrid_mod, "get_vector_store", lambda: FakeVectorStore())
    monkeypatch.setattr(hybrid_mod, "_get_knowledge_index", lambda: [])

    results = hybrid_mod.hybrid_search("anumana inference", ["inference"], k=5)
    ids = [r["id"] for r in results]
    assert "SEM1" in ids, (
        "Semantic-only result without 'supports' field was incorrectly dropped by the post-fusion filter"
    )
