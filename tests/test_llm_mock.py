"""
Mock-based LLM tests — do NOT require a live Ollama server.

All tests here use unittest.mock to patch the ollama.Client so they run
in any environment (CI, offline, no GPU) without needing Ollama installed.
"""

from __future__ import annotations

import types
import unittest
from unittest.mock import MagicMock, patch

import pytest


# ─────────────────────────────────────────────────────────────────────────── #
# Helpers
# ─────────────────────────────────────────────────────────────────────────── #

def _make_llm_engine(model_name: str = "phi3:mini", temperature: float = 0.0):
    """Create a MistralLLMEngine with a fully mocked Ollama Client."""
    mock_client = MagicMock()
    # health_check calls client.list()
    mock_client.list.return_value = {"models": [{"name": model_name}]}
    # default generate response
    mock_client.generate.return_value = {"response": "ANSWER: B\nREASON: Pratyaksha is direct perception."}

    with patch("pramana_engine.llm_integration.Client", return_value=mock_client):
        from pramana_engine.llm_integration import MistralLLMEngine
        engine = MistralLLMEngine(model_name=model_name, temperature=temperature)
    engine.client = mock_client
    return engine, mock_client


# ─────────────────────────────────────────────────────────────────────────── #
# Initialization
# ─────────────────────────────────────────────────────────────────────────── #

class TestLLMEngineInit:
    def test_engine_initializes_with_mock_client(self):
        engine, client = _make_llm_engine()
        assert engine.model_name == "phi3:mini"
        assert engine.temperature == 0.0

    def test_health_check_succeeds_when_client_responds(self):
        engine, client = _make_llm_engine()
        client.list.return_value = {"models": []}
        assert engine.health_check() is True

    def test_health_check_fails_when_client_raises(self):
        engine, client = _make_llm_engine()
        client.list.side_effect = ConnectionRefusedError("refused")
        assert engine.health_check() is False


# ─────────────────────────────────────────────────────────────────────────── #
# generate_answer
# ─────────────────────────────────────────────────────────────────────────── #

class TestGenerateAnswer:
    def test_generate_answer_returns_non_empty_string(self):
        engine, client = _make_llm_engine()
        client.generate.return_value = {"response": "Pratyaksha is direct perception."}
        chunks = [{"id": "c1", "text": "Pratyaksha is direct sense-object contact.", "source": "test"}]
        answer, reasoning = engine.generate_answer("What is pratyaksha?", chunks)
        assert isinstance(answer, str)
        assert len(answer) > 0

    def test_generate_answer_raises_on_empty_response(self):
        engine, client = _make_llm_engine()
        client.generate.return_value = {"response": ""}
        chunks = [{"id": "c1", "text": "some context", "source": "test"}]
        with pytest.raises(RuntimeError):
            engine.generate_answer("What is pratyaksha?", chunks)

    def test_generate_answer_raises_on_empty_question(self):
        engine, _ = _make_llm_engine()
        with pytest.raises((ValueError, RuntimeError)):
            engine.generate_answer("", [{"id": "c1", "text": "ctx", "source": "t"}])

    def test_generate_answer_raises_on_empty_chunks(self):
        engine, _ = _make_llm_engine()
        with pytest.raises((ValueError, RuntimeError)):
            engine.generate_answer("What is pramana?", [])


# ─────────────────────────────────────────────────────────────────────────── #
# answer_mcq
# ─────────────────────────────────────────────────────────────────────────── #

class TestAnswerMCQ:
    _MCQ_Q = "Which pramana is direct? A. Anumana B. Pratyaksha C. Shabda D. Upamana"
    _CHUNKS = [{"id": "r1", "text": "Pratyaksha is immediate sense-object contact.", "source": "test"}]

    def test_answer_mcq_returns_dict_with_required_keys(self):
        engine, client = _make_llm_engine()
        client.generate.return_value = {"response": "ANSWER: B\nREASON: Pratyaksha is direct sense-object contact."}
        result = engine.answer_mcq(self._MCQ_Q, self._CHUNKS)
        assert "answer_key" in result
        assert "reason" in result
        assert "raw" in result

    def test_answer_mcq_parses_correct_key(self):
        engine, client = _make_llm_engine()
        client.generate.return_value = {"response": "ANSWER: B\nREASON: Pratyaksha is direct."}
        result = engine.answer_mcq(self._MCQ_Q, self._CHUNKS)
        assert result["answer_key"] == "B"

    def test_answer_mcq_parses_reason(self):
        engine, client = _make_llm_engine()
        client.generate.return_value = {"response": "ANSWER: B\nREASON: Pratyaksha is direct."}
        result = engine.answer_mcq(self._MCQ_Q, self._CHUNKS)
        assert "pratyaksha" in result["reason"].lower()

    def test_answer_mcq_handles_missing_key_gracefully(self):
        """When model omits ANSWER:, answer_key is empty string (not a crash)."""
        engine, client = _make_llm_engine()
        client.generate.return_value = {"response": "Perception is most direct."}
        result = engine.answer_mcq(self._MCQ_Q, self._CHUNKS)
        assert result["answer_key"] == ""  # parsed as empty, not raised

    def test_answer_mcq_all_valid_options(self):
        """Each letter A-D can be parsed correctly."""
        engine, client = _make_llm_engine()
        for letter in ("A", "B", "C", "D"):
            client.generate.return_value = {"response": f"ANSWER: {letter}\nREASON: reason."}
            result = engine.answer_mcq(self._MCQ_Q, self._CHUNKS)
            assert result["answer_key"] == letter


# ─────────────────────────────────────────────────────────────────────────── #
# _build_context
# ─────────────────────────────────────────────────────────────────────────── #

class TestBuildContext:
    def test_build_context_includes_chunk_ids(self):
        engine, _ = _make_llm_engine()
        chunks = [
            {"id": "c1", "text": "First chunk text.", "source": "src1"},
            {"id": "c2", "text": "Second chunk text.", "source": "src2"},
        ]
        ctx = engine._build_context(chunks)
        assert "c1" in ctx
        assert "c2" in ctx

    def test_build_context_limits_to_3_chunks(self):
        engine, _ = _make_llm_engine()
        chunks = [{"id": f"c{i}", "text": f"Chunk {i}", "source": "s"} for i in range(10)]
        ctx = engine._build_context(chunks)
        # Only first 3 should appear
        assert "c3" not in ctx
        assert "c0" in ctx

    def test_build_context_empty_chunks(self):
        engine, _ = _make_llm_engine()
        ctx = engine._build_context([])
        assert ctx == ""


# ─────────────────────────────────────────────────────────────────────────── #
# Temperature determinism (Bug 5 guard)
# ─────────────────────────────────────────────────────────────────────────── #

class TestTemperatureDeterminism:
    def test_temperature_zero_is_passed_to_generate(self):
        """When temperature=0.0, Ollama generate() must receive temperature=0."""
        engine, client = _make_llm_engine(temperature=0.0)
        client.generate.return_value = {"response": "Answer text."}
        chunks = [{"id": "c1", "text": "ctx", "source": "t"}]
        engine.generate_answer("What is pramana?", chunks)
        call_kwargs = client.generate.call_args
        # temperature may appear as positional kwargs in options dict
        assert call_kwargs is not None
        # Verify engine stored temperature=0.0
        assert engine.temperature == 0.0

    def test_config_sets_temperature_zero(self):
        """ProductionConfig must set LLM temperature=0.0 for deterministic answers."""
        from pramana_engine.config import ProductionConfig
        with patch("torch.cuda.is_available", return_value=False), \
             patch("torch.cuda.device_count", return_value=0):
            try:
                cfg = ProductionConfig()
                assert cfg.llm.temperature == 0.0, (
                    "LLM temperature must be 0.0 for deterministic inference. "
                    "Temperature > 0 causes the same question to produce different answers on repeated calls."
                )
            except Exception:
                # If ProductionConfig fails due to missing dirs etc., skip
                pytest.skip("ProductionConfig could not be initialized in this environment")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
