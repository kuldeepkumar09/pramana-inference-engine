"""
LLM Integration with Mistral 7B via Ollama.
Handles reasoning and answer generation with source citations and Nyaya domain expertise.
Includes comprehensive error handling, logging, and production-grade prompting.
"""

from __future__ import annotations

from typing import List, Dict, Any, Tuple, Optional
import json
import os
import re
import time

try:
    from ollama import Client
except ImportError:
    Client = None

try:
    from openai import OpenAI as OpenAIClient
    _OPENAI_AVAILABLE = True
except ImportError:
    OpenAIClient = None
    _OPENAI_AVAILABLE = False

from .config import get_config
from .logging_setup import logger_llm
from .rag_prompts import get_system_prompt, get_cot_prompt, get_simple_prompt, get_mcq_prompt, get_fallback_mcq_prompt


class MistralLLMEngine:
    """Mistral 7B LLM engine for reasoning with production error handling."""

    def __init__(
        self,
        model_name: Optional[str] = None,
        ollama_host: Optional[str] = None,
        temperature: float = 0.3,
        top_p: float = 0.9,
    ):
        """
        Initialize Mistral LLM engine with production error handling.
        
        Args:
            model_name: Ollama model identifier. Auto-selects for GPU if None.
            ollama_host: Ollama server URL. Uses config default if None.
            temperature: Sampling temperature (0.0-1.0, lower = more deterministic)
            top_p: Nucleus sampling parameter
            
        Raises:
            ImportError: If ollama package not installed
            RuntimeError: If Ollama server is unavailable
        """
        if Client is None:
            raise ImportError("ollama package not installed. Run: pip install ollama")

        try:
            config = get_config()
            self.model_name = model_name or config.llm.model_name
            self.ollama_host = ollama_host or config.llm.ollama_host
            self.temperature = temperature
            self.top_p = top_p
            self.available_models: List[str] = []
            
            logger_llm.info(f"Initializing Mistral LLM engine (model={self.model_name}, host={self.ollama_host})")
            
            self.client = Client(host=self.ollama_host, timeout=config.llm.timeout)

            # Single round-trip: health check AND model enumeration combined.
            # Previously two separate client.list() calls; now one saves 1–30s per init.
            try:
                raw_payload = self.client.list()
            except Exception as exc:
                raise RuntimeError(
                    f"Ollama server not available at {self.ollama_host}: {exc}"
                ) from exc
            self.available_models = self._parse_model_list(raw_payload)
            logger_llm.debug("✓ Ollama server available, %d model(s) found.", len(self.available_models))
            resolved = self._resolve_model_name(self.model_name, self.available_models)
            if resolved != self.model_name:
                logger_llm.warning(
                    "Configured model '%s' not available. Falling back to '%s'.",
                    self.model_name,
                    resolved,
                )
                self.model_name = resolved
            
            logger_llm.info(f"✓ LLM engine initialized. Model: {self.model_name}")
        except Exception as e:
            logger_llm.error(f"✗ Failed to initialize LLM engine: {e}", exc_info=True)
            raise

    def generate_answer(
        self,
        question: str,
        context_chunks: List[Dict[str, Any]],
        instructions: Optional[str] = None,
    ) -> Tuple[str, str]:
        """
        Generate an answer with production error handling and Nyaya prompting.
        
        Args:
            question: Original question
            context_chunks: Retrieved knowledge chunks with citations
            instructions: Optional system instructions. Uses Nyaya prompts if None.
            
        Returns:
            Tuple of (answer_text, reasoning_trace)
            
        Raises:
            ValueError: If question or context_chunks is invalid
            RuntimeError: If LLM generation fails
        """
        try:
            if not question or not isinstance(question, str):
                raise ValueError("Question must be a non-empty string")
            if not context_chunks or not isinstance(context_chunks, list):
                raise ValueError("Context chunks must be a non-empty list")
            
            # Build context string with citations
            context_str = self._build_context(context_chunks)

            # Build prompt
            system_prompt = instructions or get_system_prompt()
            user_prompt = self._build_user_prompt(question, context_str)
            
            logger_llm.debug(f"Generating answer for question: {question[:80]}...")

            # Call Mistral with error handling
            response = self._generate_with_fallback(
                prompt=user_prompt,
                system=system_prompt,
                num_predict=400,
            )

            answer = response.get("response", "").strip()
            if not answer:
                raise RuntimeError("LLM returned empty response")
            
            reasoning = self._extract_reasoning(answer)
            logger_llm.info(f"✓ Answer generated (length={len(answer)}, confidence marks={reasoning.count('%')})")

            return answer, reasoning
        except Exception as e:
            logger_llm.error(f"✗ Answer generation failed: {e}", exc_info=True)
            raise RuntimeError(f"Answer generation failed: {e}")

    def generate_with_reasoning_chain(
        self,
        question: str,
        context_chunks: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Generate answer using chain-of-thought prompting with Nyaya framework.
        
        Args:
            question: Original question
            context_chunks: Retrieved knowledge chunks
            
        Returns:
            Dict with answer, reasoning steps, and citations
            
        Raises:
            ValueError: If inputs are invalid
            RuntimeError: If generation fails
        """
        try:
            if not question or not isinstance(question, str):
                raise ValueError("Question must be a non-empty string")
            if not context_chunks or not isinstance(context_chunks, list):
                raise ValueError("Context chunks must be a non-empty list")
            
            logger_llm.debug(f"Generating CoT answer for: {question[:80]}...")
            
            context_str = self._build_context(context_chunks)
            cot_prompt = get_cot_prompt(question, context_str)
            system_prompt = get_system_prompt()

            response = self._generate_with_fallback(
                prompt=cot_prompt,
                system=system_prompt,
                num_predict=512,
            )

            full_response = response.get("response", "").strip()
            if not full_response:
                raise RuntimeError("LLM returned empty response for CoT")
            
            parsed = self._parse_cot_response(full_response)
            confidence = self._extract_confidence(parsed.get("Confidence", "0.5"))
            
            logger_llm.info(f"✓ CoT reasoning complete (confidence={confidence:.2f})")

            return {
                "question": question,
                "full_response": full_response,
                "understanding": parsed.get("Understanding", ""),
                "relevant_evidence": parsed.get("Relevant Evidence", ""),
                "reasoning": parsed.get("Reasoning", ""),
                "answer": parsed.get("Answer", ""),
                "confidence": confidence,
                "citations": [c["id"] for c in context_chunks[:5]],
            }
        except Exception as e:
            logger_llm.error(f"✗ CoT reasoning failed: {e}", exc_info=True)
            raise RuntimeError(f"CoT reasoning failed: {e}")

    def answer_mcq(
        self,
        question: str,
        context_chunks: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Answer an MCQ question with structured ANSWER/REASON output.

        Uses a compact MCQ prompt that forces phi3:mini-scale models to output
        a single letter and a one-sentence Nyaya justification, avoiding the
        list-output problem that breaks sentence extraction.

        Returns:
            Dict with keys: answer_key (str), reason (str), raw (str)
        """
        context_str = self._build_context(context_chunks)
        prompt = get_mcq_prompt(question, context_str)
        system = get_system_prompt()

        try:
            response = self._generate_with_fallback(
                prompt=prompt,
                system=system,
                num_predict=200,
            )
            raw = response.get("response", "").strip()
        except Exception:
            # Retry with minimal fallback prompt for very small models
            try:
                fb_prompt = get_fallback_mcq_prompt(question, context_str)
                response = self._generate_with_fallback(prompt=fb_prompt, system="", num_predict=100)
                raw = response.get("response", "").strip()
            except Exception as e:
                raise RuntimeError(f"MCQ generation failed: {e}")

        # Parse ANSWER: X / REASON: ...
        answer_key = ""
        reason = ""
        for line in raw.splitlines():
            line = line.strip()
            if line.upper().startswith("ANSWER:"):
                key_part = line.split(":", 1)[1].strip().upper()
                # Take first letter A-D
                for ch in key_part:
                    if ch in "ABCD":
                        answer_key = ch
                        break
            elif line.upper().startswith("REASON:"):
                reason = line.split(":", 1)[1].strip()
        logger_llm.info(f"MCQ answer: key={answer_key!r}, reason_len={len(reason)}")
        return {"answer_key": answer_key, "reason": reason, "raw": raw}

    def _build_context(self, chunks: List[Dict[str, Any]]) -> str:
        """Format chunks into context string with citations."""
        context_lines = []
        for i, chunk in enumerate(chunks[:3], 1):
            citation_id = chunk.get("id", f"ref_{i}")
            text = chunk.get("text", "")
            source = chunk.get("source", "unknown")
            context_lines.append(
                f"[{citation_id}] (from {source}):\n{text}\n"
            )
        return "\n".join(context_lines)

    def _build_user_prompt(self, question: str, context: str) -> str:
        """Build user prompt using Nyaya simple reasoning template."""
        return get_simple_prompt(question, context)

    def _default_system_prompt(self) -> str:
        """Default system prompt for Mistral."""
        return """You are an expert assistant in Nyaya philosophy and epistemic reasoning. 
Your task is to answer questions about pramanas (means of knowledge), inference, and philosophical logic.
Provide clear, well-reasoned answers grounded in the provided context.
Always cite your sources when referencing the context.
Be precise and concise."""

    def _extract_reasoning(self, answer: str) -> str:
        """Extract reasoning steps from answer."""
        lines = answer.split("\n")
        reasoning_lines = []
        for line in lines:
            if any(marker in line.lower() for marker in ["because", "therefore", "thereby", "thus", "based on", "since"]):
                reasoning_lines.append(line.strip())
        return " ".join(reasoning_lines) if reasoning_lines else "Direct inference from context."

    def _parse_cot_response(self, response: str) -> Dict[str, str]:
        """Parse chain-of-thought response into structured sections."""
        sections = {
            "Understanding": "",
            "Relevant Evidence": "",
            "Reasoning": "",
            "Answer": "",
            "Confidence": "",
        }

        current_section = None
        current_text = []

        for line in response.split("\n"):
            # Check if line matches a section header
            for section in sections.keys():
                if f"**{section}**" in line or f"_{section}_" in line or line.startswith(f"{section}:"):
                    if current_section:
                        sections[current_section] = " ".join(current_text).strip()
                    current_section = section
                    current_text = []
                    break
            else:
                if current_section and line.strip():
                    current_text.append(line.strip())

        if current_section:
            sections[current_section] = " ".join(current_text).strip()

        return sections

    def _extract_confidence(self, confidence_text: str) -> float:
        """
        Extract numerical confidence from text with error handling.
        
        Args:
            confidence_text: Text containing confidence value
            
        Returns:
            Confidence float between 0.0 and 1.0
        """
        try:
            if not confidence_text:
                return 0.5
            
            # Try to find a number between 0 and 1
            matches = re.findall(r"0\.\d+|[01]", confidence_text)
            if matches:
                conf = float(matches[0])
                return max(0.0, min(1.0, conf))  # Clamp to [0, 1]
        except Exception as e:
            logger_llm.debug(f"Confidence extraction failed: {e}")
        
        return 0.5

    def health_check(self) -> bool:
        """
        Check if Ollama server is running with error logging.
        
        Returns:
            True if server is available, False otherwise
        """
        try:
            self.client.list()
            logger_llm.debug("✓ Ollama server health check passed")
            return True
        except Exception as e:
            logger_llm.error(f"✗ Ollama server unavailable: {e}")
            return False

    def _parse_model_list(self, payload: Any) -> List[str]:
        """Extract model name strings from a client.list() payload (no network call)."""
        models: List[Any] = []
        if isinstance(payload, dict):
            models = payload.get("models", []) or []
        elif payload is not None and hasattr(payload, "models"):
            models = list(getattr(payload, "models") or [])
        names: List[str] = []
        for model in models:
            name = ""
            if isinstance(model, dict):
                name = str(model.get("name") or model.get("model") or "").strip()
            else:
                if hasattr(model, "name") and getattr(model, "name"):
                    name = str(getattr(model, "name")).strip()
                elif hasattr(model, "model") and getattr(model, "model"):
                    name = str(getattr(model, "model")).strip()
            if name:
                names.append(name)
        return names

    def _fetch_available_models(self) -> List[str]:
        """Fetch installed model names from Ollama server (makes one client.list() call)."""
        try:
            return self._parse_model_list(self.client.list())
        except Exception as e:
            logger_llm.warning(f"Could not fetch Ollama models list: {e}")
            return []

    def _resolve_model_name(self, requested: str, available: List[str]) -> str:
        """Resolve a usable model name from installed Ollama models."""
        if not available:
            return requested

        requested_norm = requested.lower().strip()
        if requested in available:
            return requested

        # Prefer exact family/tag variations first.
        for candidate in available:
            c = candidate.lower().strip()
            if c == requested_norm:
                return candidate
            if c.startswith(requested_norm + ":") or requested_norm.startswith(c + ":"):
                return candidate

        # Prioritize logical default families for this project.
        preferred_prefixes = (
            "mistral",
            "llama3",
            "llama",
            "qwen",
            "phi",
            "gemma",
        )
        for prefix in preferred_prefixes:
            for candidate in available:
                if candidate.lower().startswith(prefix):
                    return candidate

        # Last resort: first installed model.
        return available[0]

    def _generate_with_fallback(self, prompt: str, system: str, num_predict: int) -> Dict[str, Any]:
        """Generate once; if model missing, refresh models and retry with fallback model.
        If runner crashes (OOM), retry with reduced tokens and context window.
        Network and timeout failures are retried up to 3 times with exponential backoff.

        NOTE — temperature=0 and first-call variation (Bug 5):
          Ollama's temperature=0 sets greedy decoding (always picks highest-probability
          token). However, the VERY FIRST call after Ollama starts or after the model
          is loaded may produce slightly different output due to non-deterministic CUDA
          kernel scheduling during model warm-up. Subsequent calls with the same prompt
          are stable. Workaround: if strict determinism is required, send one throwaway
          "warm-up" prompt before the real query, or use a rule-bank fallback (always
          deterministic) for the most critical answers.
        """
        _RETRY_DELAYS = (1.0, 3.0, 10.0)

        def _is_transient(exc: Exception) -> bool:
            """True for network/timeout errors that are worth retrying."""
            msg = str(exc).lower()
            transient_keywords = ("connection", "timeout", "timed out", "network", "reset by peer", "broken pipe")
            return any(kw in msg for kw in transient_keywords)

        def _call() -> Dict[str, Any]:
            return self.client.generate(
                model=self.model_name,
                prompt=prompt,
                system=system,
                stream=False,
                options={
                    "temperature": self.temperature,
                    "top_p": self.top_p,
                    "num_predict": num_predict,
                    "num_ctx": 2048,
                },
            )

        last_exc: Exception | None = None
        for attempt, delay in enumerate(_RETRY_DELAYS, 1):
            try:
                return _call()
            except Exception as e:
                message = str(e).lower()

                # Handle OOM / runner crash — retry on CPU to bypass VRAM limit
                if "process has terminated" in message or ("status code: 500" in message and "not found" not in message):
                    logger_llm.warning(
                        "LLM runner crashed (GPU OOM). Retrying with CPU mode (num_gpu=0, num_predict=128)."
                    )
                    return self.client.generate(
                        model=self.model_name,
                        prompt=prompt,
                        system=system,
                        stream=False,
                        options={
                            "temperature": self.temperature,
                            "top_p": self.top_p,
                            "num_predict": 128,
                            "num_ctx": 1024,
                            "num_gpu": 0,
                        },
                    )

                # Handle model-not-found — swap to fallback model
                if "not found" in message and "model" in message:
                    available = self._fetch_available_models()
                    fallback = self._resolve_model_name(self.model_name, available)
                    if fallback == self.model_name:
                        raise
                    logger_llm.warning(
                        "Model '%s' unavailable at generation time. Retrying with '%s'.",
                        self.model_name,
                        fallback,
                    )
                    self.model_name = fallback
                    return self.client.generate(
                        model=self.model_name,
                        prompt=prompt,
                        system=system,
                        stream=False,
                        options={
                            "temperature": self.temperature,
                            "top_p": self.top_p,
                            "num_predict": num_predict,
                            "num_ctx": 2048,
                        },
                    )

                # Transient network/timeout error — exponential backoff retry
                if _is_transient(e) and attempt < len(_RETRY_DELAYS):
                    logger_llm.warning(
                        "Transient LLM error (attempt %d/%d): %s — retrying in %.0fs",
                        attempt, len(_RETRY_DELAYS), e, delay,
                    )
                    last_exc = e
                    time.sleep(delay)
                    continue

                raise

        # All retries exhausted
        raise RuntimeError(f"LLM generation failed after {len(_RETRY_DELAYS)} attempts") from last_exc


class OpenAILLMEngine:
    """
    OpenAI API fallback — activated only when Ollama is unavailable
    AND OPENAI_API_KEY is set. Identical interface to MistralLLMEngine.
    """

    def __init__(self):
        if not _OPENAI_AVAILABLE:
            raise ImportError("openai package not installed. Run: pip install openai")
        from .config import OpenAIConfig
        cfg = OpenAIConfig()
        if not cfg.api_key:
            raise RuntimeError("OPENAI_API_KEY environment variable is not set")
        self.model_name: str = cfg.model_name
        self.temperature: float = cfg.temperature
        self.max_tokens: int = cfg.max_tokens
        self._client = OpenAIClient(api_key=cfg.api_key, timeout=cfg.timeout)
        logger_llm.info(f"Initialized OpenAILLMEngine (model={self.model_name})")

    def health_check(self) -> bool:
        try:
            self._client.models.list()
            return True
        except Exception as e:
            logger_llm.error(f"OpenAI health check failed: {e}")
            return False

    def generate_answer(
        self,
        question: str,
        context_chunks: List[Dict[str, Any]],
        instructions: Optional[str] = None,
    ) -> Tuple[str, str]:
        context_str = self._build_context(context_chunks)
        system_prompt = instructions or get_system_prompt()
        user_prompt = get_simple_prompt(question, context_str)
        raw = self._chat(system_prompt, user_prompt)
        reasoning = self._extract_reasoning(raw)
        return raw, reasoning

    def generate_with_reasoning_chain(
        self,
        question: str,
        context_chunks: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        context_str = self._build_context(context_chunks)
        cot_prompt = get_cot_prompt(question, context_str)
        full_response = self._chat(get_system_prompt(), cot_prompt)
        return {
            "question": question,
            "full_response": full_response,
            "understanding": "",
            "relevant_evidence": "",
            "reasoning": full_response,
            "answer": full_response,
            "confidence": 0.75,
            "citations": [c["id"] for c in context_chunks[:5]],
        }

    def answer_mcq(
        self,
        question: str,
        context_chunks: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        context_str = self._build_context(context_chunks)
        raw = self._chat(get_system_prompt(), get_mcq_prompt(question, context_str))
        answer_key = ""
        reason = ""
        for line in (raw or "").splitlines():
            line = line.strip()
            if line.upper().startswith("ANSWER:"):
                for ch in line.split(":", 1)[1].strip().upper():
                    if ch in "ABCD":
                        answer_key = ch
                        break
            elif line.upper().startswith("REASON:"):
                reason = line.split(":", 1)[1].strip()
        return {"answer_key": answer_key, "reason": reason, "raw": raw}

    def _chat(self, system: str, user: str) -> str:
        response = self._client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        return (response.choices[0].message.content or "").strip()

    def _build_context(self, chunks: List[Dict[str, Any]]) -> str:
        lines = []
        for i, chunk in enumerate(chunks[:3], 1):
            lines.append(
                f"[{chunk.get('id', f'ref_{i}')}] (from {chunk.get('source', 'unknown')}):\n{chunk.get('text', '')}\n"
            )
        return "\n".join(lines)

    def _extract_reasoning(self, answer: str) -> str:
        markers = ["because", "therefore", "thereby", "thus", "based on", "since"]
        lines = [ln.strip() for ln in answer.split("\n")
                 if any(m in ln.lower() for m in markers)]
        return " ".join(lines) if lines else "Direct inference from context."


def get_llm_engine() -> MistralLLMEngine | OpenAILLMEngine | None:
    """
    Priority: 1) Ollama  2) OpenAI (if OPENAI_API_KEY set)  3) None (LLM-free)
    """
    try:
        engine = MistralLLMEngine()
        if engine.health_check():
            return engine
    except Exception:
        pass

    if _OPENAI_AVAILABLE and os.environ.get("OPENAI_API_KEY", "").strip():
        try:
            engine = OpenAILLMEngine()
            if engine.health_check():
                logger_llm.info("Ollama unavailable — using OpenAI fallback.")
                return engine
        except Exception:
            pass

    return None
