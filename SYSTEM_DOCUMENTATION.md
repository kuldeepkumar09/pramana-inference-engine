# Pramana Engine - Complete System Documentation

## Project Overview

**Pramana Engine** is an epistemically honest inference system built on the principles of the ancient Indian Nyaya framework. It combines symbolic reasoning, retrieval-augmented generation (RAG), LLM integration, and rigorous epistemic verification.

### Core Philosophy

Unlike conventional AI systems that generate responses through opaque pattern-matching, Pramana Engine:
- **Tracks sources**: Every answer identifies where knowledge comes from
- **Justifies reasoning**: Shows step-by-step logic, not just conclusions
- **Recognizes uncertainty**: Flags when confidence is low or information is provisional
- **Revises beliefs**: Updates positions when stronger evidence emerges
- **Maintains humility**: Never fabricates; says "I don't know" rather than guessing

---

## Technical Architecture

### Stack Overview

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **Embeddings** | Sentence Transformers (E5-Small) | Semantic understanding of text |
| **Vector DB** | FAISS (CPU-friendly) | Fast similarity search over 1200+ chunks |
| **LLM** | Mistral 7B via Ollama | Reasoning and answer generation |
| **Web Framework** | Flask | REST API endpoints |
| **Verification** | Epistemic Reasoner | Validates beliefs against Nyaya framework |
| **Runtime** | Python 3.13 | Core logic and orchestration |

### Knowledge Base

**Size**: ~1,205 text chunks from philosophical sources
- **Semantic Coverage**: Divided into support categories (perception, inference, comparison, testimony)
- **Tagging System**: Multi-label tags for fine-grained retrieval
  - `pramana` types: pratyaksha, anumana, upamana, shabda
  - Nyaya concepts: hetvabhasa, debate faults, debate modes
  - Epistemic status: justified, unjustified, suspended, invalid
- **Persistence**: FAISS index + metadata cache for fast startup

---

## Nyaya Question Types & Symbolic Fallback System

### 1. **Hetvabhasa Questions** (Logical Fallacies)

**Definition**: Five categories of logical fallacies in Nyaya inference

| Type | Meaning | Detection Cues |
|------|---------|---|
| **Savyabhicara** | Reason proves both presence & absence (inconsistent) | "both presence", "both absence", "inconsistent" |
| **Viruddha** | Reason contradicts the conclusion | "opposites", "negates", "contradictory" |
| **Satpratipaksha** | Evidence supports opposite conclusion | "counter-evidence", "counter-support" |
| **Asiddha** | Reason itself is unestablished | "unproven", "unsubstantiated" |
| **Badhita** | Reason is refuted by another fact | "refuted", "overturned", "contradicted" |

**Fallback Strategy**:
1. **Evidence Labels**: Search retrieved chunks for canonical Hetvabhasa type names
2. **Question Clues**: Infer type from question wording (e.g., "inconsistent" → Savyabhicara)
3. **Catalog vs Inference**: Distinguish "what are the 5 types" (list all) from "which type applies" (specific)
4. **LLM Gating**: LLM answers must contain canonical labels and be supported by evidence

**Example Query**:
```
Q: "The reason proves both presence and absence. Which Hetvabhasa is this?"
Answer: "A likely Hetvabhasa is Savyabhicara based on the inference wording."
Source: symbolic_fallback
```

---

### 2. **Debate Fault Questions** (Nyaya Debate Defects)

**Definition**: Four categorized defects in Nyaya debate methodology

| Type | Meaning | Detection Cues |
|------|---------|---|
| **Chala** | Intentional word-twisting/equivocation | "twists", "reinterprets", "changes meaning" |
| **Jati** | Misclassification; confusing categories | "wrong type", "misclassified", "wrong category" |
| **Nigrahasthana** | Point of defeat; logical contradiction | "contradiction", "point of defeat", "refutation" |
| **Vitanda** | Pure refutation without thesis | "only attacks", "no own position", "mere refutation" |

**Fallback Strategy**:
1. **Type Disambiguation**: Debate-mode keywords (e.g., "truth-seeking") excluded before debate-fault classification
2. **Evidence Extraction**: Rank fault labels by frequency and position in retrieved chunks
3. **Clue Inference**: Detect fault type from question description
4. **Cue-Consistency Gating**: LLM must mention the cue-inferred fault label if question strongly suggests one

**Example Query**:
```
Q: "Which Nyaya debate fault applies when someone intentionally twists the opponent's words?"
Answer: "A likely Nyaya debate fault is Chala based on the available evidence."
Source: symbolic_fallback
```

---

### 3. **Debate Mode Questions** (Nyaya Argumentation Types)

**Definition**: Three modes of Nyaya debate classified by purpose

| Type | Meaning | Detection Cues |
|------|---------|---|
| **Vada** | Honest truth-seeking debate | "truth-seeking", "honest", "legitimate", "seeking truth" |
| **Jalpa** | Competitive debate aimed at winning | "to win", "victory", "winning", "competitive", "win argument" |
| **Vitanda** | Destructive refutation without own thesis | "refutation only", "pure refutation", "only attacks", "no thesis" |

**Fallback Strategy**:
1. **Type-Specific Detection**: Enhanced detection includes both explicit aliases and question-clue markers
2. **Label Ranking**: Score candidate modes by evidence chunk position and frequency
3. **Clue-Based Inference**: Question wording strongly indicates intended mode (e.g., "to win" → Jalpa)
4. **LLM Gating**: LLM must not contradict mode cues; if question emphasizes Vada but LLM says Jalpa, fallback triggers

**Example Query**:
```
Q: "In a truth-seeking debate, what is the Nyaya debate type?"
Answer: "The Nyaya debate type here is Vada, characterized by its approach to argumentation and refutation."
Source: symbolic_fallback
```

---

## RAG Pipeline: Execution Flow

### Step 1: Search Preparation
- Normalize question text (lowercase, whitespace trim)
- Detect question type: Hetvabhasa? Debate fault? Debate mode?
- Prepare retrieval parameters

### Step 2: Hybrid Retrieval
- **Keyword Search**: Token overlap + pramana support bonus
- **Semantic Search**: E5-Small embeddings → FAISS similarity
- **Fusion**: Reciprocal Rank Fusion (RRF) with k=60
- **Result**: Top-10 chunks ranked by combined score

```python
# RRF formula
rrf_score = 1 / (k + rank_keyword) + 1 / (k + rank_semantic)
```

### Step 3: LLM Reasoning (Optional)
- Generate answer using Mistral 7B with retrieved context
- Fallback to Ollama if Mistral unavailable
- Extract reasoning trace for Chain-of-Thought

### Step 4: Answer Extraction
- **Evidence Gating**: LLM answer must overlap ≥2 content tokens with retrieval
- **Symbolic Fallback**: If LLM unsupported or unavailable:
  - Extract specialized labels from chunks (Hetvabhasa/fault/mode names)
  - Infer from question-wording clues
  - Validate LLM consistency with expected labels
- **Source Priority**: LLM (if gated) → symbolic fallback → raw retrieval

### Step 5: Pramana Verification
- Run epistemic evaluator on answer
- Score based on:
  - Evidence support for inference
  - Consistency with retrieved sources
  - Credibility of sources (authority weighting)
  - Conflict detection and defeater analysis
- Output: Confidence (0.0-1.0) + Epistemic Status (justified/unjustified/suspended/invalid)

### Step 6: Response Formatting
- Citations from top-5 chunks with relevance scores
- Verifier constraints and violated constraints
- CoT reasoning (if enabled)

### Step 7: Caching
- Runtime caches: Hybrid retrieval results, query embeddings, answer results
- Persistent: FAISS index, metadata, config

---

## API Reference

### `/api/rag/answer` — Full RAG Pipeline

**Method**: `POST`

**Request Body**:
```json
{
  "question": "In a truth-seeking debate, what is the Nyaya debate type?",
  "useLLM": true,
  "useReasoningChain": false,
  "pramanaTypes": ["inference"]
}
```

**Response**:
```json
{
  "question": "...",
  "answer": "The Nyaya debate type here is Vada, characterized by...",
  "answer_source": "symbolic_fallback",
  "confidence": 0.59,
  "epistemic_status": "unjustified",
  "rag_chunks": [
    {
      "id": "doc-1",
      "source": "Nyaya Commentary",
      "score": 0.85,
      "excerpt": "..."
    }
  ],
  "verifier": {
    "constraints": [...],
    "violated": [...],
    "final_status": "unjustified"
  }
}
```

**Parameters**:
- `question` (string, required): The query
- `useLLM` (boolean, optional, default=true): Use LLM generation (fallback to symbolic if unavailable)
- `useReasoningChain` (boolean, optional, default=false): Enable Chain-of-Thought reasoning
- `pramanaTypes` (array, optional): Filter to specific pramana types (e.g., ["inference", "testimony"])

---

### `/api/rag/search` — Retrieval Only (No Answer)

**Method**: `POST`

**Request Body**:
```json
{
  "question": "What is Hetvabhasa?",
  "pramanaTypes": ["inference"],
  "k": 10
}
```

**Response**:
```json
{
  "question": "What is Hetvabhasa?",
  "results": [
    {
      "id": "...",
      "text": "...",
      "source": "...",
      "fused_score": 0.89
    }
  ],
  "count": 10
}
```

---

### `/api/rag/cache/clear` — Cache Management

**Method**: `POST`

**Request Body**:
```json
{
  "scope": "all"
}
```

**Scopes**:
- `"all"`: Clear retrieval + embedding caches
- `"retrieval"`: Clear hybrid search cache only
- `"embedding"`: Clear query embedding cache only

**Authorization**: Requires admin token if configured

---

## Regression Test Suite

**File**: `tests/test_rag_production.py`
**Total Tests**: 27 (all passing ✅)

### Test Categories

| Category | Count | Coverage |
|----------|-------|----------|
| Infrastructure | 15 | Caching, endpoints, formatting |
| Hetvabhasa | 6 | Evidence extraction, inference vs catalog, clue inference, label validation |
| Debate Fault | 3 | Evidence-based fallback, clue inference, cue-consistency gating |
| Debate Mode | 3 | Evidence-based fallback, clue inference, LLM consistency checking |

### Sample Tests

```python
# Test: Question clues infer Savyabhicara without evidence labels
def test_pipeline_fallacy_question_uses_question_clues_when_chunks_are_generic():
    # Setup: Chunks don't mention "savyabhicara" explicitly
    # Query: "The hetu is inconsistent..."
    # Expected: Fallback extracts "Savyabhicara" from clue markers
    # Result: PASS ✓

# Test: Debate-mode question correctly detected (not fallacy/fault)
def test_pipeline_debate_mode_uses_question_clues_for_jalpa():
    # Query: "What is the Nyaya debate mode aimed at winning an argument?"
    # Clue: "to win" → Jalpa
    # Expected: Returns Jalpa, not generic fallacy catalog
    # Result: PASS ✓

# Test: LLM gating prevents wrong debate mode
def test_pipeline_rejects_debate_mode_llm_answer_with_wrong_mode():
    # Query: "In a truth-seeking debate, what type?"
    # LLM says: "Jalpa" (wrong)
    # Question cue: "truth-seeking" → expects Vada
    # Result: Fallback override, returns Vada
    # Result: PASS ✓
```

---

## Performance & Scaling

### Initialization Time
- **Cold Start**: ~15 seconds (models load first time)
- **Warm Start**: <1 second (cached indices)

### Query Latency
| Operation | Time |
|-----------|------|
| Embedding query | 50-100ms |
| FAISS search | 10-20ms |
| Keyword search | 5-10ms |
| LLM generation | 1-3 seconds (Mistral 7B) |
| Verification | 50-100ms |
| **Total (with LLM)** | ~2-4 seconds |
| **Total (fallback)** | ~200-300ms |

### Memory Usage
- FAISS index: ~50MB (1205 chunks, 384-dim)
- Embeddings model: ~150MB
- LLM in memory: 14GB (Mistral 7B)
- Python process: ~300MB

---

## Configuration & Extensions

### Adding New Nyaya Question Types

To add support for a new Nyaya concept (e.g., Upamana inference types):

1. **Define aliases** in `rag_pipeline.py`:
```python
_UPAMANA_TYPES_ALIASES = {
    "example_type": ("alias1", "alias2", ...),
}
```

2. **Add detection method**:
```python
@classmethod
def _looks_like_upamana_question(cls, question: str) -> bool:
    normalized = cls._normalize_text(question)
    return any(alias in normalized for aliases in _UPAMANA_TYPES_ALIASES.values() for alias in aliases)
```

3. **Add clue inference**:
```python
@staticmethod
def _infer_upamana_type_from_clues(normalized_question: str) -> Optional[str]:
    clues = {
        "type1": ("marker1", "marker2"),
        "type2": ("marker3", "marker4"),
    }
    # ... inference logic
```

4. **Integrate into `_heuristic_answer_from_chunks()`** with same fallback strategy

5. **Add regression tests** to `test_rag_production.py`

---

## Known Limitations & Future Work

### Current Limitations
1. **Epistemic Status**: Symbolic fallback answers marked "unjustified" (appropriate conservatism)
2. **LLM Dependency**: Requires Ollama running locally (for optional LLM enhancement)
3. **Evidence Scarcity**: If question type not in knowledge base, fallback to generic response
4. **Language**: Currently English-only; Sanskrit source texts not processed

### Planned Improvements
- [ ] Confidence calibration: Better tune verifier scores for fallback answers
- [ ] Cold-start handling: Pre-generate embeddings for all chunks on first load
- [ ] Multi-language support: Add Sanskrit/Hindi question processing
- [ ] User feedback loop: Learn from user corrections to refine fallback heuristics
- [ ] Batch import: Tools for adding new philosophical texts to knowledge base
- [ ] Explanation generation: More detailed "why did we choose this answer" traces

---

## Debugging & Troubleshooting

### Common Issues

**Issue**: "Model not found" when using LLM

**Solution**:
```bash
# Pull Mistral model in Ollama
ollama pull mistral:7b
# Or let system auto-fallback to symbolic-only mode
```

**Issue**: Low confidence scores on all answers

**Solution**:
- Check retrieved chunks are relevant: Use `/api/rag/search` to inspect
- Verify vector similarity: Chunks should topically match question
- Review verifier constraints: May be overly strict

**Issue**: Question type not recognized

**Solution**:
1. Check question detection: Add logging to `_looks_like_*_question()` methods
2. Review aliases: Ensure all expected keywords in constant dictionaries
3. Test clue inference: Verify question contains activation markers

### Debug Endpoints

Enable logging in `rag_pipeline.py`:
```python
import logging
logger_pipeline = logging.getLogger("pramana_rag_pipeline")
logger_pipeline.setLevel(logging.DEBUG)  # Verbose output
```

---

## References

### Nyaya Framework Sources
- **Nyaya Sutras**: Ancient logical treatise (~2nd century CE)
- **Vatsyayana Bhashya**: 6th-century authoritative commentary
- **Uddyotakara Varttika**: Further Nyaya elaboration (~7th century)

### Technical References
- **E5 Embeddings**: Wang et al., "Text Embeddings by Weakly Supervised Contrastive Pre-training"
- **FAISS**: Johnson et al., "Billion-Scale Similarity Search with GPUs"
- **Reciprocal Rank Fusion**: Cormack et al., "Reciprocal Rank Fusion Outperforms Condorcet and Individual Rank Learning Methods"

---

## Project Maintenance

**Last Updated**: March 27, 2026
**Version**: 2.0 (Debate-Mode Fallback System)
**Test Coverage**: 27 regression tests (↑ 9 from v1.0)
**Status**: Production-Ready ✅
