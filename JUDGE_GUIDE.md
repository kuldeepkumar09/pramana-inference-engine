# Pramana Engine — Judge Evaluation Guide

**System running at:** http://localhost:5000  
**Quick demo time:** ~3 minutes  

---

## 1. See the System in 3 Minutes

### Start (if not already running)
```bash
docker compose up
```
Wait until you see `[INFO] Listening at: http://0.0.0.0:5000` in the logs.

### Classic Nyaya inference (smoke → fire)
```bash
curl -s -X POST http://localhost:5000/api/infer \
  -H "Content-Type: application/json" \
  -d '{
    "paksha": "hill",
    "sadhya": "fire",
    "hetu": "smoke",
    "hetuConf": 0.9,
    "vyaptiStr": 0.95,
    "pramanaType": "Anumana"
  }' | python -m json.tool
```
Expected: `"status": "VALID"`, `"accepted": true`, full epistemic trace.

### Ask a philosophy question (RAG + LLM)
```bash
curl -s -X POST http://localhost:5000/api/rag/answer \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the difference between pratyaksha and anumana?", "useLLM": true}' \
  | python -m json.tool
```
Expected: structured answer with `pramana`, `epistemic_status`, `confidence`, `citations`.

### Pramana-constrained vs. unconstrained comparison
```bash
curl -s -X POST http://localhost:5000/api/compare \
  -H "Content-Type: application/json" \
  -d '{
    "paksha": "hill", "sadhya": "fire", "hetu": "smoke",
    "hetuConf": 0.9, "vyaptiStr": 0.95,
    "selectedPramanas": ["Anumana"],
    "constraintMode": "strict"
  }' | python -m json.tool
```

### Interactive UI
- Open http://localhost:5000 (Unified Workspace)
- Open http://localhost:5000/judge (Judge Dashboard — upload a JSON dataset)

---

## 2. R1-R5 Compliance Checklist

| Req | Criterion | Evidence in this system |
|-----|-----------|------------------------|
| **R1** | Proposition Representation | Every inference uses typed `Proposition` objects (atom / implies). Tautological implications (`A → A`) are rejected at parse time (`io.py`). |
| **R2** | Epistemic Justification | All 6 Nyaya pramanas validated: Pratyaksha, Anumana, Shabda, Upamana, Arthapatti, Anupalabdhi. Pramana authority weights applied per inference. |
| **R3** | Status Distinction | Four-valued status: `VALID`, `SUSPENDED`, `UNJUSTIFIED`, `INVALID`. Belief revision applied when contradicting evidence surfaces post-acceptance. |
| **R4** | Invalid Patterns Rejected | Hetvabhasa detection (Savyabhicara, Viruddha, Satpratipaksha, Asiddha, Badhita). Negative reliability (`< 0.0`) rejected. Rate limiting on expensive endpoints. |
| **R5** | Machine-Readable Trace | Every `/api/infer` response includes `epistemic_trace`, `trace.steps[]` with per-step requirements (`R1`–`R5`), and `confidence_decomposition`. |

---

## 3. Pramana-Constrained vs. Baseline Comparison Demo

The `/api/compare` endpoint runs the **same inference twice** — once with pramana constraints, once without — and returns both results side by side.

```bash
# Inference where hetu is weak (low hetuConf)
curl -s -X POST http://localhost:5000/api/compare \
  -H "Content-Type: application/json" \
  -d '{
    "paksha": "sky", "sadhya": "rain", "hetu": "dark_clouds",
    "hetuConf": 0.3, "vyaptiStr": 0.6,
    "selectedPramanas": ["Anumana"],
    "constraintMode": "strict"
  }' | python -m json.tool
```
The constrained result will likely be `SUSPENDED` or `UNJUSTIFIED`; the unconstrained baseline may be `VALID`. The response includes `status_changed: true/false` to highlight the epistemic difference.

---

## 4. Quality Score Interpretation

The judge dashboard (`/judge`) reports a composite quality score:

```
quality_score = (valid + 0.6 × suspended + 0.2 × unjustified) / total
```

| Score | Interpretation |
|-------|---------------|
| 0.85 – 1.00 | Excellent — most inferences are well-grounded |
| 0.65 – 0.84 | Good — moderate suspension, review suspended rows |
| 0.40 – 0.64 | Fair — significant unjustified or invalid results |
| < 0.40 | Poor — dataset or input quality issues |

Suspended is not failure — it means the system correctly identified insufficient evidence rather than accepting a weak inference as valid.

---

## 5. Six Pramanas Implemented

| Pramana | Sanskrit | Description | API value |
|---------|----------|-------------|-----------|
| Pratyaksha | प्रत्यक्ष | Direct perception | `Pratyaksha` |
| Anumana | अनुमान | Inference via hetu + vyapti | `Anumana` |
| Shabda | शब्द | Testimony from reliable source | `Shabda` |
| Upamana | उपमान | Analogical comparison | `Upamana` |
| Arthapatti | अर्थापत्ति | Postulation / best explanation | `Arthapatti` |
| Anupalabdhi | अनुपलब्धि | Non-perception / absence inference | `Anupalabdhi` |

---

## 6. Architecture at a Glance

```
User Request
    │
    ▼
Flask/Gunicorn (4 workers)   ← rate-limited, input-validated
    │
    ├─► Symbolic Engine         Pramana validity, hetvabhasa detection, belief revision
    │       └─► PramanaInferenceEngine.infer()
    │
    ├─► QA Solver               Rule bank look-up (123 rules) → symbolic verifier
    │       └─► solve_question()
    │
    └─► RAG Pipeline            Hybrid retrieval (BM25 + FAISS) → LLM → pramana verification
            ├─► hybrid_search() — weighted RRF (alpha=0.6 semantic)
            ├─► Ollama LLM      (mistral:7b or tinyllama)
            └─► _run_symbolic_verifier() — epistemic status + confidence decomposition
```

---

## 7. Example Dataset for Upload to Judge Dashboard

Save as `test_dataset.json`:
```json
[
  {
    "rules": [{"rule_id": "R1", "name": "Smoke-fire", "pattern": "modus_ponens",
               "required_pramanas": ["inference"], "min_reliability": 0.7}],
    "evidence": [
      {"evidence_id": "E1", "proposition": {"kind": "atom", "value": "smoke_on_hill"},
       "pramana": "inference", "reliability": 0.9, "source": "perception"},
      {"evidence_id": "E2", "proposition": {"kind": "implies", "antecedent": "smoke_on_hill", "consequent": "fire_on_hill"},
       "pramana": "inference", "reliability": 0.95, "source": "vyapti"}
    ],
    "request": {"rule_id": "R1", "premise_evidence_ids": ["E1", "E2"],
                 "target": {"kind": "atom", "value": "fire_on_hill"}}
  }
]
```
Upload this at http://localhost:5000/judge to see the full dashboard in action.
