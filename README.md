# Pramana-Constrained Inference Engine

A Nyaya-inspired inference engine for the Unriddling Inference 2026 IKS Hackathon. Accepts conclusions only when both logically valid (modus ponens/tollens/syllogism) and epistemically justified via pramana sources.

**Output statuses:** `valid` · `unjustified` · `suspended` · `invalid`

---

## Setup

### Option A — One command (recommended)

```bash
bash setup.sh
```

Installs dependencies, starts Ollama, pulls Mistral 7B, and launches the server.

### Option B — Docker

```bash
cp .env.example .env        # configure tokens/keys (see Environment below)
docker compose up --build   # starts Flask + Ollama, pulls Mistral 7B automatically
```

App available at `http://localhost:5000`

### Option C — Manual

```bash
pip install -r requirements.txt
python -m pramana_engine.web
```

---

## Environment

Copy `.env.example` to `.env` and fill in values:

```bash
cp .env.example .env
```

| Variable | Required | Description |
|----------|----------|-------------|
| `PRAMANA_ADMIN_TOKEN` | No | Protects `/api/rag/cache/clear`. If set, requests must include header `X-Pramana-Admin-Token: <value>`. If empty, endpoint is localhost-only. |
| `OPENAI_API_KEY` | No | OpenAI fallback — used automatically if Ollama is unavailable. |
| `OLLAMA_HOST` | No | Ollama server URL. Default: `http://localhost:11434`. Docker sets this automatically. |

**Setting `PRAMANA_ADMIN_TOKEN` (recommended for demos):**

1. Open `.env` and set: `PRAMANA_ADMIN_TOKEN=pramana2026`
2. Restart the server
3. Cache clear now requires the header:
   ```bash
   curl -X POST http://localhost:5000/api/rag/cache/clear \
     -H "X-Pramana-Admin-Token: pramana2026" \
     -H "Content-Type: application/json" \
     -d '{"scope": "all"}'
   ```

---

## CLI

```bash
python -m pramana_engine.cli list                                        # list scenarios
python -m pramana_engine.cli infer valid                                 # run a scenario
python -m pramana_engine.cli infer-file "path/to/input.json"             # single inference
python -m pramana_engine.cli infer-batch "path/to/input.json" --out out.json  # batch
python -m pramana_engine.cli run-all                                     # run all scenarios
```

Input JSON: native schema (`rules`, `evidence`, `request`) or corpus-list (`proposition`, `evidence`).

---

## Web UI

```bash
python -m pramana_engine.web
```

| URL | Description |
|-----|-------------|
| `http://127.0.0.1:5000/` | Main demo UI |
| `http://127.0.0.1:5000/judge` | Judge dashboard — upload dataset, get quality scores |
| `http://127.0.0.1:5000/compare` | Baseline vs pramana-constrained side-by-side |
| `http://127.0.0.1:5000/dashboard` | RAG operations dashboard |

---

## API Endpoints

| Method | Route | Description |
|--------|-------|-------------|
| `GET` | `/api/scenarios` | List scenario names |
| `POST` | `/api/infer` | Run inference (`paksha`, `sadhya`, `hetu`, `hetuConf`, `vyaptiStr`, `pramanaTypes`, ...) |
| `POST` | `/api/compare` | Baseline vs constrained comparison |
| `POST` | `/api/question-solve` | QA solver: `{"question": "..."}` |
| `POST` | `/api/infer-upload` | Batch inference via multipart file upload |
| `POST` | `/api/judge-report` | Quality score from `{"rows": [...]}` |
| `POST` | `/api/conference-qa` | Conference-domain QA with confidence + citations |
| `POST` | `/api/rag/search` | Retrieval only: `{"question": "...", "k": 5}` |
| `POST` | `/api/rag/answer` | Full RAG pipeline with LLM reasoning |
| `POST` | `/api/rag/explain` | Fallback decision path breakdown |
| `POST` | `/api/rag/batch` | Batch question processing |
| `POST` | `/api/rag/cache/clear` | Clear RAG caches — admin protected (see Environment) |

---

## Tests

```bash
# Run all 120 tests
pytest tests/ -v

# Run with coverage report
pytest --cov=pramana_engine --cov-report=term-missing tests/

# Quick smoke check
python smoke_check.py --full
```

---

## Benchmark

Measures API latency (cold vs warm cache) — no Ollama required:

```bash
python demo_benchmark.py
python demo_benchmark.py --iterations 10
```

Sample output:
```
+-----------------------+-------+---------+---------+---------+---+
| Endpoint              | Cache | Avg ms  | P95 ms  | Min ms  | N |
+-----------------------+-------+---------+---------+---------+---+
| /api/infer            | cold  |    45.2 |    45.2 |    45.2 | 1 |
| /api/infer            | warm  |     8.1 |    10.3 |     7.4 | 5 |
| /api/rag/search       | cold  |  3200.4 |  3200.4 |  3200.4 | 1 |
| /api/rag/search       | warm  |    18.7 |    22.1 |    16.9 | 5 |
+-----------------------+-------+---------+---------+---------+---+
```

---

## LLM Configuration

| Mode | When active | Quality |
|------|-------------|---------|
| Mistral 7B (Ollama) | Ollama running locally | Full reasoning + citations |
| OpenAI gpt-4o-mini | Ollama unavailable + `OPENAI_API_KEY` set | Full reasoning + citations |
| Symbolic fallback | No LLM available | Retrieval + rule-based answers |

The engine selects the mode automatically — no code changes needed.

---

## Example Output

```json
{"status": "valid", "accepted": true, "trace": {"steps": [
  {"stage": "premise_lookup", "ok": true},
  {"stage": "logical_pattern_check", "ok": true, "reason": "Premises satisfy modus ponens."},
  {"stage": "epistemic_justification_check", "status": "valid"}
]}}
```
