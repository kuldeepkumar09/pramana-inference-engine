#!/usr/bin/env bash
# Pramana Engine - One-command non-Docker setup
# Usage: bash setup.sh
set -euo pipefail

echo "=== Pramana Engine Setup ==="

# 1. Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 2. Install Python dependencies
pip install --upgrade pip -q
pip install -r requirements.txt -q

# 3. Copy env template if .env doesn't exist
if [ ! -f .env ]; then
    cp .env.example .env
    echo "Created .env from .env.example — edit it to set PRAMANA_ADMIN_TOKEN or OPENAI_API_KEY"
fi

# 4. Start Ollama and pull model (optional — engine works without it in LLM-free mode)
if command -v ollama &>/dev/null; then
    echo "Starting Ollama server..."
    ollama serve &>/dev/null &
    sleep 3
    echo "Pulling mistral:7b (first run: ~4 GB download)..."
    ollama pull mistral:7b
else
    echo "WARNING: ollama not found. Install from https://ollama.com"
    echo "The engine will run in LLM-free mode (symbolic + RAG retrieval only)."
fi

# 5. Start Flask
echo ""
echo "=== Starting Pramana Engine at http://127.0.0.1:5000 ==="
python -m pramana_engine.web
