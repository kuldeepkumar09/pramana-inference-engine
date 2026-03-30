FROM python:3.13-slim

WORKDIR /app

# libgomp1 is required by faiss-cpu (OpenMP); curl is used by the healthcheck
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies (cached layer — only rebuilds when requirements change)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download the embedding model at build time (~130 MB) to avoid cold-start delay
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('intfloat/e5-small-v2')"

# Copy application source
COPY pramana_engine/ ./pramana_engine/

# Create FAISS cache directory and run as non-root
RUN useradd -m -u 1000 pramana && \
    mkdir -p /app/vector_store_cache && \
    chown -R pramana /app
USER pramana

ENV PYTHONUNBUFFERED=1
EXPOSE 5000

CMD ["python", "-m", "pramana_engine.web"]
