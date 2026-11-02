# BLK NOISE - FastAPI + Streamlit in one container for a Hugging Face Docker Space.
# Streamlit is the public app on port 7860; FastAPI listens on 127.0.0.1:8000.

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/home/user/.cache/huggingface \
    DB_PATH=/home/user/app/data/listings.db \
    API_URL=http://127.0.0.1:8000 \
    PORT=7860

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

# Spaces runs containers with UID 1000, so create that user first.
RUN useradd -m -u 1000 user
USER user
WORKDIR /home/user/app

COPY --chown=user requirements.txt ./
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Cache the embedding model in the image so cold starts never download it.
RUN python -c "from sentence_transformers import SentenceTransformer; \
SentenceTransformer('all-MiniLM-L6-v2')"

COPY --chown=user . .

# Bake the SQLite database (with embeddings) at build time. Set
# --build-arg BUILD_DATA=0 to skip and build it on first start instead.
ARG BUILD_DATA=1
ARG CSV_PATH=ResaleflatpricesbasedonregistrationdatefromJan2017onwards.csv
RUN if [ "$BUILD_DATA" = "1" ]; then \
        python scripts/load_data.py "$CSV_PATH" --db "$DB_PATH" \
        && python scripts/build_embeddings.py --db "$DB_PATH"; \
    fi

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD curl -fsS "http://127.0.0.1:${PORT}/_stcore/health" || exit 1

CMD ["bash", "scripts/start.sh"]
