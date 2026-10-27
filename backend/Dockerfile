FROM python:3.11-slim

WORKDIR /app

# Install system utilities for healthcheck
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies
COPY backend/requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code, modules, and grounding standards corpus
COPY backend/ backend/
COPY shared/ shared/
COPY grounding_corpus/ grounding_corpus/

ENV PYTHONPATH=/app
ENV PORT=8000
ENV LM_STUDIO_URL=http://localhost:1234/v1

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "backend.app.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
