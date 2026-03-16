FROM python:3.11-slim

WORKDIR /app

# Install system dependencies required for faiss and sentence-transformers.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies first (layer cache optimisation).
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application source.
COPY src/ src/
COPY scripts/ scripts/

# Pre-download models at build time to avoid cold-start delays.
RUN python -c "\
from sentence_transformers import SentenceTransformer, CrossEncoder; \
SentenceTransformer('all-MiniLM-L6-v2'); \
CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')" || true

# Create indexes directory.
RUN mkdir -p indexes

# Expose API port.
EXPOSE 8000

# Run the FastAPI server.
CMD ["uvicorn", "src.serving.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
