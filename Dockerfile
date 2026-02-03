# GPU-enabled image with CUDA 12.4 support
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY summerizer/ ./summerizer/
COPY models/ ./models/

# Set environment variables
ENV PYTHONPATH=/app/summerizer
ENV REDIS_HOST=redis
ENV MILVUS_HOST=milvus-standalone
ENV MILVUS_PORT=19530
ENV E5_MODEL_PATH=/app/models/e5-large-v2
ENV LLM_BACKEND=vllm
ENV VLLM_URL=http://vllm:8000

# Expose API port
EXPOSE 8000

# Default command - run API
CMD ["uvicorn", "doc_analysis.api:app", "--host", "0.0.0.0", "--port", "8000"]
