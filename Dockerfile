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

# Copy application code to doc_analysis package
COPY api.py config.py logging_config.py openapi.yaml ./doc_analysis/
COPY chunking/ ./doc_analysis/chunking/
COPY cleanup/ ./doc_analysis/cleanup/
COPY embedding/ ./doc_analysis/embedding/
COPY jobs/ ./doc_analysis/jobs/
COPY pdf/ ./doc_analysis/pdf/
COPY qa/ ./doc_analysis/qa/
COPY queues/ ./doc_analysis/queues/
COPY realtime/ ./doc_analysis/realtime/
COPY summarization/ ./doc_analysis/summarization/
COPY vector_store/ ./doc_analysis/vector_store/
COPY vision/ ./doc_analysis/vision/
COPY workers/ ./doc_analysis/workers/

# Copy models
COPY models/ ./models/

# Create __init__.py for doc_analysis package
RUN touch ./doc_analysis/__init__.py

# Set environment variables
ENV PYTHONPATH=/app
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
