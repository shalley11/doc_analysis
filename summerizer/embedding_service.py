# embedding_service.py
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from embedding_utils import E5LargeEmbedder

# -----------------------------
# Request & Response Schemas
# -----------------------------
class EmbeddingRequest(BaseModel):
    texts: List[str]

class EmbeddingResponse(BaseModel):
    embeddings: List[List[float]]

# -----------------------------
# Initialize FastAPI & Embedder
# -----------------------------
app = FastAPI(title="Embedding Service")

# The embedder will automatically choose device (GPU if available)
# and use quantized model on GPU to save memory
embedder = E5LargeEmbedder(
    cache_dir="./offline_models/e5-large",  # Path to save / load offline
    quantize=True                           # Use INT8 quantization if GPU is available
)

# -----------------------------
# POST /embed endpoint
# -----------------------------
@app.post("/embed", response_model=EmbeddingResponse)
def embed_text(request: EmbeddingRequest):
    """
    Accepts a list of texts and returns their embeddings.
    """
    embeddings = embedder.embed_texts(request.texts)
    return EmbeddingResponse(embeddings=embeddings)
