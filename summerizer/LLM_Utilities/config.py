import os

# LLM Backend: "vllm" (default for airgapped) or "ollama"
LLM_BACKEND = os.getenv("LLM_BACKEND", "vllm")

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
VLLM_URL = os.getenv("VLLM_URL", "http://localhost:8000")

DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "gemma-3-12b-it")  # vLLM model name

# =========================
# Model Context Lengths
# =========================

MODEL_CONTEXT_LENGTHS = {
    "gemma3:4b": 8192,        # Ollama POC
    "gemma3:12b": 8192,       # Ollama PROD
    "gemma-3-12b-it": 8192,   # vLLM airgapped
}

DEFAULT_CONTEXT_LENGTH = 8192  # Fallback for unknown models

# Context usage warning thresholds (percentage)
CONTEXT_WARNING_THRESHOLD = 80
CONTEXT_ERROR_THRESHOLD = 95
CONTEXT_REJECT_THRESHOLD = 100  # Reject requests exceeding this percentage

# =========================
# Redis Configuration
# =========================

REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
REDIS_DB = int(os.getenv("REDIS_DB", 0))
REFINEMENT_TTL = 7200  # 2 hours TTL for session expiry


def get_model_context_length(model: str) -> int:
    """Get context length for a model."""
    return MODEL_CONTEXT_LENGTHS.get(model, DEFAULT_CONTEXT_LENGTH)


def estimate_tokens(text: str) -> int:
    """Estimate token count (~4 chars per token)."""
    return len(text) // 4
