import os
from dataclasses import dataclass

# =========================
# Redis Configuration
# =========================
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))

# =========================
# Milvus Configuration
# =========================
MILVUS_HOST = os.getenv("MILVUS_HOST", "localhost")
MILVUS_PORT = os.getenv("MILVUS_PORT", "19530")

# =========================
# E5 Embedding Model Configuration
# =========================
E5_MODEL_PATH = os.getenv("E5_MODEL_PATH", "/home/labuser/Documents/doc_analysis/models/e5-large-v2")
E5_MODEL_NAME = "intfloat/e5-large-v2"  # HuggingFace fallback
E5_MAX_WORDS = 500

OLLAMA_URL = "http://localhost:11434/api/generate"
VISION_MODEL_POC = "gemma3:4b"
VISION_MODEL_PROD = "gemma3-12b"

MIN_TEXT_CHARS = 300

MAX_PDFS_PER_BATCH = 5
BATCH_TTL_SECONDS = 24 * 3600  # 1 day
EMBEDDING_DIM = 1024  # E5-large

IMAGE_STORE_DIR = "/tmp/doc_images"

# Cleanup service settings
CLEANUP_MAX_FILE_AGE_HOURS = 24      # Delete files older than this (in hours)
CLEANUP_INTERVAL_MINUTES = 60        # Run cleanup every X minutes
CLEANUP_INCLUDE_TEMP_PDFS = True     # Also clean /tmp batch PDF files

# Vision model settings
ENABLE_TABLE_VISION = True   # Set False to skip vision processing for tables
ENABLE_IMAGE_VISION = True   # Set False to skip vision processing for images
VISION_BATCH_SIZE = 3        # Number of parallel vision requests
VISION_TIMEOUT = 900         # 15 minutes timeout per image

# Summarization settings
SUMMARY_MODEL = "gemma3:4b"  # Model for text summarization
SUMMARY_TIMEOUT = 300  # 5 minutes timeout for summary generation

# Hierarchical summarization settings
SUMMARY_MAX_WORDS_PER_BATCH = 3000  # Max words per LLM call
SUMMARY_MAX_CHUNKS_PER_BATCH = 20   # Max chunks per batch
SUMMARY_INTERMEDIATE_WORDS = 300    # Target words for intermediate summaries
SUMMARY_FINAL_WORDS = 500           # Target words for final summary
SUMMARY_REDIS_TTL = 3600            # 1 hour TTL for intermediate summaries in Redis

# Summary storage mode: "hybrid" (Redis + Milvus) or "redis_only"
# - hybrid: Store intermediate in Redis, final summary in Milvus (searchable)
# - redis_only: Store all summaries in Redis only (temporary, not searchable)
SUMMARY_STORAGE_MODE = "hybrid"

# Testing: Page range filter (set to None to process all pages)
TEST_START_PAGE = 2  # Start page (1-indexed, inclusive)
TEST_END_PAGE = 4    # End page (1-indexed, inclusive)
# Set both to None to disable page filtering:
# TEST_START_PAGE = None
# TEST_END_PAGE = None


@dataclass
class ChunkingConfig:
    """Simple configuration for semantic chunking."""

    # Word count thresholds
    min_words: int = 50
    max_words: int = 500

    # Overlap between consecutive chunks (in words)
    overlap_words: int = 50

    # Similarity threshold for semantic grouping
    similarity_threshold: float = 0.6

    # Include bounding box in metadata
    include_bbox: bool = True


# Default chunking configuration
DEFAULT_CHUNKING_CONFIG = ChunkingConfig()
