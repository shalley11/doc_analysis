"""
Summary storage module for hierarchical summarization.

Supports two modes:
- hybrid: Redis for intermediate summaries, Milvus for final (searchable)
- redis_only: All summaries in Redis (temporary)
"""
import json
import redis
from typing import Optional, Dict, List
from datetime import datetime

from doc_analysis.config import (
    SUMMARY_STORAGE_MODE,
    SUMMARY_REDIS_TTL,
    BATCH_TTL_SECONDS
)
from doc_analysis.logging_config import get_api_logger

logger = get_api_logger()

# Redis connection
_redis_client = None


def _get_redis() -> redis.Redis:
    """Get Redis client (lazy initialization)."""
    global _redis_client
    if _redis_client is None:
        _redis_client = redis.Redis(host="localhost", port=6379, decode_responses=True)
    return _redis_client


# ============================================================================
# Redis Keys
# ============================================================================

def _batch_summary_key(batch_id: str) -> str:
    """Key for batch summary progress."""
    return f"summary:{batch_id}:progress"


def _batch_summary_data_key(batch_id: str, batch_index: int) -> str:
    """Key for individual batch summary."""
    return f"summary:{batch_id}:batch:{batch_index}"


def _final_summary_key(batch_id: str, pdf_name: Optional[str], summary_type: str) -> str:
    """Key for final summary."""
    pdf_part = pdf_name if pdf_name else "all"
    return f"summary:{batch_id}:{pdf_part}:{summary_type}"


# ============================================================================
# Progress Tracking (Redis)
# ============================================================================

def init_summary_progress(
    batch_id: str,
    total_batches: int,
    pdf_name: Optional[str] = None
) -> None:
    """Initialize summary progress tracking."""
    r = _get_redis()
    key = _batch_summary_key(batch_id)

    progress = {
        "batch_id": batch_id,
        "pdf_name": pdf_name or "all",
        "total_batches": total_batches,
        "completed_batches": 0,
        "status": "processing",
        "started_at": datetime.utcnow().isoformat(),
        "batch_status": {str(i): "pending" for i in range(total_batches)}
    }

    r.setex(key, SUMMARY_REDIS_TTL, json.dumps(progress))
    logger.info(f"Initialized summary progress: {batch_id}, {total_batches} batches")


def update_batch_progress(
    batch_id: str,
    batch_index: int,
    status: str = "completed"
) -> None:
    """Update progress for a specific batch."""
    r = _get_redis()
    key = _batch_summary_key(batch_id)

    data = r.get(key)
    if not data:
        return

    progress = json.loads(data)
    progress["batch_status"][str(batch_index)] = status

    if status == "completed":
        progress["completed_batches"] = sum(
            1 for s in progress["batch_status"].values() if s == "completed"
        )

    r.setex(key, SUMMARY_REDIS_TTL, json.dumps(progress))


def get_summary_progress(batch_id: str) -> Optional[Dict]:
    """Get current summary progress."""
    r = _get_redis()
    key = _batch_summary_key(batch_id)

    data = r.get(key)
    if data:
        return json.loads(data)
    return None


def mark_summary_complete(batch_id: str) -> None:
    """Mark summary generation as complete."""
    r = _get_redis()
    key = _batch_summary_key(batch_id)

    data = r.get(key)
    if not data:
        return

    progress = json.loads(data)
    progress["status"] = "completed"
    progress["completed_at"] = datetime.utcnow().isoformat()

    r.setex(key, SUMMARY_REDIS_TTL, json.dumps(progress))


# ============================================================================
# Batch Summary Storage (Redis)
# ============================================================================

def store_batch_summary(
    batch_id: str,
    batch_index: int,
    summary: str,
    metadata: Optional[Dict] = None
) -> None:
    """Store intermediate batch summary in Redis."""
    r = _get_redis()
    key = _batch_summary_data_key(batch_id, batch_index)

    data = {
        "batch_index": batch_index,
        "summary": summary,
        "created_at": datetime.utcnow().isoformat(),
        "metadata": metadata or {}
    }

    r.setex(key, SUMMARY_REDIS_TTL, json.dumps(data))
    update_batch_progress(batch_id, batch_index, "completed")

    logger.debug(f"Stored batch summary: {batch_id}, batch {batch_index}")


def get_batch_summary(batch_id: str, batch_index: int) -> Optional[str]:
    """Get a specific batch summary from Redis."""
    r = _get_redis()
    key = _batch_summary_data_key(batch_id, batch_index)

    data = r.get(key)
    if data:
        return json.loads(data).get("summary")
    return None


def get_all_batch_summaries(batch_id: str) -> List[str]:
    """Get all batch summaries for a batch_id."""
    r = _get_redis()
    progress = get_summary_progress(batch_id)

    if not progress:
        return []

    total_batches = progress.get("total_batches", 0)
    summaries = []

    for i in range(total_batches):
        summary = get_batch_summary(batch_id, i)
        if summary:
            summaries.append(summary)

    return summaries


# ============================================================================
# Final Summary Storage (Redis or Milvus based on mode)
# ============================================================================

def store_final_summary(
    batch_id: str,
    summary: str,
    summary_type: str,
    pdf_name: Optional[str] = None,
    metadata: Optional[Dict] = None
) -> None:
    """
    Store final summary based on storage mode.

    Args:
        batch_id: Batch identifier
        summary: Generated summary text
        summary_type: brief, bulletwise, or detailed
        pdf_name: PDF name (None for combined summary)
        metadata: Additional metadata (chunks, pages, etc.)
    """
    if SUMMARY_STORAGE_MODE == "hybrid":
        _store_final_summary_hybrid(batch_id, summary, summary_type, pdf_name, metadata)
    else:
        _store_final_summary_redis(batch_id, summary, summary_type, pdf_name, metadata)

    mark_summary_complete(batch_id)


def _store_final_summary_redis(
    batch_id: str,
    summary: str,
    summary_type: str,
    pdf_name: Optional[str],
    metadata: Optional[Dict]
) -> None:
    """Store final summary in Redis only."""
    r = _get_redis()
    key = _final_summary_key(batch_id, pdf_name, summary_type)

    data = {
        "batch_id": batch_id,
        "pdf_name": pdf_name,
        "summary_type": summary_type,
        "summary": summary,
        "created_at": datetime.utcnow().isoformat(),
        "metadata": metadata or {},
        "storage": "redis"
    }

    # Use batch TTL for final summaries
    r.setex(key, BATCH_TTL_SECONDS, json.dumps(data))
    logger.info(f"Stored final summary in Redis: {key}")


def _store_final_summary_hybrid(
    batch_id: str,
    summary: str,
    summary_type: str,
    pdf_name: Optional[str],
    metadata: Optional[Dict]
) -> None:
    """Store final summary in both Redis (cache) and Milvus (persistent)."""
    # Store in Redis for quick access
    _store_final_summary_redis(batch_id, summary, summary_type, pdf_name, metadata)

    # Store in Milvus for persistence and search
    try:
        from doc_analysis.vector_store.milvus_store import MilvusStore
        from doc_analysis.embedding.e5_embedder import embed_passages

        collection_name = f"batch_{batch_id.replace('-', '_')}"
        store = MilvusStore(collection_name)

        # Generate embedding for the summary
        embeddings = embed_passages([summary])

        if embeddings:
            summary_chunk = {
                "chunk_id": f"summary_{pdf_name or 'all'}_{summary_type}",
                "text": summary,
                "pdf_name": pdf_name or "combined",
                "pdf_id": "summary",
                "word_count": len(summary.split()),
                "embedding": embeddings[0],
            }

            store.insert_summary(summary_chunk)
            logger.info(f"Stored final summary in Milvus: {collection_name}")

    except Exception as e:
        logger.warning(f"Failed to store summary in Milvus (Redis still has it): {e}")


def get_final_summary(
    batch_id: str,
    summary_type: str,
    pdf_name: Optional[str] = None
) -> Optional[Dict]:
    """
    Retrieve final summary (checks Redis first, then Milvus if hybrid).

    Returns:
        Dictionary with summary and metadata, or None if not found
    """
    r = _get_redis()
    key = _final_summary_key(batch_id, pdf_name, summary_type)

    # Check Redis first
    data = r.get(key)
    if data:
        return json.loads(data)

    # If hybrid mode, check Milvus
    if SUMMARY_STORAGE_MODE == "hybrid":
        try:
            from doc_analysis.vector_store.milvus_store import MilvusStore

            collection_name = f"batch_{batch_id.replace('-', '_')}"
            store = MilvusStore(collection_name)

            summary_data = store.get_summary(pdf_name, summary_type)
            if summary_data:
                return {
                    "batch_id": batch_id,
                    "pdf_name": pdf_name,
                    "summary_type": summary_type,
                    "summary": summary_data.get("text", ""),
                    "metadata": summary_data.get("metadata", {}),
                    "storage": "milvus"
                }
        except Exception as e:
            logger.warning(f"Failed to retrieve summary from Milvus: {e}")

    return None


def is_summary_cached(
    batch_id: str,
    summary_type: str,
    pdf_name: Optional[str] = None
) -> bool:
    """Check if a summary is already cached."""
    return get_final_summary(batch_id, summary_type, pdf_name) is not None


# ============================================================================
# Cleanup
# ============================================================================

def cleanup_batch_summaries(batch_id: str) -> None:
    """Clean up all summary-related data for a batch."""
    r = _get_redis()

    # Delete progress
    r.delete(_batch_summary_key(batch_id))

    # Delete batch summaries
    pattern = f"summary:{batch_id}:batch:*"
    for key in r.scan_iter(pattern):
        r.delete(key)

    # Delete final summaries
    pattern = f"summary:{batch_id}:*"
    for key in r.scan_iter(pattern):
        r.delete(key)

    logger.info(f"Cleaned up summaries for batch: {batch_id}")
