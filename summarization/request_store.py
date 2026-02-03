"""
Request store for tracking summary requests in Redis.

Manages request_id generation and metadata storage for:
- Initial summary generation
- Summary refinement
- Summary regeneration

Each request stores:
- request_id: Unique identifier (UUID)
- batch_id: Batch identifier
- pdf_name: PDF filename
- summary_type: Type of summary (brief, bulletwise, detailed, executive)
- summary: Generated/refined summary text
- user_feedback: User feedback (for refine/regenerate)
- method: Generation method (direct, hierarchical, refine, regenerate)
- created_at: Timestamp
- parent_request_id: Previous request ID (for refinement chain)
"""
import json
import uuid
from datetime import datetime
from typing import Optional, Dict, Any, List
import redis

from doc_analysis.config import REDIS_HOST, REDIS_PORT, REQUEST_TTL_SECONDS
from doc_analysis.logging_config import get_summarization_logger

logger = get_summarization_logger()

# Redis key prefix for summary requests
REQUEST_KEY_PREFIX = "summary_request:"

# Redis client
_redis_client = None


def _get_redis() -> redis.Redis:
    """Get or create Redis client."""
    global _redis_client
    if _redis_client is None:
        _redis_client = redis.Redis(
            host=REDIS_HOST,
            port=REDIS_PORT,
            decode_responses=True
        )
    return _redis_client


def generate_request_id() -> str:
    """Generate a unique request ID."""
    return str(uuid.uuid4())


def _request_key(request_id: str) -> str:
    """Get Redis key for a request."""
    return f"{REQUEST_KEY_PREFIX}{request_id}"


def store_summary_request(
    request_id: str,
    batch_id: str,
    pdf_name: str,
    summary_type: str,
    summary: str,
    method: str,
    user_feedback: Optional[str] = None,
    parent_request_id: Optional[str] = None,
    chunks_used: int = 0,
    total_chunks: int = 0,
    total_pages: int = 0,
    additional_metadata: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Store a summary request in Redis.

    Args:
        request_id: Unique request identifier
        batch_id: Batch identifier
        pdf_name: PDF filename
        summary_type: Type of summary
        summary: Generated summary text
        method: Generation method (direct, hierarchical, refine, regenerate)
        user_feedback: User feedback if any
        parent_request_id: Previous request ID for refinement chain
        chunks_used: Number of chunks used
        total_chunks: Total chunks in document
        total_pages: Total pages in document
        additional_metadata: Any additional metadata to store

    Returns:
        Dictionary with stored request data
    """
    r = _get_redis()

    request_data = {
        "request_id": request_id,
        "batch_id": batch_id,
        "pdf_name": pdf_name,
        "summary_type": summary_type,
        "summary": summary,
        "method": method,
        "user_feedback": user_feedback,
        "parent_request_id": parent_request_id,
        "chunks_used": chunks_used,
        "total_chunks": total_chunks,
        "total_pages": total_pages,
        "created_at": datetime.utcnow().isoformat(),
    }

    # Add any additional metadata
    if additional_metadata:
        request_data["metadata"] = additional_metadata

    # Store in Redis with TTL
    key = _request_key(request_id)
    r.setex(key, REQUEST_TTL_SECONDS, json.dumps(request_data))

    logger.info(f"[REQUEST_STORE] Stored request {request_id} | batch={batch_id} | pdf={pdf_name} | method={method}")

    return request_data


def get_summary_request(request_id: str) -> Optional[Dict[str, Any]]:
    """
    Retrieve a summary request from Redis.

    Args:
        request_id: Request identifier

    Returns:
        Request data dictionary or None if not found
    """
    r = _get_redis()
    key = _request_key(request_id)

    data = r.get(key)
    if data:
        logger.debug(f"[REQUEST_STORE] Retrieved request {request_id}")
        return json.loads(data)

    logger.warning(f"[REQUEST_STORE] Request not found: {request_id}")
    return None


def delete_summary_request(request_id: str) -> bool:
    """
    Delete a summary request from Redis.

    Args:
        request_id: Request identifier

    Returns:
        True if deleted, False if not found
    """
    r = _get_redis()
    key = _request_key(request_id)

    result = r.delete(key)
    if result:
        logger.info(f"[REQUEST_STORE] Deleted request {request_id}")
        return True

    logger.warning(f"[REQUEST_STORE] Request not found for deletion: {request_id}")
    return False


def get_request_history(
    batch_id: str,
    pdf_name: Optional[str] = None,
    limit: int = 10
) -> List[Dict[str, Any]]:
    """
    Get request history for a batch/PDF.

    Note: This scans Redis keys, so use sparingly.

    Args:
        batch_id: Batch identifier
        pdf_name: Optional PDF name filter
        limit: Maximum number of requests to return

    Returns:
        List of request data dictionaries, sorted by created_at descending
    """
    r = _get_redis()

    # Scan for all request keys
    requests = []
    cursor = 0

    while True:
        cursor, keys = r.scan(cursor, match=f"{REQUEST_KEY_PREFIX}*", count=100)

        for key in keys:
            data = r.get(key)
            if data:
                request_data = json.loads(data)

                # Filter by batch_id
                if request_data.get("batch_id") == batch_id:
                    # Filter by pdf_name if specified
                    if pdf_name is None or request_data.get("pdf_name") == pdf_name:
                        requests.append(request_data)

        if cursor == 0:
            break

    # Sort by created_at descending
    requests.sort(key=lambda x: x.get("created_at", ""), reverse=True)

    return requests[:limit]


def extend_request_ttl(request_id: str, ttl_seconds: Optional[int] = None) -> bool:
    """
    Extend the TTL of a request.

    Args:
        request_id: Request identifier
        ttl_seconds: New TTL in seconds (defaults to REQUEST_TTL_SECONDS)

    Returns:
        True if extended, False if not found
    """
    r = _get_redis()
    key = _request_key(request_id)

    if ttl_seconds is None:
        ttl_seconds = REQUEST_TTL_SECONDS

    if r.exists(key):
        r.expire(key, ttl_seconds)
        logger.debug(f"[REQUEST_STORE] Extended TTL for request {request_id} to {ttl_seconds}s")
        return True

    return False
