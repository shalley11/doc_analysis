"""
Job status store with module-wise organization.

Modules:
- document_processing: PDF upload, extraction, chunking, vision, embedding, indexing
- summarization: Summary generation for PDFs
"""
import json
import time
from datetime import datetime
from typing import List, Optional, Dict, Any
from redis import Redis
from doc_analysis import config

redis_client = Redis(host=config.REDIS_HOST, port=config.REDIS_PORT, decode_responses=True)


# ============================================================================
# Timestamp Formatting
# ============================================================================

TIMESTAMP_FORMAT = "%Y-%m-%d %H:%M:%S"


def _format_timestamp(ts: Optional[float]) -> Optional[str]:
    """Convert Unix timestamp to readable format."""
    if ts is None:
        return None
    try:
        return datetime.fromtimestamp(ts).strftime(TIMESTAMP_FORMAT)
    except (ValueError, OSError):
        return None


def _format_timestamps_in_dict(data: Any) -> Any:
    """Recursively format all timestamp fields in a dictionary."""
    if isinstance(data, dict):
        result = {}
        for key, value in data.items():
            if key in ("created_at", "updated_at", "started_at", "completed_at") and isinstance(value, (int, float)):
                result[key] = _format_timestamp(value)
            else:
                result[key] = _format_timestamps_in_dict(value)
        return result
    elif isinstance(data, list):
        return [_format_timestamps_in_dict(item) for item in data]
    else:
        return data


# ============================================================================
# Module Definitions
# ============================================================================

DOCUMENT_PROCESSING_STAGES = [
    "queued",
    "extraction",      # PDF text/table/image extraction
    "chunking",        # Building chunks
    "vision_tables",   # Vision model for tables
    "vision_images",   # Vision model for images
    "embedding",       # Generating embeddings
    "indexing",        # Inserting into Milvus
    "completed"
]

SUMMARIZATION_STAGES = [
    "pending",
    "fetching_chunks",   # Retrieving chunks from Milvus
    "batch_summarizing", # Summarizing batches (hierarchical)
    "combining",         # Combining batch summaries
    "storing",           # Storing final summary
    "completed"
]


def _job_key(batch_id: str) -> str:
    return f"job:{batch_id}"


def _get_stage_index(stages: List[str], stage: str) -> int:
    """Get index of stage for progress calculation."""
    try:
        return stages.index(stage)
    except ValueError:
        return 0


def _calculate_progress(stages: List[str], stage: str) -> int:
    """Calculate progress percentage based on current stage."""
    idx = _get_stage_index(stages, stage)
    total = len(stages) - 1  # Exclude 'completed'
    if stage == "completed":
        return 100
    return int((idx / total) * 100)


def _create_stage_dict(stages: List[str]) -> Dict:
    """Create initial stage dictionary."""
    return {
        stage: {
            "status": "pending",
            "started_at": None,
            "completed_at": None,
            "details": None
        }
        for stage in stages
    }


# ============================================================================
# Job Creation
# ============================================================================

def create_job(batch_id: str, pdf_names: List[str]):
    """Create a new job with module-wise structure."""
    job = {
        "batch_id": batch_id,
        "status": "queued",
        "created_at": time.time(),
        "updated_at": time.time(),

        # Document Processing Module
        "document_processing": {
            "status": "queued",
            "progress": 0,
            "current_stage": "queued",
            "stages": _create_stage_dict(DOCUMENT_PROCESSING_STAGES),
            "stats": {
                "total_pdfs": len(pdf_names),
                "total_pages": 0,
                "total_elements": 0,
                "total_chunks": 0,
                "tables_found": 0,
                "tables_processed": 0,
                "images_found": 0,
                "images_processed": 0,
                "embeddings_generated": 0
            },
            "pdfs": {
                f"pdf_{i+1}": {
                    "name": name,
                    "status": "queued",
                    "pages": 0,
                    "elements": 0,
                    "chunks": 0,
                    "tables": 0,
                    "images": 0,
                    "error": None
                }
                for i, name in enumerate(pdf_names)
            }
        },

        # Summarization Module
        "summarization": {
            "status": "pending",
            "progress": 0,
            "current_stage": "pending",
            "stages": _create_stage_dict(SUMMARIZATION_STAGES),
            "stats": {
                "total_chunks_processed": 0,
                "batch_count": 0,
                "batches_completed": 0,
                "summaries_generated": 0,
                "cached_summaries_used": 0
            },
            "summaries": {}  # Will store per-PDF and combined summary status
        }
    }

    # Mark queued as started for document processing
    job["document_processing"]["stages"]["queued"]["status"] = "completed"
    job["document_processing"]["stages"]["queued"]["started_at"] = time.time()
    job["document_processing"]["stages"]["queued"]["completed_at"] = time.time()

    redis_client.set(_job_key(batch_id), json.dumps(job))


# ============================================================================
# Document Processing Module Updates
# ============================================================================

def update_batch_status(batch_id: str, status: str, error: Optional[str] = None):
    """Update overall batch status."""
    job = get_job_raw(batch_id)
    job["status"] = status
    job["document_processing"]["status"] = status
    if error:
        job["error"] = error
    job["updated_at"] = time.time()
    redis_client.set(_job_key(batch_id), json.dumps(job))


def start_stage(batch_id: str, stage: str, details: Optional[str] = None):
    """Mark a document processing stage as started."""
    job = get_job_raw(batch_id)
    module = job["document_processing"]

    module["current_stage"] = stage
    module["status"] = "processing"
    module["progress"] = _calculate_progress(DOCUMENT_PROCESSING_STAGES, stage)
    module["stages"][stage]["status"] = "in_progress"
    module["stages"][stage]["started_at"] = time.time()
    if details:
        module["stages"][stage]["details"] = details

    job["status"] = "processing"
    job["updated_at"] = time.time()
    redis_client.set(_job_key(batch_id), json.dumps(job))


def complete_stage(batch_id: str, stage: str, details: Optional[str] = None):
    """Mark a document processing stage as completed."""
    job = get_job_raw(batch_id)
    module = job["document_processing"]

    module["stages"][stage]["status"] = "completed"
    module["stages"][stage]["completed_at"] = time.time()
    if details:
        module["stages"][stage]["details"] = details
    module["progress"] = _calculate_progress(DOCUMENT_PROCESSING_STAGES, stage)

    if stage == "completed":
        module["status"] = "completed"
        job["status"] = "completed"

    job["updated_at"] = time.time()
    redis_client.set(_job_key(batch_id), json.dumps(job))


def skip_stage(batch_id: str, stage: str, reason: Optional[str] = None):
    """Mark a document processing stage as skipped."""
    job = get_job_raw(batch_id)
    module = job["document_processing"]

    module["stages"][stage]["status"] = "skipped"
    module["stages"][stage]["details"] = reason or "Skipped"

    job["updated_at"] = time.time()
    redis_client.set(_job_key(batch_id), json.dumps(job))


def fail_stage(batch_id: str, stage: str, error: str):
    """Mark a document processing stage as failed."""
    job = get_job_raw(batch_id)
    module = job["document_processing"]

    module["stages"][stage]["status"] = "failed"
    module["stages"][stage]["completed_at"] = time.time()
    module["stages"][stage]["details"] = error
    module["status"] = "failed"

    job["status"] = "failed"
    job["error"] = error
    job["updated_at"] = time.time()
    redis_client.set(_job_key(batch_id), json.dumps(job))


def update_stats(batch_id: str, **kwargs):
    """Update document processing statistics."""
    job = get_job_raw(batch_id)
    stats = job["document_processing"]["stats"]

    for key, value in kwargs.items():
        if key in stats:
            stats[key] = value

    job["updated_at"] = time.time()
    redis_client.set(_job_key(batch_id), json.dumps(job))


def update_pdf_status(batch_id: str, pdf_id: str, status: str, error=None, **kwargs):
    """Update individual PDF status."""
    job = get_job_raw(batch_id)
    pdf_info = job["document_processing"]["pdfs"].get(pdf_id, {})

    pdf_info["status"] = status
    pdf_info["error"] = error
    for key, value in kwargs.items():
        pdf_info[key] = value

    job["document_processing"]["pdfs"][pdf_id] = pdf_info
    job["updated_at"] = time.time()
    redis_client.set(_job_key(batch_id), json.dumps(job))


# ============================================================================
# Summarization Module Updates
# ============================================================================

def start_summary_stage(batch_id: str, stage: str, details: Optional[str] = None):
    """Mark a summarization stage as started."""
    job = get_job_raw(batch_id)
    module = job["summarization"]

    module["current_stage"] = stage
    module["status"] = "processing"
    module["progress"] = _calculate_progress(SUMMARIZATION_STAGES, stage)
    module["stages"][stage]["status"] = "in_progress"
    module["stages"][stage]["started_at"] = time.time()
    if details:
        module["stages"][stage]["details"] = details

    job["updated_at"] = time.time()
    redis_client.set(_job_key(batch_id), json.dumps(job))


def complete_summary_stage(batch_id: str, stage: str, details: Optional[str] = None):
    """Mark a summarization stage as completed."""
    job = get_job_raw(batch_id)
    module = job["summarization"]

    module["stages"][stage]["status"] = "completed"
    module["stages"][stage]["completed_at"] = time.time()
    if details:
        module["stages"][stage]["details"] = details
    module["progress"] = _calculate_progress(SUMMARIZATION_STAGES, stage)

    if stage == "completed":
        module["status"] = "completed"

    job["updated_at"] = time.time()
    redis_client.set(_job_key(batch_id), json.dumps(job))


def fail_summary_stage(batch_id: str, stage: str, error: str):
    """Mark a summarization stage as failed."""
    job = get_job_raw(batch_id)
    module = job["summarization"]

    module["stages"][stage]["status"] = "failed"
    module["stages"][stage]["completed_at"] = time.time()
    module["stages"][stage]["details"] = error
    module["status"] = "failed"

    job["updated_at"] = time.time()
    redis_client.set(_job_key(batch_id), json.dumps(job))


def update_summary_stats(batch_id: str, **kwargs):
    """Update summarization statistics."""
    job = get_job_raw(batch_id)
    stats = job["summarization"]["stats"]

    for key, value in kwargs.items():
        if key in stats:
            stats[key] = value

    job["updated_at"] = time.time()
    redis_client.set(_job_key(batch_id), json.dumps(job))


def update_summary_status(
    batch_id: str,
    pdf_name: Optional[str],
    summary_type: str,
    status: str,
    cached: bool = False
):
    """Update status for a specific summary."""
    job = get_job_raw(batch_id)
    summaries = job["summarization"]["summaries"]

    key = pdf_name or "combined"
    if key not in summaries:
        summaries[key] = {}

    summaries[key][summary_type] = {
        "status": status,
        "cached": cached,
        "updated_at": time.time()
    }

    job["updated_at"] = time.time()
    redis_client.set(_job_key(batch_id), json.dumps(job))


# ============================================================================
# Job Retrieval
# ============================================================================

def get_job_raw(batch_id: str) -> Dict:
    """Get job status with raw Unix timestamps (for internal use)."""
    data = redis_client.get(_job_key(batch_id))
    if not data:
        raise KeyError(f"Job {batch_id} not found")
    return json.loads(data)


def get_job(batch_id: str) -> Dict:
    """Get job status with formatted timestamps."""
    job = get_job_raw(batch_id)
    return _format_timestamps_in_dict(job)


def get_job_summary(batch_id: str) -> Dict:
    """Get a condensed summary of job status with formatted timestamps."""
    job = get_job_raw(batch_id)

    doc_module = job["document_processing"]
    sum_module = job["summarization"]

    result = {
        "batch_id": batch_id,
        "status": job["status"],
        "created_at": _format_timestamp(job["created_at"]),
        "updated_at": _format_timestamp(job["updated_at"]),

        "document_processing": {
            "status": doc_module["status"],
            "progress": doc_module["progress"],
            "current_stage": doc_module["current_stage"],
            "stats": doc_module["stats"]
        },

        "summarization": {
            "status": sum_module["status"],
            "progress": sum_module["progress"],
            "current_stage": sum_module["current_stage"],
            "stats": sum_module["stats"],
            "summaries": _format_timestamps_in_dict(sum_module["summaries"])
        }
    }
    return result
