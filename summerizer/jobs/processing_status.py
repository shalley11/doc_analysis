"""
Processing status model for granular stage tracking.
Provides detailed status updates for UI consumption.
"""
import json
import time
from enum import Enum
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field, asdict
import redis

redis_conn = redis.Redis(host="localhost", port=6379, decode_responses=True)


class ProcessingStage(str, Enum):
    """Processing stages in order of execution."""
    QUEUED = "queued"
    INITIALIZING = "initializing"
    EXTRACTING = "extracting"          # Extracting text/images/tables
    VISION_PROCESSING = "vision"        # Vision model processing
    CHUNKING = "chunking"               # Creating chunks
    EMBEDDING = "embedding"             # Generating embeddings
    INDEXING = "indexing"               # Milvus indexing
    COMPLETED = "completed"
    FAILED = "failed"


class StageStatus(str, Enum):
    """Status of individual stage."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    SKIPPED = "skipped"
    FAILED = "failed"


@dataclass
class StageInfo:
    """Information about a processing stage."""
    name: str
    status: str = StageStatus.PENDING
    progress: int = 0
    current_item: str = ""
    message: str = ""
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    error: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        d = asdict(self)
        # Calculate duration if stage has started
        if self.started_at:
            if self.completed_at:
                d["duration_seconds"] = round(self.completed_at - self.started_at, 2)
            else:
                d["duration_seconds"] = round(time.time() - self.started_at, 2)
        return d


@dataclass
class PDFStatus:
    """Status of individual PDF processing."""
    pdf_name: str
    total_pages: int = 0
    processed_pages: int = 0
    current_stage: str = ProcessingStage.QUEUED
    stages: Dict[str, StageInfo] = field(default_factory=dict)
    error: Optional[str] = None

    def __post_init__(self):
        # Initialize all stages
        if not self.stages:
            for stage in ProcessingStage:
                if stage not in [ProcessingStage.QUEUED, ProcessingStage.COMPLETED, ProcessingStage.FAILED]:
                    self.stages[stage.value] = StageInfo(name=stage.value)

    def to_dict(self) -> Dict:
        return {
            "pdf_name": self.pdf_name,
            "total_pages": self.total_pages,
            "processed_pages": self.processed_pages,
            "progress_percent": int((self.processed_pages / self.total_pages * 100) if self.total_pages > 0 else 0),
            "current_stage": self.current_stage,
            "stages": {k: v.to_dict() for k, v in self.stages.items()},
            "error": self.error
        }


@dataclass
class BatchStatus:
    """Overall batch processing status."""
    batch_id: str
    state: str = ProcessingStage.QUEUED
    total_pdfs: int = 0
    processed_pdfs: int = 0
    total_pages: int = 0
    processed_pages: int = 0
    total_chunks: int = 0
    current_stage: str = ProcessingStage.QUEUED
    current_pdf: str = ""
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    pdfs: Dict[str, PDFStatus] = field(default_factory=dict)
    error: Optional[str] = None
    milvus_indexed: bool = False

    def to_dict(self) -> Dict:
        result = {
            "batch_id": self.batch_id,
            "state": self.state,
            "total_pdfs": self.total_pdfs,
            "processed_pdfs": self.processed_pdfs,
            "total_pages": self.total_pages,
            "processed_pages": self.processed_pages,
            "total_chunks": self.total_chunks,
            "progress_percent": int((self.processed_pages / self.total_pages * 100) if self.total_pages > 0 else 0),
            "current_stage": self.current_stage,
            "current_pdf": self.current_pdf,
            "started_at": self.started_at,
            "milvus_indexed": self.milvus_indexed,
            "error": self.error,
            "pdfs": {k: v.to_dict() for k, v in self.pdfs.items()}
        }

        if self.started_at:
            if self.completed_at:
                result["duration_seconds"] = round(self.completed_at - self.started_at, 2)
                result["completed_at"] = self.completed_at
            else:
                result["duration_seconds"] = round(time.time() - self.started_at, 2)

        return result


def _status_key(batch_id: str) -> str:
    """Redis key for batch status."""
    return f"pdf:status:{batch_id}"


def init_batch_status(batch_id: str, pdf_names: List[str], rq_job_id: str = "") -> BatchStatus:
    """Initialize batch status when job is created."""
    status = BatchStatus(
        batch_id=batch_id,
        state=ProcessingStage.QUEUED,
        total_pdfs=len(pdf_names)
    )

    for pdf_name in pdf_names:
        status.pdfs[pdf_name] = PDFStatus(pdf_name=pdf_name)

    _save_status(batch_id, status)
    return status


def _save_status(batch_id: str, status: BatchStatus):
    """Save status to Redis."""
    data = status.to_dict()
    redis_conn.set(_status_key(batch_id), json.dumps(data))

    # Also publish update for real-time subscribers
    redis_conn.publish(f"status:{batch_id}", json.dumps({
        "type": "status_update",
        "batch_id": batch_id,
        "data": data
    }))


def _load_status(batch_id: str) -> Optional[BatchStatus]:
    """Load status from Redis."""
    data = redis_conn.get(_status_key(batch_id))
    if not data:
        return None

    d = json.loads(data)
    status = BatchStatus(
        batch_id=d["batch_id"],
        state=d["state"],
        total_pdfs=d["total_pdfs"],
        processed_pdfs=d["processed_pdfs"],
        total_pages=d["total_pages"],
        processed_pages=d["processed_pages"],
        total_chunks=d["total_chunks"],
        current_stage=d["current_stage"],
        current_pdf=d["current_pdf"],
        started_at=d.get("started_at"),
        completed_at=d.get("completed_at"),
        error=d.get("error"),
        milvus_indexed=d.get("milvus_indexed", False)
    )

    # Reconstruct PDF statuses
    for pdf_name, pdf_data in d.get("pdfs", {}).items():
        pdf_status = PDFStatus(
            pdf_name=pdf_data["pdf_name"],
            total_pages=pdf_data["total_pages"],
            processed_pages=pdf_data["processed_pages"],
            current_stage=pdf_data["current_stage"],
            error=pdf_data.get("error")
        )
        # Reconstruct stages
        for stage_name, stage_data in pdf_data.get("stages", {}).items():
            pdf_status.stages[stage_name] = StageInfo(
                name=stage_data["name"],
                status=stage_data["status"],
                progress=stage_data["progress"],
                current_item=stage_data["current_item"],
                message=stage_data["message"],
                started_at=stage_data.get("started_at"),
                completed_at=stage_data.get("completed_at"),
                error=stage_data.get("error"),
                details=stage_data.get("details", {})
            )
        status.pdfs[pdf_name] = pdf_status

    return status


def get_batch_status(batch_id: str) -> Optional[Dict]:
    """Get current batch status for API response."""
    status = _load_status(batch_id)
    if not status:
        return None
    return status.to_dict()


def get_batch_status_summary(batch_id: str) -> Optional[Dict]:
    """Get simplified status summary for polling."""
    status = _load_status(batch_id)
    if not status:
        return None

    return {
        "batch_id": status.batch_id,
        "state": status.state,
        "progress_percent": int((status.processed_pages / status.total_pages * 100) if status.total_pages > 0 else 0),
        "current_stage": status.current_stage,
        "current_pdf": status.current_pdf,
        "message": _get_status_message(status),
        "error": status.error
    }


def _get_status_message(status: BatchStatus) -> str:
    """Generate human-readable status message."""
    if status.state == ProcessingStage.QUEUED:
        return "Waiting in queue..."
    elif status.state == ProcessingStage.COMPLETED:
        return f"Completed! Processed {status.total_pages} pages into {status.total_chunks} chunks."
    elif status.state == ProcessingStage.FAILED:
        return f"Failed: {status.error}"
    else:
        stage_messages = {
            ProcessingStage.INITIALIZING: "Initializing...",
            ProcessingStage.EXTRACTING: f"Extracting content from {status.current_pdf}...",
            ProcessingStage.VISION_PROCESSING: f"Processing images/tables with vision model...",
            ProcessingStage.CHUNKING: f"Creating text chunks...",
            ProcessingStage.EMBEDDING: f"Generating embeddings...",
            ProcessingStage.INDEXING: f"Indexing into vector database..."
        }
        return stage_messages.get(status.current_stage, f"Processing {status.current_pdf}...")


# ============================================================================
# Status Update Functions (called from jobs.py)
# ============================================================================

def start_batch(batch_id: str):
    """Mark batch as started."""
    status = _load_status(batch_id)
    if status:
        status.state = ProcessingStage.INITIALIZING
        status.current_stage = ProcessingStage.INITIALIZING
        status.started_at = time.time()
        _save_status(batch_id, status)


def set_batch_totals(batch_id: str, total_pages: int):
    """Set total page count for batch."""
    status = _load_status(batch_id)
    if status:
        status.total_pages = total_pages
        _save_status(batch_id, status)


def start_pdf(batch_id: str, pdf_name: str, total_pages: int):
    """Start processing a PDF."""
    status = _load_status(batch_id)
    if status:
        status.current_pdf = pdf_name
        if pdf_name not in status.pdfs:
            status.pdfs[pdf_name] = PDFStatus(pdf_name=pdf_name)
        status.pdfs[pdf_name].total_pages = total_pages
        status.pdfs[pdf_name].current_stage = ProcessingStage.EXTRACTING
        _save_status(batch_id, status)


def update_stage(
    batch_id: str,
    pdf_name: str,
    stage: ProcessingStage,
    status: StageStatus = StageStatus.RUNNING,
    progress: int = 0,
    current_item: str = "",
    message: str = "",
    details: Dict = None
):
    """Update a specific stage's status."""
    batch_status = _load_status(batch_id)
    if not batch_status:
        return

    batch_status.current_stage = stage

    if pdf_name and pdf_name in batch_status.pdfs:
        pdf_status = batch_status.pdfs[pdf_name]
        pdf_status.current_stage = stage

        if stage in pdf_status.stages:
            stage_info = pdf_status.stages[stage]
            stage_info.status = status
            stage_info.progress = progress
            stage_info.current_item = current_item
            stage_info.message = message
            if details:
                stage_info.details = details

            if status == StageStatus.RUNNING and not stage_info.started_at:
                stage_info.started_at = time.time()
            elif status in [StageStatus.COMPLETED, StageStatus.FAILED, StageStatus.SKIPPED]:
                stage_info.completed_at = time.time()

    _save_status(batch_id, batch_status)


def update_extraction_progress(batch_id: str, pdf_name: str, page_no: int, total_pages: int):
    """Update extraction progress for a PDF."""
    batch_status = _load_status(batch_id)
    if not batch_status:
        return

    if pdf_name in batch_status.pdfs:
        pdf_status = batch_status.pdfs[pdf_name]
        pdf_status.processed_pages = page_no + 1

        # Update extraction stage
        if ProcessingStage.EXTRACTING in pdf_status.stages:
            stage = pdf_status.stages[ProcessingStage.EXTRACTING]
            stage.progress = int((page_no + 1) / total_pages * 100)
            stage.current_item = f"Page {page_no + 1} of {total_pages}"
            stage.message = f"Extracting text and images from page {page_no + 1}"

    # Update batch totals
    batch_status.processed_pages = sum(p.processed_pages for p in batch_status.pdfs.values())
    _save_status(batch_id, batch_status)


def update_chunking_progress(batch_id: str, pdf_name: str, chunk_count: int):
    """Update chunking progress."""
    batch_status = _load_status(batch_id)
    if not batch_status:
        return

    batch_status.total_chunks = chunk_count

    if pdf_name in batch_status.pdfs:
        stage = batch_status.pdfs[pdf_name].stages.get(ProcessingStage.CHUNKING)
        if stage:
            stage.message = f"Created {chunk_count} chunks"
            stage.details = {"chunk_count": chunk_count}

    _save_status(batch_id, batch_status)


def complete_pdf(batch_id: str, pdf_name: str, chunk_count: int):
    """Mark PDF as completed."""
    batch_status = _load_status(batch_id)
    if not batch_status:
        return

    if pdf_name in batch_status.pdfs:
        batch_status.pdfs[pdf_name].current_stage = ProcessingStage.COMPLETED

    batch_status.processed_pdfs += 1
    batch_status.total_chunks += chunk_count
    _save_status(batch_id, batch_status)


def complete_batch(batch_id: str):
    """Mark batch as completed."""
    batch_status = _load_status(batch_id)
    if batch_status:
        batch_status.state = ProcessingStage.COMPLETED
        batch_status.current_stage = ProcessingStage.COMPLETED
        batch_status.completed_at = time.time()
        batch_status.milvus_indexed = True
        _save_status(batch_id, batch_status)


def fail_batch(batch_id: str, error: str):
    """Mark batch as failed."""
    batch_status = _load_status(batch_id)
    if batch_status:
        batch_status.state = ProcessingStage.FAILED
        batch_status.current_stage = ProcessingStage.FAILED
        batch_status.error = error
        batch_status.completed_at = time.time()
        _save_status(batch_id, batch_status)


def fail_pdf(batch_id: str, pdf_name: str, error: str):
    """Mark PDF as failed."""
    batch_status = _load_status(batch_id)
    if batch_status and pdf_name in batch_status.pdfs:
        batch_status.pdfs[pdf_name].current_stage = ProcessingStage.FAILED
        batch_status.pdfs[pdf_name].error = error
        _save_status(batch_id, batch_status)
