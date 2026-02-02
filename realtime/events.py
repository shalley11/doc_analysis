"""
Event types and schemas for real-time updates.

Two main flows:
1. PDF Processing: upload → extract → chunk → embed → store
2. Summarization: request → batch summarize → reduce → final
"""
from enum import Enum
from typing import Optional, Dict, Any
from dataclasses import dataclass, field, asdict
from datetime import datetime
import json


class EventType(str, Enum):
    """All event types for real-time updates."""

    # PDF Processing Events
    PDF_STARTED = "pdf.started"
    PDF_EXTRACTING = "pdf.extracting"
    PDF_EXTRACTED = "pdf.extracted"
    PDF_CHUNKING = "pdf.chunking"
    PDF_CHUNKED = "pdf.chunked"
    PDF_EMBEDDING = "pdf.embedding"
    PDF_EMBEDDED = "pdf.embedded"
    PDF_COMPLETED = "pdf.completed"
    PDF_FAILED = "pdf.failed"

    # Summarization Events
    SUMMARY_STARTED = "summary.started"
    SUMMARY_CACHE_HIT = "summary.cache_hit"
    SUMMARY_METHOD_SELECTED = "summary.method_selected"
    SUMMARY_BATCH_STARTED = "summary.batch_started"
    SUMMARY_BATCH_COMPLETED = "summary.batch_completed"
    SUMMARY_REDUCE_STARTED = "summary.reduce_started"
    SUMMARY_REDUCE_LEVEL = "summary.reduce_level"
    SUMMARY_LLM_CALL_STARTED = "summary.llm_call_started"
    SUMMARY_LLM_CALL_COMPLETED = "summary.llm_call_completed"
    SUMMARY_COMPLETED = "summary.completed"
    SUMMARY_FAILED = "summary.failed"

    # Multi-PDF Summary Events
    MULTI_PDF_STARTED = "multi_pdf.started"
    MULTI_PDF_PDF_STARTED = "multi_pdf.pdf_started"
    MULTI_PDF_PDF_COMPLETED = "multi_pdf.pdf_completed"
    MULTI_PDF_COMBINING = "multi_pdf.combining"
    MULTI_PDF_COMPLETED = "multi_pdf.completed"


@dataclass
class Event:
    """Base event class for all real-time events."""

    event_type: EventType
    batch_id: str
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    data: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert event to dictionary for JSON serialization."""
        return {
            "event_type": self.event_type.value,
            "batch_id": self.batch_id,
            "timestamp": self.timestamp,
            "data": self.data
        }

    def to_json(self) -> str:
        """Convert event to JSON string."""
        return json.dumps(self.to_dict())

    @classmethod
    def from_json(cls, json_str: str) -> "Event":
        """Create event from JSON string."""
        data = json.loads(json_str)
        return cls(
            event_type=EventType(data["event_type"]),
            batch_id=data["batch_id"],
            timestamp=data["timestamp"],
            data=data.get("data", {})
        )


# ============================================
# PDF Processing Event Factories
# ============================================

def pdf_started(batch_id: str, pdf_count: int, filenames: list) -> Event:
    """Create PDF processing started event."""
    return Event(
        event_type=EventType.PDF_STARTED,
        batch_id=batch_id,
        data={
            "pdf_count": pdf_count,
            "filenames": filenames,
            "message": f"Processing {pdf_count} PDF(s)"
        }
    )


def pdf_extracting(batch_id: str, pdf_name: str, page: int, total_pages: int) -> Event:
    """Create PDF extraction progress event."""
    return Event(
        event_type=EventType.PDF_EXTRACTING,
        batch_id=batch_id,
        data={
            "pdf_name": pdf_name,
            "page": page,
            "total_pages": total_pages,
            "progress_pct": round((page / total_pages) * 100, 1),
            "message": f"Extracting {pdf_name}: page {page}/{total_pages}"
        }
    )


def pdf_extracted(batch_id: str, pdf_name: str, total_pages: int, tables: int, images: int) -> Event:
    """Create PDF extraction completed event."""
    return Event(
        event_type=EventType.PDF_EXTRACTED,
        batch_id=batch_id,
        data={
            "pdf_name": pdf_name,
            "total_pages": total_pages,
            "tables_found": tables,
            "images_found": images,
            "message": f"Extracted {pdf_name}: {total_pages} pages, {tables} tables, {images} images"
        }
    )


def pdf_chunking(batch_id: str, pdf_name: str, chunks_created: int) -> Event:
    """Create chunking progress event."""
    return Event(
        event_type=EventType.PDF_CHUNKING,
        batch_id=batch_id,
        data={
            "pdf_name": pdf_name,
            "chunks_created": chunks_created,
            "message": f"Chunking {pdf_name}: {chunks_created} chunks created"
        }
    )


def pdf_chunked(batch_id: str, pdf_name: str, total_chunks: int) -> Event:
    """Create chunking completed event."""
    return Event(
        event_type=EventType.PDF_CHUNKED,
        batch_id=batch_id,
        data={
            "pdf_name": pdf_name,
            "total_chunks": total_chunks,
            "message": f"Chunked {pdf_name}: {total_chunks} total chunks"
        }
    )


def pdf_embedding(batch_id: str, pdf_name: str, current: int, total: int) -> Event:
    """Create embedding progress event."""
    return Event(
        event_type=EventType.PDF_EMBEDDING,
        batch_id=batch_id,
        data={
            "pdf_name": pdf_name,
            "current": current,
            "total": total,
            "progress_pct": round((current / total) * 100, 1),
            "message": f"Embedding {pdf_name}: {current}/{total}"
        }
    )


def pdf_embedded(batch_id: str, pdf_name: str, total_embedded: int) -> Event:
    """Create embedding completed event."""
    return Event(
        event_type=EventType.PDF_EMBEDDED,
        batch_id=batch_id,
        data={
            "pdf_name": pdf_name,
            "total_embedded": total_embedded,
            "message": f"Embedded {pdf_name}: {total_embedded} chunks"
        }
    )


def pdf_completed(batch_id: str, total_pdfs: int, total_chunks: int, elapsed_seconds: float) -> Event:
    """Create PDF processing completed event."""
    return Event(
        event_type=EventType.PDF_COMPLETED,
        batch_id=batch_id,
        data={
            "total_pdfs": total_pdfs,
            "total_chunks": total_chunks,
            "elapsed_seconds": round(elapsed_seconds, 2),
            "message": f"Completed: {total_pdfs} PDFs, {total_chunks} chunks in {elapsed_seconds:.1f}s"
        }
    )


def pdf_failed(batch_id: str, error: str, pdf_name: Optional[str] = None) -> Event:
    """Create PDF processing failed event."""
    return Event(
        event_type=EventType.PDF_FAILED,
        batch_id=batch_id,
        data={
            "error": error,
            "pdf_name": pdf_name,
            "message": f"Failed: {error}"
        }
    )


# ============================================
# Summarization Event Factories
# ============================================

def summary_started(
    batch_id: str,
    pdf_name: Optional[str],
    summary_type: str,
    total_chunks: int,
    total_words: int
) -> Event:
    """Create summarization started event."""
    target = pdf_name or "all_pdfs"
    return Event(
        event_type=EventType.SUMMARY_STARTED,
        batch_id=batch_id,
        data={
            "pdf_name": pdf_name,
            "target": target,
            "summary_type": summary_type,
            "total_chunks": total_chunks,
            "total_words": total_words,
            "message": f"Starting {summary_type} summary for {target}"
        }
    )


def summary_cache_hit(batch_id: str, pdf_name: Optional[str], summary_type: str) -> Event:
    """Create cache hit event."""
    target = pdf_name or "all_pdfs"
    return Event(
        event_type=EventType.SUMMARY_CACHE_HIT,
        batch_id=batch_id,
        data={
            "pdf_name": pdf_name,
            "summary_type": summary_type,
            "message": f"Cache hit for {target} ({summary_type})"
        }
    )


def summary_method_selected(batch_id: str, method: str, batches: int, reason: str) -> Event:
    """Create method selection event."""
    return Event(
        event_type=EventType.SUMMARY_METHOD_SELECTED,
        batch_id=batch_id,
        data={
            "method": method,
            "batches": batches,
            "reason": reason,
            "message": f"Using {method} method ({batches} batches)"
        }
    )


def summary_batch_started(batch_id: str, batch_index: int, total_batches: int, chunks: int, words: int) -> Event:
    """Create batch summarization started event."""
    return Event(
        event_type=EventType.SUMMARY_BATCH_STARTED,
        batch_id=batch_id,
        data={
            "batch_index": batch_index,
            "total_batches": total_batches,
            "chunks": chunks,
            "words": words,
            "progress_pct": round((batch_index / total_batches) * 100, 1),
            "message": f"Summarizing batch {batch_index + 1}/{total_batches}"
        }
    )


def summary_batch_completed(batch_id: str, batch_index: int, total_batches: int, elapsed_seconds: float) -> Event:
    """Create batch summarization completed event."""
    return Event(
        event_type=EventType.SUMMARY_BATCH_COMPLETED,
        batch_id=batch_id,
        data={
            "batch_index": batch_index,
            "total_batches": total_batches,
            "elapsed_seconds": round(elapsed_seconds, 2),
            "progress_pct": round(((batch_index + 1) / total_batches) * 100, 1),
            "message": f"Batch {batch_index + 1}/{total_batches} completed in {elapsed_seconds:.1f}s"
        }
    )


def summary_reduce_started(batch_id: str, total_summaries: int) -> Event:
    """Create reduce phase started event."""
    return Event(
        event_type=EventType.SUMMARY_REDUCE_STARTED,
        batch_id=batch_id,
        data={
            "total_summaries": total_summaries,
            "message": f"Combining {total_summaries} summaries"
        }
    )


def summary_reduce_level(batch_id: str, level: int, summaries_in: int, summaries_out: int) -> Event:
    """Create reduce level progress event."""
    return Event(
        event_type=EventType.SUMMARY_REDUCE_LEVEL,
        batch_id=batch_id,
        data={
            "level": level,
            "summaries_in": summaries_in,
            "summaries_out": summaries_out,
            "message": f"Reduce level {level}: {summaries_in} → {summaries_out} summaries"
        }
    )


def summary_llm_call_started(batch_id: str, context: str, prompt_words: int) -> Event:
    """Create LLM call started event."""
    return Event(
        event_type=EventType.SUMMARY_LLM_CALL_STARTED,
        batch_id=batch_id,
        data={
            "context": context,
            "prompt_words": prompt_words,
            "message": f"LLM call: {context}"
        }
    )


def summary_llm_call_completed(batch_id: str, context: str, elapsed_seconds: float, response_words: int) -> Event:
    """Create LLM call completed event."""
    return Event(
        event_type=EventType.SUMMARY_LLM_CALL_COMPLETED,
        batch_id=batch_id,
        data={
            "context": context,
            "elapsed_seconds": round(elapsed_seconds, 2),
            "response_words": response_words,
            "message": f"LLM completed: {context} in {elapsed_seconds:.1f}s"
        }
    )


def summary_completed(
    batch_id: str,
    pdf_name: Optional[str],
    summary_type: str,
    method: str,
    elapsed_seconds: float,
    summary_words: int
) -> Event:
    """Create summarization completed event."""
    target = pdf_name or "all_pdfs"
    return Event(
        event_type=EventType.SUMMARY_COMPLETED,
        batch_id=batch_id,
        data={
            "pdf_name": pdf_name,
            "target": target,
            "summary_type": summary_type,
            "method": method,
            "elapsed_seconds": round(elapsed_seconds, 2),
            "summary_words": summary_words,
            "message": f"Summary completed: {target} ({summary_type}) in {elapsed_seconds:.1f}s"
        }
    )


def summary_failed(batch_id: str, error: str, pdf_name: Optional[str] = None) -> Event:
    """Create summarization failed event."""
    return Event(
        event_type=EventType.SUMMARY_FAILED,
        batch_id=batch_id,
        data={
            "error": error,
            "pdf_name": pdf_name,
            "message": f"Summary failed: {error}"
        }
    )


# ============================================
# Multi-PDF Summary Event Factories
# ============================================

def multi_pdf_started(batch_id: str, pdf_names: list, summary_type: str) -> Event:
    """Create multi-PDF summarization started event."""
    return Event(
        event_type=EventType.MULTI_PDF_STARTED,
        batch_id=batch_id,
        data={
            "pdf_names": pdf_names,
            "pdf_count": len(pdf_names),
            "summary_type": summary_type,
            "message": f"Starting multi-PDF summary for {len(pdf_names)} PDFs"
        }
    )


def multi_pdf_pdf_started(batch_id: str, pdf_name: str, pdf_index: int, total_pdfs: int) -> Event:
    """Create individual PDF summarization started event."""
    return Event(
        event_type=EventType.MULTI_PDF_PDF_STARTED,
        batch_id=batch_id,
        data={
            "pdf_name": pdf_name,
            "pdf_index": pdf_index,
            "total_pdfs": total_pdfs,
            "progress_pct": round((pdf_index / total_pdfs) * 100, 1),
            "message": f"Summarizing PDF {pdf_index + 1}/{total_pdfs}: {pdf_name}"
        }
    )


def multi_pdf_pdf_completed(batch_id: str, pdf_name: str, pdf_index: int, total_pdfs: int) -> Event:
    """Create individual PDF summarization completed event."""
    return Event(
        event_type=EventType.MULTI_PDF_PDF_COMPLETED,
        batch_id=batch_id,
        data={
            "pdf_name": pdf_name,
            "pdf_index": pdf_index,
            "total_pdfs": total_pdfs,
            "progress_pct": round(((pdf_index + 1) / total_pdfs) * 100, 1),
            "message": f"PDF {pdf_index + 1}/{total_pdfs} completed: {pdf_name}"
        }
    )


def multi_pdf_combining(batch_id: str, pdf_count: int, summary_type: str) -> Event:
    """Create multi-PDF combining event."""
    return Event(
        event_type=EventType.MULTI_PDF_COMBINING,
        batch_id=batch_id,
        data={
            "pdf_count": pdf_count,
            "summary_type": summary_type,
            "message": f"Combining {pdf_count} PDF summaries into {summary_type} summary"
        }
    )


def multi_pdf_completed(batch_id: str, pdf_count: int, elapsed_seconds: float) -> Event:
    """Create multi-PDF summarization completed event."""
    return Event(
        event_type=EventType.MULTI_PDF_COMPLETED,
        batch_id=batch_id,
        data={
            "pdf_count": pdf_count,
            "elapsed_seconds": round(elapsed_seconds, 2),
            "message": f"Multi-PDF summary completed: {pdf_count} PDFs in {elapsed_seconds:.1f}s"
        }
    )
