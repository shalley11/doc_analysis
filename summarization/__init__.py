"""Summarization module for document analysis."""

from doc_analysis.summarization.summary_service import (
    generate_pdf_summary,
    generate_batch_summary,
    list_pdfs_in_batch,
    get_cached_summary,
    SummaryType
)
from doc_analysis.summarization.hierarchical_summarizer import (
    summarize_chunks,
    summarize_multiple_pdfs,
    SummarizerConfig,
    # Async/parallel versions for high throughput
    summarize_chunks_async,
    summarize_chunks_parallel,
)
from doc_analysis.summarization.summary_store import (
    get_final_summary,
    is_summary_cached,
    get_summary_progress,
    cleanup_batch_summaries
)

__all__ = [
    # Service functions
    "generate_pdf_summary",
    "generate_batch_summary",
    "list_pdfs_in_batch",
    "get_cached_summary",
    "SummaryType",
    # Hierarchical summarizer (sync)
    "summarize_chunks",
    "summarize_multiple_pdfs",
    "SummarizerConfig",
    # Hierarchical summarizer (async/parallel)
    "summarize_chunks_async",
    "summarize_chunks_parallel",
    # Storage
    "get_final_summary",
    "is_summary_cached",
    "get_summary_progress",
    "cleanup_batch_summaries",
]
