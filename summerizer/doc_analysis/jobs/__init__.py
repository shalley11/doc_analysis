"""Job status tracking module."""

from doc_analysis.jobs.job_store import (
    # Job creation
    create_job,
    get_job,
    get_job_summary,

    # Document processing stages
    update_batch_status,
    start_stage,
    complete_stage,
    skip_stage,
    fail_stage,
    update_stats,
    update_pdf_status,

    # Summarization stages
    start_summary_stage,
    complete_summary_stage,
    fail_summary_stage,
    update_summary_stats,
    update_summary_status,

    # Stage definitions
    DOCUMENT_PROCESSING_STAGES,
    SUMMARIZATION_STAGES,
)

__all__ = [
    "create_job",
    "get_job",
    "get_job_summary",
    "update_batch_status",
    "start_stage",
    "complete_stage",
    "skip_stage",
    "fail_stage",
    "update_stats",
    "update_pdf_status",
    "start_summary_stage",
    "complete_summary_stage",
    "fail_summary_stage",
    "update_summary_stats",
    "update_summary_status",
    "DOCUMENT_PROCESSING_STAGES",
    "SUMMARIZATION_STAGES",
]
