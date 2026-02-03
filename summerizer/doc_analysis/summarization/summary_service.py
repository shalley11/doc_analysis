"""
Summary generation service for PDF documents.

Supports two modes:
- Direct: For small documents that fit in LLM context
- Hierarchical: For large documents using map-reduce approach

Storage modes (configurable via SUMMARY_STORAGE_MODE):
- hybrid: Redis for intermediate, Milvus for final (searchable)
- redis_only: All in Redis (temporary)

Refinement modes:
- Simple: Refine using only the original summary + feedback
- Contextual: Refine using original summary + feedback + relevant source chunks
"""
from typing import Optional, List, Dict
from enum import Enum

from doc_analysis.config import SUMMARY_STORAGE_MODE
from doc_analysis.vector_store.milvus_store import MilvusStore
from doc_analysis.logging_config import get_summarization_logger, BatchContext
from doc_analysis.summarization.hierarchical_summarizer import (
    summarize_chunks,
    summarize_multiple_pdfs,
    SummarizerConfig,
    _call_llm,
    DEFAULT_CONFIG
)
from doc_analysis.summarization.summary_store import (
    get_final_summary,
    is_summary_cached
)
from doc_analysis.summarization.summary_prompts import (
    get_simple_refinement_prompt,
    get_contextual_refinement_prompt
)
from doc_analysis.embedding.e5_embedder import embed_passages

logger = get_summarization_logger()


class SummaryType(str, Enum):
    BRIEF = "brief"
    BULLETWISE = "bulletwise"
    DETAILED = "detailed"
    EXECUTIVE = "executive"


def _safe_collection_name(batch_id: str) -> str:
    """Convert batch_id to safe collection name."""
    return f"batch_{batch_id.replace('-', '_')}"


def generate_pdf_summary(
    batch_id: str,
    pdf_name: str,
    summary_type: SummaryType,
    use_cache: bool = True
) -> Dict:
    """
    Generate summary for a specific PDF in the batch.

    Uses hierarchical summarization for large documents.

    Args:
        batch_id: Batch identifier
        pdf_name: Name of the PDF to summarize
        summary_type: Type of summary (brief, bulletwise, detailed)
        use_cache: Whether to use cached summaries if available

    Returns:
        Dictionary with summary and metadata
    """
    with BatchContext(batch_id):
        logger.info(f"[SERVICE] generate_pdf_summary START | pdf={pdf_name} | type={summary_type.value} | use_cache={use_cache}")

        collection_name = _safe_collection_name(batch_id)
        logger.debug(f"[SERVICE] Connecting to Milvus collection: {collection_name}")
        store = MilvusStore(collection_name)

        # Get chunks for the specific PDF
        logger.debug(f"[SERVICE] Querying chunks for PDF: {pdf_name}")
        chunks = store.query_chunks(pdf_name=pdf_name)

        if not chunks:
            logger.error(f"[SERVICE] No chunks found for PDF: {pdf_name}")
            raise ValueError(f"No chunks found for PDF: {pdf_name}")

        total_pages = len(set(c.get("page_no", 0) for c in chunks))
        logger.info(f"[SERVICE] Retrieved {len(chunks)} chunks across {total_pages} pages")

        # Use hierarchical summarizer (handles both small and large docs)
        logger.debug(f"[SERVICE] Calling summarize_chunks")
        result = summarize_chunks(
            batch_id=batch_id,
            chunks=chunks,
            summary_type=summary_type.value,
            pdf_name=pdf_name,
            use_cache=use_cache
        )

        response = {
            "batch_id": batch_id,
            "pdf_name": pdf_name,
            "summary_type": summary_type.value,
            "total_chunks": result.get("total_chunks", len(chunks)),
            "total_pages": total_pages,
            "summary": result["summary"],
            "method": result.get("method", "unknown"),
            "cached": result.get("cached", False),
            "storage_mode": SUMMARY_STORAGE_MODE
        }

        logger.info(f"[SERVICE] generate_pdf_summary END | method={response['method']} | cached={response['cached']} | summary_length={len(response['summary'])} chars")
        return response


def generate_batch_summary(
    batch_id: str,
    summary_type: SummaryType,
    use_cache: bool = True
) -> Dict:
    """
    Generate summary for all PDFs in the batch.

    Uses hierarchical summarization with per-PDF summaries first,
    then combines them into a final summary.

    Args:
        batch_id: Batch identifier
        summary_type: Type of summary (brief, bulletwise, detailed)
        use_cache: Whether to use cached summaries if available

    Returns:
        Dictionary with summary and metadata
    """
    with BatchContext(batch_id):
        logger.info(f"[SERVICE] generate_batch_summary START | type={summary_type.value} | use_cache={use_cache}")

        collection_name = _safe_collection_name(batch_id)
        logger.debug(f"[SERVICE] Connecting to Milvus collection: {collection_name}")
        store = MilvusStore(collection_name)

        # Get PDF names
        logger.debug(f"[SERVICE] Fetching PDF names from collection")
        pdf_names = store.get_pdf_names()

        if not pdf_names:
            logger.error(f"[SERVICE] No PDFs found for batch: {batch_id}")
            raise ValueError(f"No PDFs found for batch: {batch_id}")

        logger.info(f"[SERVICE] Found {len(pdf_names)} PDFs: {pdf_names}")

        # Group chunks by PDF
        pdf_chunks = {}
        for pdf_name in pdf_names:
            logger.debug(f"[SERVICE] Querying chunks for PDF: {pdf_name}")
            chunks = store.query_chunks(pdf_name=pdf_name)
            if chunks:
                pdf_chunks[pdf_name] = chunks
                logger.debug(f"[SERVICE] PDF '{pdf_name}' has {len(chunks)} chunks")

        if not pdf_chunks:
            logger.error(f"[SERVICE] No chunks found for batch: {batch_id}")
            raise ValueError(f"No chunks found for batch: {batch_id}")

        total_chunks = sum(len(chunks) for chunks in pdf_chunks.values())
        logger.info(f"[SERVICE] Total chunks across all PDFs: {total_chunks}")

        # Use hierarchical multi-PDF summarizer
        logger.debug(f"[SERVICE] Calling summarize_multiple_pdfs")
        result = summarize_multiple_pdfs(
            batch_id=batch_id,
            pdf_chunks=pdf_chunks,
            summary_type=summary_type.value,
            use_cache=use_cache
        )

        response = {
            "batch_id": batch_id,
            "pdf_names": result.get("pdf_names", pdf_names),
            "summary_type": summary_type.value,
            "total_pdfs": len(pdf_names),
            "total_chunks": total_chunks,
            "summary": result["summary"],
            "pdf_summaries": result.get("pdf_summaries", {}),
            "cached": result.get("cached", False),
            "storage_mode": SUMMARY_STORAGE_MODE
        }

        logger.info(f"[SERVICE] generate_batch_summary END | total_pdfs={len(pdf_names)} | cached={response['cached']} | summary_length={len(response['summary'])} chars")
        return response


def list_pdfs_in_batch(batch_id: str) -> List[str]:
    """Get list of PDF names in a batch."""
    collection_name = _safe_collection_name(batch_id)
    store = MilvusStore(collection_name)
    return store.get_pdf_names()


def get_cached_summary(
    batch_id: str,
    summary_type: str,
    pdf_name: Optional[str] = None
) -> Optional[Dict]:
    """
    Get a cached summary if available.

    Args:
        batch_id: Batch identifier
        summary_type: brief, bulletwise, or detailed
        pdf_name: PDF name or None for combined summary

    Returns:
        Cached summary data or None
    """
    return get_final_summary(batch_id, summary_type, pdf_name)


# =========================
# Summary Refinement Functions
# =========================

def refine_summary_simple(
    batch_id: str,
    pdf_name: str,
    original_summary: str,
    user_feedback: str,
    config: Optional[SummarizerConfig] = None
) -> Dict:
    """
    Refine a summary using only the original summary and user feedback.

    This is a fast approach that doesn't require access to source chunks.
    Best for: style changes, condensing, reformatting, minor adjustments.

    Args:
        batch_id: Batch identifier
        pdf_name: Name of the PDF
        original_summary: The existing summary to refine
        user_feedback: User's feedback/instructions for refinement
        config: Optional summarizer configuration

    Returns:
        Dictionary with refined summary and metadata
    """
    with BatchContext(batch_id):
        logger.info(f"[REFINE_SIMPLE] START | pdf={pdf_name} | feedback_length={len(user_feedback)}")

        if config is None:
            config = DEFAULT_CONFIG

        # Generate refinement prompt
        prompt = get_simple_refinement_prompt(original_summary, user_feedback)

        # Call LLM for refinement
        refined_summary = _call_llm(
            prompt,
            config,
            context=f"refine_simple_{pdf_name}",
            batch_id=batch_id
        )

        logger.info(f"[REFINE_SIMPLE] END | original_length={len(original_summary)} | refined_length={len(refined_summary)}")

        return {
            "batch_id": batch_id,
            "pdf_name": pdf_name,
            "original_summary": original_summary,
            "refined_summary": refined_summary,
            "user_feedback": user_feedback,
            "method": "simple",
            "chunks_used": 0
        }


def refine_summary_contextual(
    batch_id: str,
    pdf_name: str,
    original_summary: str,
    user_feedback: str,
    top_k: int = 10,
    config: Optional[SummarizerConfig] = None
) -> Dict:
    """
    Refine a summary using original summary, user feedback, and relevant source chunks.

    This approach retrieves chunks semantically similar to the user's feedback,
    allowing the LLM to incorporate missing details from the source document.
    Best for: adding missing information, expanding on specific topics.

    Args:
        batch_id: Batch identifier
        pdf_name: Name of the PDF
        original_summary: The existing summary to refine
        user_feedback: User's feedback/instructions for refinement
        top_k: Number of relevant chunks to retrieve (default: 10)
        config: Optional summarizer configuration

    Returns:
        Dictionary with refined summary and metadata
    """
    with BatchContext(batch_id):
        logger.info(f"[REFINE_CONTEXTUAL] START | pdf={pdf_name} | feedback_length={len(user_feedback)} | top_k={top_k}")

        if config is None:
            config = DEFAULT_CONFIG

        collection_name = _safe_collection_name(batch_id)
        store = MilvusStore(collection_name)

        # Step 1: Generate embedding for user feedback
        logger.debug(f"[REFINE_CONTEXTUAL] Generating embedding for user feedback")
        # E5 requires "query: " prefix for queries
        feedback_with_prefix = f"query: {user_feedback}"
        feedback_embedding = embed_passages([feedback_with_prefix])[0]

        # Step 2: Search for relevant chunks
        logger.debug(f"[REFINE_CONTEXTUAL] Searching for {top_k} relevant chunks")
        search_results = store.search(feedback_embedding, k=top_k)

        # Step 3: Filter chunks for the specific PDF and extract text
        relevant_chunks = []
        if search_results and len(search_results) > 0:
            for hit in search_results[0]:
                entity = hit.entity
                chunk_pdf = entity.get("pdf_name", "")

                # Only include chunks from the target PDF
                if chunk_pdf == pdf_name:
                    chunk_text = entity.get("text", "")
                    page_no = entity.get("page_no", 0)
                    content_type = entity.get("content_type", "text")
                    score = hit.score

                    relevant_chunks.append({
                        "text": chunk_text,
                        "page_no": page_no,
                        "content_type": content_type,
                        "relevance_score": score
                    })

        logger.info(f"[REFINE_CONTEXTUAL] Found {len(relevant_chunks)} relevant chunks for PDF '{pdf_name}'")

        # Step 4: Format relevant context
        if relevant_chunks:
            context_parts = []
            for i, chunk in enumerate(relevant_chunks):
                context_parts.append(
                    f"[Chunk {i+1}] (Page {chunk['page_no']}, {chunk['content_type']}):\n{chunk['text']}"
                )
            relevant_context = "\n\n".join(context_parts)
        else:
            relevant_context = "(No additional context found matching the feedback)"
            logger.warning(f"[REFINE_CONTEXTUAL] No relevant chunks found, proceeding with limited context")

        # Step 5: Generate refinement prompt with context
        prompt = get_contextual_refinement_prompt(
            original_summary,
            user_feedback,
            relevant_context
        )

        # Step 6: Call LLM for refinement
        refined_summary = _call_llm(
            prompt,
            config,
            context=f"refine_contextual_{pdf_name}",
            batch_id=batch_id
        )

        logger.info(f"[REFINE_CONTEXTUAL] END | original_length={len(original_summary)} | refined_length={len(refined_summary)} | chunks_used={len(relevant_chunks)}")

        return {
            "batch_id": batch_id,
            "pdf_name": pdf_name,
            "original_summary": original_summary,
            "refined_summary": refined_summary,
            "user_feedback": user_feedback,
            "method": "contextual",
            "chunks_used": len(relevant_chunks),
            "chunks_detail": [
                {
                    "page_no": c["page_no"],
                    "content_type": c["content_type"],
                    "relevance_score": c["relevance_score"],
                    "preview": c["text"][:100] + "..." if len(c["text"]) > 100 else c["text"]
                }
                for c in relevant_chunks
            ]
        }
