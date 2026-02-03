from typing import List, Optional
import uuid
import shutil

from fastapi import FastAPI, UploadFile, WebSocket, WebSocketDisconnect, Query, HTTPException
from pydantic import BaseModel, Field

# ---- logging ----
from doc_analysis.logging_config import get_api_logger, setup_all_loggers
setup_all_loggers()
logger = get_api_logger()

# ---- app must be defined BEFORE decorators ----
app = FastAPI(title="Document Analysis API")

@app.on_event("startup")
async def startup_event():
    logger.info("Document Analysis API starting...")
    await ws_manager.start()
    logger.info("Document Analysis API started")


@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Document Analysis API shutting down...")
    await ws_manager.stop()
    logger.info("Document Analysis API stopped")

# ---- config ----
from doc_analysis.config import MAX_PDFS_PER_BATCH, EMBEDDING_DIM, BATCH_TTL_SECONDS

# ---- job store ----
from doc_analysis.jobs.job_store import (
    create_job,
    get_job
)

# ---- ingestion enqueue ----
from doc_analysis.queues.enqueue import enqueue_pdf_ingestion

# ---- websocket manager ----
from doc_analysis.realtime.ws_manager import WebSocketManager
ws_manager = WebSocketManager()

def job_update_hook(batch_id, job_data):
    import asyncio
    asyncio.create_task(
        ws_manager.broadcast(batch_id, job_data)
    )


@app.post("/upload-pdfs")
def upload_pdfs(
    files: List[UploadFile],
    start_page: Optional[int] = Query(None, description="Start page (1-indexed, inclusive)"),
    end_page: Optional[int] = Query(None, description="End page (1-indexed, inclusive)")
):
    logger.info(f"Received upload request with {len(files)} file(s)")

    if len(files) > MAX_PDFS_PER_BATCH:
        logger.warning(f"Upload rejected: {len(files)} files exceeds limit of {MAX_PDFS_PER_BATCH}")
        return {
            "error": f"Maximum {MAX_PDFS_PER_BATCH} PDFs allowed per batch"
        }

    batch_id = str(uuid.uuid4())
    logger.info(f"Created batch_id: {batch_id}")
    if start_page or end_page:
        logger.info(f"Page range: {start_page} to {end_page}")

    # create job entry
    create_job(batch_id, [f.filename for f in files])

    paths = []

    for idx, file in enumerate(files):
        pdf_id = f"pdf_{idx + 1}"
        path = f"/tmp/{batch_id}_{pdf_id}.pdf"

        with open(path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        paths.append(path)
        logger.debug(f"Saved {file.filename} to {path}")

    enqueue_pdf_ingestion(
        batch_id=batch_id,
        filenames=[f.filename for f in files],
        paths=paths,
        embedding_dim=EMBEDDING_DIM,
        ttl_seconds=BATCH_TTL_SECONDS,
        start_page=start_page,
        end_page=end_page
    )

    logger.info(f"Batch {batch_id} queued for processing with {len(files)} PDFs")

    return {
        "batch_id": batch_id,
        "pdf_count": len(files),
        "status": "queued",
        "page_range": {"start": start_page, "end": end_page} if start_page or end_page else None
    }

@app.get("/job-status/{batch_id}")
def job_status(
    batch_id: str,
    module: Optional[str] = Query(None, description="Module to filter: document_processing, summarization, or None for all")
):
    """
    Get job status with optional module filter.

    Args:
        batch_id: Batch identifier
        module: Optional filter - 'document_processing', 'summarization', or None for complete status
    """
    logger.debug(f"Job status request for batch_id: {batch_id}, module: {module}")

    job = get_job(batch_id)

    if module == "document_processing":
        return {
            "batch_id": job["batch_id"],
            "status": job["status"],
            "created_at": job.get("created_at"),
            "updated_at": job.get("updated_at"),
            "document_processing": job["document_processing"]
        }
    elif module == "summarization":
        return {
            "batch_id": job["batch_id"],
            "status": job["status"],
            "created_at": job.get("created_at"),
            "updated_at": job.get("updated_at"),
            "summarization": job["summarization"]
        }
    else:
        # Return complete job status
        return job

@app.websocket("/ws/job-status/{batch_id}")
async def job_status_ws(websocket: WebSocket, batch_id: str):
    """Legacy WebSocket endpoint for job status (maps to PDF updates)."""
    await ws_manager.connect_pdf(batch_id, websocket)

    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        await ws_manager.disconnect_pdf(batch_id, websocket)


@app.websocket("/ws/pdf/{batch_id}")
async def pdf_status_ws(websocket: WebSocket, batch_id: str):
    """
    WebSocket endpoint for PDF processing updates.

    Events:
        - pdf.started: Processing started
        - pdf.extracting: Extracting pages
        - pdf.extracted: Extraction complete
        - pdf.chunking: Creating chunks
        - pdf.chunked: Chunking complete
        - pdf.embedding: Generating embeddings
        - pdf.embedded: Embedding complete
        - pdf.completed: All processing done
        - pdf.failed: Error occurred
    """
    await ws_manager.connect_pdf(batch_id, websocket)

    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        await ws_manager.disconnect_pdf(batch_id, websocket)


@app.websocket("/ws/summary/{batch_id}")
async def summary_status_ws(websocket: WebSocket, batch_id: str):
    """
    WebSocket endpoint for summarization updates.

    Events:
        - summary.started: Summarization started
        - summary.cache_hit: Using cached summary
        - summary.method_selected: Direct or hierarchical
        - summary.batch_started: Batch N/M started
        - summary.batch_completed: Batch N/M completed
        - summary.reduce_started: Combining summaries
        - summary.reduce_level: Reduce level progress
        - summary.llm_call_started: LLM call started
        - summary.llm_call_completed: LLM call completed
        - summary.completed: Summary ready
        - summary.failed: Error occurred
    """
    await ws_manager.connect_summary(batch_id, websocket)

    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        await ws_manager.disconnect_summary(batch_id, websocket)


@app.get("/ws/stats")
def get_websocket_stats():
    """Get WebSocket connection statistics."""
    return ws_manager.get_connection_stats()


# ---- Summary APIs ----
from doc_analysis.summarization.summary_service import (
    generate_pdf_summary,
    generate_batch_summary,
    list_pdfs_in_batch,
    SummaryType,
    refine_summary_simple,
    refine_summary_contextual,
    get_cached_summary
)


@app.get("/summary/pdfs/{batch_id}")
def get_pdfs_in_batch(batch_id: str):
    """Get list of PDF names in a batch for summary generation."""
    try:
        pdf_names = list_pdfs_in_batch(batch_id)
        return {
            "batch_id": batch_id,
            "pdf_names": pdf_names,
            "count": len(pdf_names)
        }
    except Exception as e:
        logger.error(f"Failed to list PDFs: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/summary/pdf")
def summarize_pdf(
    batch_id: str = Query(..., description="Batch ID"),
    pdf_name: Optional[str] = Query(None, description="PDF name (optional - if not provided, summarizes all PDFs individually)"),
    summary_type: SummaryType = Query(SummaryType.BRIEF, description="Summary type: brief, bulletwise, or detailed")
):
    """
    Generate summary for PDF document(s).

    - If pdf_name provided: Returns summary for that specific PDF
    - If pdf_name not provided: Returns individual summaries for ALL PDFs in batch
    """
    if pdf_name:
        # Summarize specific PDF
        logger.info(f"Summary request for PDF: {pdf_name}, type: {summary_type}, batch: {batch_id}")

        try:
            result = generate_pdf_summary(
                batch_id=batch_id,
                pdf_name=pdf_name,
                summary_type=summary_type
            )
            return result
        except ValueError as e:
            raise HTTPException(status_code=404, detail=str(e))
        except Exception as e:
            logger.error(f"Summary generation failed: {str(e)}", exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))
    else:
        # Summarize all PDFs individually
        logger.info(f"Summary request for ALL PDFs individually, type: {summary_type}, batch: {batch_id}")

        try:
            pdf_names = list_pdfs_in_batch(batch_id)

            if not pdf_names:
                raise HTTPException(status_code=404, detail=f"No PDFs found for batch: {batch_id}")

            summaries = {}
            for name in pdf_names:
                result = generate_pdf_summary(
                    batch_id=batch_id,
                    pdf_name=name,
                    summary_type=summary_type
                )
                summaries[name] = result

            return {
                "batch_id": batch_id,
                "summary_type": summary_type.value,
                "total_pdfs": len(pdf_names),
                "summaries": summaries
            }
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Summary generation failed: {str(e)}", exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))


@app.get("/summary/all")
def summarize_all_pdfs(
    batch_id: str = Query(..., description="Batch ID"),
    summary_type: SummaryType = Query(SummaryType.BRIEF, description="Summary type: brief, bulletwise, or detailed")
):
    """
    Generate COMBINED summary for all PDFs in the batch.

    Unlike /summary/pdf (which gives individual summaries), this combines
    all PDF content into a single unified summary.
    """
    logger.info(f"Combined summary request for all PDFs, type: {summary_type}, batch: {batch_id}")

    try:
        result = generate_batch_summary(
            batch_id=batch_id,
            summary_type=summary_type
        )
        return result
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Batch summary generation failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ---- Summary Refinement APIs ----

class RefinementRequest(BaseModel):
    """Request body for summary refinement endpoints."""
    batch_id: str = Field(..., description="Batch ID")
    pdf_name: str = Field(..., description="Name of the PDF to refine summary for")
    summary_type: SummaryType = Field(
        SummaryType.DETAILED,
        description="Type of the original summary (brief, bulletwise, detailed, executive)"
    )
    user_feedback: str = Field(
        ...,
        description="User's feedback or instructions for refinement",
        min_length=1,
        max_length=2000
    )
    original_summary: Optional[str] = Field(
        None,
        description="Original summary to refine. If not provided, will be fetched from cache."
    )
    user_id: Optional[str] = Field(
        None,
        description="Optional user identifier for tracking and logging purposes"
    )


class ContextualRefinementRequest(RefinementRequest):
    """Request body for contextual refinement with additional options."""
    top_k: int = Field(
        10,
        description="Number of relevant source chunks to retrieve (1-50)",
        ge=1,
        le=50
    )


@app.post("/summary/refine/simple")
def refine_summary_simple_endpoint(request: RefinementRequest):
    """
    Refine a summary using ONLY the original summary and user feedback.

    **Option A: Summary-Only Refinement**

    This endpoint is fast and doesn't require access to source chunks.
    The LLM refines the summary based solely on the provided feedback.

    Best for:
    - Style changes (make it more formal, casual, etc.)
    - Condensing or expanding existing content
    - Reformatting (bullets to paragraphs, etc.)
    - Minor adjustments to tone or focus

    Limitations:
    - Cannot add new information not present in original summary
    - May not be able to add specific details the user requests

    Example:
    ```json
    {
        "batch_id": "abc-123",
        "pdf_name": "report.pdf",
        "summary_type": "detailed",
        "user_feedback": "Make it more concise and use bullet points",
        "user_id": "user123"
    }
    ```
    """
    user_info = f", user_id: {request.user_id}" if request.user_id else ""
    logger.info(f"Simple refinement request for PDF: {request.pdf_name}, batch: {request.batch_id}{user_info}")

    try:
        # Get original summary if not provided
        original_summary = request.original_summary
        if not original_summary:
            cached = get_cached_summary(
                request.batch_id,
                request.summary_type.value,
                request.pdf_name
            )
            if not cached:
                raise HTTPException(
                    status_code=404,
                    detail=f"No cached summary found for PDF '{request.pdf_name}' with type '{request.summary_type.value}'. "
                           f"Either generate a summary first or provide the original_summary in the request."
                )
            original_summary = cached.get("summary", "")

        result = refine_summary_simple(
            batch_id=request.batch_id,
            pdf_name=request.pdf_name,
            original_summary=original_summary,
            user_feedback=request.user_feedback
        )

        # Add user_id to response if provided
        if request.user_id:
            result["user_id"] = request.user_id

        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Simple refinement failed: {str(e)}{user_info}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/summary/refine/contextual")
def refine_summary_contextual_endpoint(request: ContextualRefinementRequest):
    """
    Refine a summary using original summary, user feedback, AND relevant source chunks.

    **Option C: Hybrid Contextual Refinement**

    This endpoint uses vector search to find source chunks relevant to the user's
    feedback, then provides both the original summary and relevant chunks to the LLM.

    How it works:
    1. Generates embedding for user feedback
    2. Searches vector store for chunks similar to the feedback
    3. Filters chunks belonging to the target PDF
    4. Sends original summary + feedback + relevant chunks to LLM
    5. LLM refines the summary incorporating new information

    Best for:
    - Adding missing details ("include more about financial metrics")
    - Expanding on specific topics ("elaborate on the methodology section")
    - Incorporating information that was omitted from original summary
    - Answering specific questions about content

    Parameters:
    - top_k: Number of relevant chunks to retrieve (default: 10, max: 50)

    Example:
    ```json
    {
        "batch_id": "abc-123",
        "pdf_name": "report.pdf",
        "summary_type": "detailed",
        "user_feedback": "Include more details about the financial projections and risks",
        "top_k": 15,
        "user_id": "user123"
    }
    ```

    Response includes:
    - refined_summary: The new refined summary
    - chunks_used: Number of relevant chunks found
    - chunks_detail: Preview of each chunk used (page, type, relevance score)
    """
    user_info = f", user_id: {request.user_id}" if request.user_id else ""
    logger.info(f"Contextual refinement request for PDF: {request.pdf_name}, batch: {request.batch_id}, top_k: {request.top_k}{user_info}")

    try:
        # Get original summary if not provided
        original_summary = request.original_summary
        if not original_summary:
            cached = get_cached_summary(
                request.batch_id,
                request.summary_type.value,
                request.pdf_name
            )
            if not cached:
                raise HTTPException(
                    status_code=404,
                    detail=f"No cached summary found for PDF '{request.pdf_name}' with type '{request.summary_type.value}'. "
                           f"Either generate a summary first or provide the original_summary in the request."
                )
            original_summary = cached.get("summary", "")

        result = refine_summary_contextual(
            batch_id=request.batch_id,
            pdf_name=request.pdf_name,
            original_summary=original_summary,
            user_feedback=request.user_feedback,
            top_k=request.top_k
        )

        # Add user_id to response if provided
        if request.user_id:
            result["user_id"] = request.user_id

        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Contextual refinement failed: {str(e)}{user_info}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
