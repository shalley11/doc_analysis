"""
Gemma-3 Text Task API with comprehensive logging and refinement cycle.

Endpoints:
- POST /process-text: Process text with various tasks (returns request_id)
- POST /refine/{request_id}: Refine previous result based on feedback (uses current_result)
- POST /regenerate/{request_id}: Regenerate from original text with new instructions (uses original_text)
- GET /refine/{request_id}: Get current refinement status
- DELETE /refine/{request_id}: End refinement session
- GET /health: Health check endpoint
"""
from fastapi import FastAPI, HTTPException, Path
from schemas import (
    TextTaskRequest,
    TextTaskResponse,
    RefinementRequest,
    RefinementResponse,
    RefinementStatusResponse,
    RegenerateRequest,
    RegenerateResponse
)
from prompts import get_prompt, get_refinement_prompt, get_regenerate_prompt
from llm_client import generate_text_with_logging
from config import (
    DEFAULT_MODEL,
    get_model_context_length,
    estimate_tokens,
    CONTEXT_REJECT_THRESHOLD
)
from logging_config import (
    setup_llm_logging,
    get_llm_logger,
    RequestContext
)
from refinement_store import get_refinement_store, RefinementData

# Initialize logging
setup_llm_logging()
logger = get_llm_logger()

app = FastAPI(
    title="Gemma-3 Text Task API",
    description="""
## Text Processing API with Refinement & Regeneration

Supports **Summary, Translation, Rephrase, and Deduplication** tasks with iterative improvement cycles.

### Core Endpoints

| Endpoint | Description |
|----------|-------------|
| `POST /process-text` | Process text and get a `request_id` for further refinement |
| `POST /refine/{id}` | Refine current result (uses `current_result` - smaller context) |
| `POST /regenerate/{id}` | Regenerate from original text (uses `original_text` - fresh start) |
| `GET /refine/{id}` | Get session status, counts, and TTL |
| `DELETE /refine/{id}` | End session and cleanup |

### Refine vs Regenerate

- **Refine**: Incrementally improve the current output. Best for polishing, tweaking, minor corrections.
- **Regenerate**: Start fresh from original text with new instructions. Best when output went in wrong direction.

### Context Length Protection

Requests exceeding model context limit (e.g., 8192 tokens for gemma3) are rejected with HTTP 400 to prevent truncation.
""",
    version="2.1.0"
)


@app.on_event("startup")
async def startup_event():
    logger.info("=" * 50)
    logger.info("TEXT TASK API STARTING")
    logger.info(f"Default model: {DEFAULT_MODEL}")
    logger.info("=" * 50)

    # Initialize refinement store
    try:
        store = get_refinement_store()
        logger.info("Refinement store initialized")
    except Exception as e:
        logger.warning(f"Refinement store initialization failed: {e}")


@app.on_event("shutdown")
async def shutdown_event():
    logger.info("TEXT TASK API SHUTTING DOWN")


@app.post("/process-text", response_model=TextTaskResponse)
def process_text(req: TextTaskRequest):
    """
    Process text with the specified task.

    Returns a request_id that can be used for subsequent refinements.

    Tasks:
    - summary: Generate a summary of the text
    - translate: Translate text to another language
    - rephrase: Rephrase the text
    - remove_repetitions: Remove duplicate content

    All requests are logged with:
    - Request ID for tracing
    - User ID (if provided)
    - Prompt details
    - Response metrics (latency, length)
    """
    with RequestContext(user_id=req.user_id) as ctx:
        user_info = f" | user_id={req.user_id}" if req.user_id else ""
        logger.info(
            f"API_REQUEST | task={req.task} | model={req.model or DEFAULT_MODEL} | "
            f"text_length={len(req.text)} | summary_type={req.summary_type}{user_info}"
        )

        try:
            # Build prompt
            prompt = get_prompt(
                task=req.task,
                text=req.text,
                summary_type=req.summary_type
            )

            # Check context length before processing
            model_name = req.model or DEFAULT_MODEL
            context_limit = get_model_context_length(model_name)
            estimated_tokens = estimate_tokens(prompt)
            usage_percent = (estimated_tokens / context_limit) * 100 if context_limit > 0 else 0

            if usage_percent >= CONTEXT_REJECT_THRESHOLD:
                logger.warning(
                    f"CONTEXT_EXCEEDED | model={model_name} | "
                    f"estimated_tokens={estimated_tokens} | context_limit={context_limit} | "
                    f"usage={usage_percent:.1f}%{user_info}"
                )
                raise HTTPException(
                    status_code=400,
                    detail={
                        "error": "context_length_exceeded",
                        "message": "Input text is too long and would be truncated. Please reduce the input size.",
                        "model": model_name,
                        "context_limit": context_limit,
                        "estimated_tokens": estimated_tokens,
                        "usage_percent": round(usage_percent, 2)
                    }
                )

            # Generate with logging (user_id is in context)
            output = generate_text_with_logging(
                prompt=prompt,
                model=req.model,
                task=req.task
            )

            # Store in refinement store for potential refinement cycle
            store = get_refinement_store()
            refinement_data = store.create(
                task=req.task,
                result=output,
                original_text=req.text,
                model=req.model or DEFAULT_MODEL,
                user_id=req.user_id
            )

            logger.info(
                f"API_RESPONSE | task={req.task} | output_length={len(output)} | "
                f"request_id={refinement_data.request_id} | status=success{user_info}"
            )

            return TextTaskResponse(
                request_id=refinement_data.request_id,
                task=req.task,
                model=req.model or DEFAULT_MODEL,
                output=output,
                user_id=req.user_id
            )

        except Exception as e:
            logger.error(
                f"API_ERROR | task={req.task} | error={str(e)}{user_info}",
                exc_info=True
            )
            raise HTTPException(status_code=500, detail=str(e))


@app.post("/refine/{request_id}", response_model=RefinementResponse)
def refine_result(
    request_id: str = Path(..., description="Request ID from process-text response"),
    req: RefinementRequest = ...
):
    """
    Refine a previous result based on user feedback.

    This endpoint:
    1. Retrieves the stored result from Redis
    2. Combines it with user feedback
    3. Generates a refined version
    4. Overwrites the stored result (only keeps latest)

    Use cases: proofreading, rephrasing, iterative improvements
    """
    user_info = f" | user_id={req.user_id}" if req.user_id else ""
    logger.info(
        f"REFINE_REQUEST | request_id={request_id} | "
        f"feedback_length={len(req.user_feedback)}{user_info}"
    )

    try:
        # Get stored data
        store = get_refinement_store()
        data = store.get(request_id)

        if not data:
            logger.warning(f"REFINE_NOT_FOUND | request_id={request_id}")
            raise HTTPException(
                status_code=404,
                detail=f"Request ID '{request_id}' not found or expired. "
                       f"Please start a new session with /process-text"
            )

        # Build refinement prompt
        prompt = get_refinement_prompt(
            current_result=data.current_result,
            user_feedback=req.user_feedback,
            task=data.task
        )

        # Check context length before processing
        context_limit = get_model_context_length(data.model)
        est_tokens = estimate_tokens(prompt)
        usage_percent = (est_tokens / context_limit) * 100 if context_limit > 0 else 0

        if usage_percent >= CONTEXT_REJECT_THRESHOLD:
            logger.warning(
                f"CONTEXT_EXCEEDED | model={data.model} | "
                f"estimated_tokens={est_tokens} | context_limit={context_limit} | "
                f"usage={usage_percent:.1f}%{user_info}"
            )
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "context_length_exceeded",
                    "message": "Combined text (result + feedback) is too long. Please reduce the feedback size.",
                    "model": data.model,
                    "context_limit": context_limit,
                    "estimated_tokens": est_tokens,
                    "usage_percent": round(usage_percent, 2)
                }
            )

        # Set user context for logging
        with RequestContext(user_id=req.user_id or data.user_id):
            # Generate refined result
            refined_output = generate_text_with_logging(
                prompt=prompt,
                model=data.model,
                task=f"refine_{data.task}"
            )

            # Update store (overwrites previous result)
            updated_data = store.update(
                request_id=request_id,
                new_result=refined_output,
                user_id=req.user_id
            )

            logger.info(
                f"REFINE_RESPONSE | request_id={request_id} | "
                f"refinement_count={updated_data.refinement_count} | "
                f"output_length={len(refined_output)}{user_info}"
            )

            return RefinementResponse(
                request_id=request_id,
                refined_output=refined_output,
                refinement_count=updated_data.refinement_count,
                task=data.task,
                model=data.model,
                user_id=updated_data.user_id
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            f"REFINE_ERROR | request_id={request_id} | error={str(e)}{user_info}",
            exc_info=True
        )
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/regenerate/{request_id}", response_model=RegenerateResponse)
def regenerate_result(
    request_id: str = Path(..., description="Request ID from process-text response"),
    req: RegenerateRequest = ...
):
    """
    Regenerate output from ORIGINAL TEXT with new instructions.

    Unlike /refine which modifies the current result, /regenerate:
    1. Uses the ORIGINAL input text (not current result)
    2. Applies user's new instructions to regenerate fresh output
    3. Overwrites the current result

    Use this when:
    - The current result went in wrong direction
    - You want to start fresh with different instructions
    - You need output based on original content, not refined version
    """
    user_info = f" | user_id={req.user_id}" if req.user_id else ""
    logger.info(
        f"REGENERATE_REQUEST | request_id={request_id} | "
        f"feedback_length={len(req.user_feedback)}{user_info}"
    )

    try:
        # Get stored data
        store = get_refinement_store()
        data = store.get(request_id)

        if not data:
            logger.warning(f"REGENERATE_NOT_FOUND | request_id={request_id}")
            raise HTTPException(
                status_code=404,
                detail=f"Request ID '{request_id}' not found or expired. "
                       f"Please start a new session with /process-text"
            )

        # Build regeneration prompt (uses ORIGINAL TEXT)
        prompt = get_regenerate_prompt(
            original_text=data.original_text,
            user_feedback=req.user_feedback,
            task=data.task
        )

        # Check context length before processing
        context_limit = get_model_context_length(data.model)
        est_tokens = estimate_tokens(prompt)
        usage_percent = (est_tokens / context_limit) * 100 if context_limit > 0 else 0

        if usage_percent >= CONTEXT_REJECT_THRESHOLD:
            logger.warning(
                f"CONTEXT_EXCEEDED | model={data.model} | "
                f"estimated_tokens={est_tokens} | context_limit={context_limit} | "
                f"usage={usage_percent:.1f}%{user_info}"
            )
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "context_length_exceeded",
                    "message": "Original text + feedback is too long. Please reduce the feedback size or start with shorter text.",
                    "model": data.model,
                    "context_limit": context_limit,
                    "estimated_tokens": est_tokens,
                    "usage_percent": round(usage_percent, 2)
                }
            )

        # Set user context for logging
        with RequestContext(user_id=req.user_id or data.user_id):
            # Generate regenerated result
            regenerated_output = generate_text_with_logging(
                prompt=prompt,
                model=data.model,
                task=f"regenerate_{data.task}"
            )

            # Update store with regeneration (overwrites previous result)
            updated_data = store.update_regeneration(
                request_id=request_id,
                new_result=regenerated_output,
                user_id=req.user_id
            )

            logger.info(
                f"REGENERATE_RESPONSE | request_id={request_id} | "
                f"regeneration_count={updated_data.regeneration_count} | "
                f"output_length={len(regenerated_output)}{user_info}"
            )

            return RegenerateResponse(
                request_id=request_id,
                regenerated_output=regenerated_output,
                regeneration_count=updated_data.regeneration_count,
                task=data.task,
                model=data.model,
                user_id=updated_data.user_id
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            f"REGENERATE_ERROR | request_id={request_id} | error={str(e)}{user_info}",
            exc_info=True
        )
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/refine/{request_id}", response_model=RefinementStatusResponse)
def get_refinement_status(
    request_id: str = Path(..., description="Request ID to check")
):
    """
    Get the current status of a refinement session.

    Returns the current result and metadata.
    """
    logger.debug(f"REFINE_STATUS_REQUEST | request_id={request_id}")

    try:
        store = get_refinement_store()
        data = store.get(request_id)

        if not data:
            raise HTTPException(
                status_code=404,
                detail=f"Request ID '{request_id}' not found or expired"
            )

        ttl = store.get_ttl(request_id)

        return RefinementStatusResponse(
            request_id=data.request_id,
            task=data.task,
            model=data.model,
            current_output=data.current_result,
            refinement_count=data.refinement_count,
            regeneration_count=data.regeneration_count,
            user_id=data.user_id,
            created_at=data.created_at,
            updated_at=data.updated_at,
            ttl_seconds=ttl
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"REFINE_STATUS_ERROR | request_id={request_id} | error={str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/refine/{request_id}")
def end_refinement_session(
    request_id: str = Path(..., description="Request ID to delete")
):
    """
    End a refinement session and clean up stored data.

    Call this when the user is satisfied with the result
    or wants to abandon the refinement cycle.
    """
    logger.info(f"REFINE_DELETE_REQUEST | request_id={request_id}")

    try:
        store = get_refinement_store()

        # Get data first to log user_id
        data = store.get(request_id)
        user_info = f" | user_id={data.user_id}" if data and data.user_id else ""

        if store.delete(request_id):
            logger.info(f"REFINE_DELETED | request_id={request_id}{user_info}")
            return {
                "status": "deleted",
                "request_id": request_id,
                "message": "Refinement session ended successfully"
            }
        else:
            raise HTTPException(
                status_code=404,
                detail=f"Request ID '{request_id}' not found"
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"REFINE_DELETE_ERROR | request_id={request_id} | error={str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/refine/{request_id}/extend")
def extend_refinement_ttl(
    request_id: str = Path(..., description="Request ID to extend"),
    ttl_seconds: int = 3600
):
    """
    Extend the TTL for a refinement session.

    Default extension is 1 hour (3600 seconds).
    """
    logger.info(f"REFINE_EXTEND_REQUEST | request_id={request_id} | ttl={ttl_seconds}")

    try:
        store = get_refinement_store()

        if store.extend_ttl(request_id, ttl_seconds):
            new_ttl = store.get_ttl(request_id)
            return {
                "status": "extended",
                "request_id": request_id,
                "ttl_seconds": new_ttl
            }
        else:
            raise HTTPException(
                status_code=404,
                detail=f"Request ID '{request_id}' not found"
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"REFINE_EXTEND_ERROR | request_id={request_id} | error={str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
def health():
    """Health check endpoint."""
    return {"status": "ok", "model": DEFAULT_MODEL}


@app.get("/logs/stats")
def get_log_stats():
    """Get logging statistics (for monitoring)."""
    from pathlib import Path
    from logging_config import LOG_DIR

    stats = {}
    for log_file in LOG_DIR.glob("*.log"):
        stats[log_file.name] = {
            "size_bytes": log_file.stat().st_size,
            "size_mb": round(log_file.stat().st_size / (1024 * 1024), 2)
        }

    return {
        "log_directory": str(LOG_DIR),
        "files": stats
    }
