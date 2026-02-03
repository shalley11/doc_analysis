import uuid
import json
import asyncio
from pathlib import Path
import shutil
from typing import List, Dict, Set

from fastapi import FastAPI, UploadFile, File, Query, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import redis
from rq import Queue

from pydantic import BaseModel
from typing import Optional, List as TypingList

from pdf.pdf_utils import process_page, page_to_markdown
from jobs.jobs import process_pdf_batch, process_pdf_batch_multimodal, process_pdf_batch_structured
from jobs.job_state import init_job, get_job
from jobs.processing_status import (
    get_batch_status,
    get_batch_status_summary,
    init_batch_status
)
from embedding.embedding_client import EmbeddingClient
from vector_store.milvus_store import MilvusVectorStore
from qa.retriever import Retriever
from qa.generator import create_generator, get_default_generator
from qa.qa_service import QAService
from qa.summary_service import SummaryService, get_summary_service


# Q&A Request Models
class QuestionRequest(BaseModel):
    question: str
    top_k: int = 5
    temperature: float = 0.7
    include_sources: bool = True
    llm_provider: Optional[str] = None  # ollama, openai, anthropic, gemini
    llm_model: Optional[str] = None


class SummaryRequest(BaseModel):
    summary_type: str = "brief"  # brief, detailed, bullets
    max_chunks: int = 20
    temperature: float = 0.7
    llm_provider: Optional[str] = None
    llm_model: Optional[str] = None


class AdvancedSummaryRequest(BaseModel):
    scope: str = "all"  # "topic", "document", "all"
    summary_format: str = "brief"  # brief, detailed, bullets
    num_topics: int = 5  # For topic-wise summary
    max_chunks: int = 100
    temperature: float = 0.7
    llm_provider: Optional[str] = None
    llm_model: Optional[str] = None


class ChatMessage(BaseModel):
    role: str  # user or assistant
    content: str


class ChatRequest(BaseModel):
    messages: TypingList[ChatMessage]
    top_k: int = 5
    temperature: float = 0.7
    llm_provider: Optional[str] = None
    llm_model: Optional[str] = None


class DocumentSummaryRequest(BaseModel):
    summary_type: str = "detailed"  # "brief", "detailed", "bullets"


class CorpusSummaryRequest(BaseModel):
    summary_type: str = "detailed"  # "brief", "detailed" for individual docs before corpus
    include_individual: bool = True  # Include individual document summaries in response


# Configuration
EMBEDDING_SERVICE_URL = "http://localhost:8000"
MILVUS_HOST = "localhost"
MILVUS_PORT = 19530


# =============================================================================
# WebSocket Connection Manager
# =============================================================================

class ConnectionManager:
    """Manages WebSocket connections for real-time status updates."""

    def __init__(self):
        # batch_id -> set of WebSocket connections
        self.active_connections: Dict[str, Set[WebSocket]] = {}
        self._redis_tasks: Dict[str, asyncio.Task] = {}

    async def connect(self, websocket: WebSocket, batch_id: str):
        """Accept connection and add to batch subscribers."""
        await websocket.accept()
        if batch_id not in self.active_connections:
            self.active_connections[batch_id] = set()
        self.active_connections[batch_id].add(websocket)

        # Start Redis listener for this batch if not already running
        if batch_id not in self._redis_tasks:
            self._redis_tasks[batch_id] = asyncio.create_task(
                self._redis_listener(batch_id)
            )

    def disconnect(self, websocket: WebSocket, batch_id: str):
        """Remove connection from batch subscribers."""
        if batch_id in self.active_connections:
            self.active_connections[batch_id].discard(websocket)
            # Clean up if no more connections for this batch
            if not self.active_connections[batch_id]:
                del self.active_connections[batch_id]
                # Cancel Redis listener task
                if batch_id in self._redis_tasks:
                    self._redis_tasks[batch_id].cancel()
                    del self._redis_tasks[batch_id]

    async def broadcast(self, batch_id: str, message: dict):
        """Send message to all connections subscribed to a batch."""
        if batch_id not in self.active_connections:
            return

        disconnected = set()
        for connection in self.active_connections[batch_id]:
            try:
                await connection.send_json(message)
            except Exception:
                disconnected.add(connection)

        # Clean up disconnected clients
        for conn in disconnected:
            self.active_connections[batch_id].discard(conn)

    async def _redis_listener(self, batch_id: str):
        """Listen to Redis pub/sub and broadcast to WebSocket clients."""
        redis_async = redis.Redis(host="localhost", port=6379, decode_responses=True)
        pubsub = redis_async.pubsub()
        pubsub.subscribe(f"status:{batch_id}")

        try:
            while batch_id in self.active_connections:
                message = pubsub.get_message(ignore_subscribe_messages=True, timeout=0.1)
                if message and message['type'] == 'message':
                    try:
                        data = json.loads(message['data'])
                        await self.broadcast(batch_id, data)
                    except json.JSONDecodeError:
                        pass
                await asyncio.sleep(0.1)
        except asyncio.CancelledError:
            pass
        finally:
            pubsub.unsubscribe(f"status:{batch_id}")
            pubsub.close()


# Global connection manager
ws_manager = ConnectionManager()


app = FastAPI(
    title="PDF Analysis API",
    description="Upload PDFs for async batch processing with multimodal support",
    version="2.0.0"
)

# CORS middleware for UI
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files for serving images/tables
app.mount("/static", StaticFiles(directory="outputs"), name="static")

# Mount UI files
app.mount("/ui", StaticFiles(directory="ui", html=True), name="ui")

redis_conn = redis.Redis(host="localhost", port=6379)
queue = Queue("pdf-processing", connection=redis_conn)


@app.post("/api/v1/pdf/analyze", tags=["PDFs"])
def analyze_pdfs(
    files: List[UploadFile] = File(...),
    preview_only: str = Query("yes"),
    preview_pages: int = Query(1)
):
    batch_id = str(uuid.uuid4())
    input_dir = Path("uploads") / batch_id
    output_dir = Path("outputs") / batch_id
    input_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    previews = {}

    for file in files:
        pdf_path = input_dir / file.filename
        with open(pdf_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        if preview_only.lower() == "yes":
            import fitz
            doc = fitz.open(pdf_path)
            pages_preview = []
            for i, page in enumerate(doc[:preview_pages]):
                page_result = process_page(page, i, file.filename, output_dir / file.filename)
                pages_preview.append(page_to_markdown(page_result))
            previews[file.filename] = "\n\n".join(pages_preview)

    job = queue.enqueue(process_pdf_batch, batch_id, str(input_dir), str(output_dir))
    init_job(batch_id, rq_job_id=job.id)

    return JSONResponse({
        "batch_id": batch_id,
        "rq_job_id": job.id,
        "uploaded_files": [f.filename for f in files],
        "message": "Files uploaded. Background processing started.",
        "preview_markdown": previews if preview_only.lower() == "yes" else None
    })


@app.get("/api/v1/pdf/status/{batch_id}", tags=["Status"])
def job_status(batch_id: str):
    job = get_job(batch_id)
    if not job:
        return {"error": "batch not found"}

    return {
        "batch_id": batch_id,
        "status": job["state"],
        "progress_percent": job["progress"],
        "rq_job_id": job["rq_job_id"],
        "total_pages": job["total_pages"],
        "processed_pages": job["processed_pages"],
        "chunk_count": job["chunk_count"],
        "milvus_indexed": job["milvus_indexed"],
        "pdfs": job["pdfs"]
    }


# =============================================================================
# Detailed Status API (for UI)
# =============================================================================

@app.get("/api/v2/status/{batch_id}", tags=["Status - Detailed"])
def get_detailed_status(batch_id: str):
    """
    Get detailed processing status with stage-by-stage breakdown.

    Returns comprehensive status including:
    - Overall batch progress
    - Current processing stage
    - Per-PDF status with individual stage progress
    - Timing information (duration, started_at, completed_at)
    - Error details if any stage failed

    Ideal for building interactive progress UIs.
    """
    status = get_batch_status(batch_id)
    if not status:
        # Fallback to basic job status
        job = get_job(batch_id)
        if not job:
            return JSONResponse({"error": "Batch not found"}, status_code=404)
        return JSONResponse({
            "batch_id": batch_id,
            "state": job.get("state", "unknown"),
            "progress_percent": job.get("progress", 0),
            "message": "Detailed status not available"
        })

    return JSONResponse(status)


@app.get("/api/v2/status/{batch_id}/summary", tags=["Status - Detailed"])
def get_status_summary(batch_id: str):
    """
    Get simplified status summary for polling.

    Returns minimal status info optimized for frequent polling:
    - state: Current state (queued, extracting, chunking, indexing, completed, failed)
    - progress_percent: Overall progress (0-100)
    - current_stage: Current processing stage
    - current_pdf: PDF currently being processed
    - message: Human-readable status message

    Use this for lightweight status checks. Use /api/v2/status/{batch_id}
    for full details when user expands the status view.
    """
    summary = get_batch_status_summary(batch_id)
    if not summary:
        job = get_job(batch_id)
        if not job:
            return JSONResponse({"error": "Batch not found"}, status_code=404)
        return JSONResponse({
            "batch_id": batch_id,
            "state": job.get("state", "unknown"),
            "progress_percent": job.get("progress", 0),
            "message": f"Processing... {job.get('processed_pages', 0)}/{job.get('total_pages', 0)} pages"
        })

    return JSONResponse(summary)


@app.get("/api/v2/status/{batch_id}/stages", tags=["Status - Detailed"])
def get_stage_timeline(batch_id: str):
    """
    Get processing stages timeline for visualization.

    Returns ordered list of stages with their status and timing,
    perfect for rendering a progress timeline or stepper UI.

    Stage order: initializing -> extracting -> vision -> chunking -> indexing -> completed
    """
    status = get_batch_status(batch_id)
    if not status:
        return JSONResponse({"error": "Batch not found"}, status_code=404)

    # Define stage order for timeline
    stage_order = [
        ("initializing", "Initializing", "Setting up processing environment"),
        ("extracting", "Extracting", "Extracting text, images, and tables from PDF"),
        ("vision", "Vision Processing", "Analyzing images and tables with AI"),
        ("chunking", "Chunking", "Creating semantic text chunks"),
        ("indexing", "Indexing", "Storing in vector database for search")
    ]

    timeline = []
    for stage_key, stage_name, stage_description in stage_order:
        stage_info = {
            "key": stage_key,
            "name": stage_name,
            "description": stage_description,
            "status": "pending",
            "progress": 0
        }

        # Aggregate status from all PDFs
        all_statuses = []
        for pdf_name, pdf_data in status.get("pdfs", {}).items():
            stages = pdf_data.get("stages", {})
            if stage_key in stages:
                all_statuses.append(stages[stage_key])

        if all_statuses:
            # Determine overall stage status
            statuses = [s.get("status") for s in all_statuses]
            if all(s == "completed" for s in statuses):
                stage_info["status"] = "completed"
                stage_info["progress"] = 100
            elif any(s == "running" for s in statuses):
                stage_info["status"] = "running"
                stage_info["progress"] = sum(s.get("progress", 0) for s in all_statuses) // len(all_statuses)
            elif any(s == "failed" for s in statuses):
                stage_info["status"] = "failed"
            elif any(s == "completed" for s in statuses):
                stage_info["status"] = "partial"
                stage_info["progress"] = sum(s.get("progress", 0) for s in all_statuses) // len(all_statuses)

            # Get timing from first PDF (representative)
            first_stage = all_statuses[0]
            if first_stage.get("started_at"):
                stage_info["started_at"] = first_stage["started_at"]
            if first_stage.get("completed_at"):
                stage_info["completed_at"] = first_stage["completed_at"]
            if first_stage.get("duration_seconds"):
                stage_info["duration_seconds"] = first_stage["duration_seconds"]
            if first_stage.get("message"):
                stage_info["message"] = first_stage["message"]

        timeline.append(stage_info)

    # Add final completed/failed stage
    final_stage = {
        "key": "completed",
        "name": "Completed",
        "description": "Processing finished, ready for Q&A",
        "status": "pending" if status.get("state") != "completed" else "completed",
        "progress": 100 if status.get("state") == "completed" else 0
    }
    if status.get("state") == "failed":
        final_stage["key"] = "failed"
        final_stage["name"] = "Failed"
        final_stage["status"] = "failed"
        final_stage["error"] = status.get("error")

    timeline.append(final_stage)

    return JSONResponse({
        "batch_id": batch_id,
        "current_stage": status.get("current_stage"),
        "overall_progress": status.get("progress_percent", 0),
        "timeline": timeline
    })


# =============================================================================
# WebSocket for Real-time Status Updates
# =============================================================================

@app.websocket("/ws/status/{batch_id}")
async def websocket_status(websocket: WebSocket, batch_id: str):
    """
    WebSocket endpoint for real-time status updates.

    Connect to receive live updates as processing progresses:
    - ws://localhost:8080/ws/status/{batch_id}

    Messages are JSON objects with structure:
    {
        "type": "status_update",
        "batch_id": "abc-123",
        "data": {
            "state": "extracting",
            "progress_percent": 45,
            "current_stage": "extracting",
            "current_pdf": "document.pdf",
            ...
        }
    }

    The connection will send:
    1. Initial status immediately upon connection
    2. Updates whenever processing state changes
    3. Final status when processing completes or fails

    Client Example (JavaScript):
        const ws = new WebSocket(`ws://localhost:8080/ws/status/${batchId}`);
        ws.onmessage = (event) => {
            const update = JSON.parse(event.data);
            updateUI(update.data);
        };
    """
    await ws_manager.connect(websocket, batch_id)

    try:
        # Send initial status immediately
        status = get_batch_status(batch_id)
        if status:
            await websocket.send_json({
                "type": "status_update",
                "batch_id": batch_id,
                "data": status
            })
        else:
            # Fallback to basic job status
            job = get_job(batch_id)
            if job:
                await websocket.send_json({
                    "type": "status_update",
                    "batch_id": batch_id,
                    "data": {
                        "state": job.get("state", "unknown"),
                        "progress_percent": job.get("progress", 0)
                    }
                })
            else:
                await websocket.send_json({
                    "type": "error",
                    "message": "Batch not found"
                })
                await websocket.close()
                return

        # Keep connection alive and wait for client messages or disconnect
        while True:
            try:
                # Wait for ping/pong or client messages
                data = await asyncio.wait_for(
                    websocket.receive_text(),
                    timeout=30.0
                )
                # Handle client messages (e.g., ping)
                if data == "ping":
                    await websocket.send_text("pong")
            except asyncio.TimeoutError:
                # Send keepalive ping
                try:
                    await websocket.send_json({"type": "ping"})
                except Exception:
                    break

    except WebSocketDisconnect:
        pass
    except Exception:
        pass
    finally:
        ws_manager.disconnect(websocket, batch_id)


@app.post("/api/v2/pdf/analyze", tags=["PDFs - Multimodal"])
def analyze_pdfs_multimodal(
    files: List[UploadFile] = File(...),
    use_vision: str = Query("yes", description="Use vision model for image/table descriptions"),
    use_semantic_chunking: str = Query("no", description="Use embedding-based semantic chunking for better topic boundaries"),
    use_structure_chunking: str = Query("no", description="Use structure-based chunking with section hierarchy detection"),
    semantic_similarity_threshold: float = Query(0.5, description="Similarity threshold for semantic chunking (0-1)"),
    semantic_percentile_threshold: float = Query(25, description="Use bottom N percentile of similarities as breakpoints"),
    semantic_min_chunk_size: int = Query(50, description="Minimum words per chunk for semantic chunking"),
    semantic_max_chunk_size: int = Query(500, description="Maximum words per chunk for semantic chunking"),
    preview_only: str = Query("no"),
    preview_pages: int = Query(1)
):
    """
    Upload PDFs for multimodal processing (text, tables, images).

    This endpoint:
    - Extracts content in reading order
    - Saves all tables as images
    - Uses vision models (if configured) for image/table descriptions
    - Creates chunks with image_link and table_link for explainability
    - Optionally uses embedding-based semantic chunking for better topic boundaries
    - Optionally uses structure-based chunking with section hierarchy detection
    - Indexes all content into Milvus

    Semantic Chunking:
    - When enabled, uses sentence embeddings to detect topic shifts
    - Splits text at semantic boundaries rather than fixed word counts
    - Produces more coherent chunks that preserve topical context

    Structure-Based Chunking:
    - Analyzes PDF fonts to detect headings (H1, H2, H3)
    - Builds section hierarchy across pages
    - Adds [Section: ...] prefix to chunks for context
    - Stores section_hierarchy and heading_level in each chunk

    Set OPENAI_API_KEY or ANTHROPIC_API_KEY environment variable for vision features.
    """
    batch_id = str(uuid.uuid4())
    input_dir = Path("uploads") / batch_id
    output_dir = Path("outputs") / batch_id
    input_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    previews = {}

    for file in files:
        pdf_path = input_dir / file.filename
        with open(pdf_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        if preview_only.lower() == "yes":
            import fitz
            doc = fitz.open(pdf_path)
            pages_preview = []
            for i, page in enumerate(doc[:preview_pages]):
                page_result = process_page(page, i, file.filename, output_dir / file.filename)
                pages_preview.append(page_to_markdown(page_result))
            previews[file.filename] = "\n\n".join(pages_preview)

    # Determine chunking mode and enqueue appropriate job
    semantic_enabled = use_semantic_chunking.lower() == "yes"
    structure_enabled = use_structure_chunking.lower() == "yes"

    if structure_enabled:
        # Use structure-based chunking with section hierarchy
        job = queue.enqueue(
            process_pdf_batch_structured,
            batch_id,
            str(input_dir),
            str(output_dir),
            use_vision.lower() == "yes",
            semantic_max_chunk_size,  # max_words
            semantic_min_chunk_size   # min_words
        )
        chunking_mode = "structure"
    else:
        # Use multimodal processing job
        job = queue.enqueue(
            process_pdf_batch_multimodal,
            batch_id,
            str(input_dir),
            str(output_dir),
            use_vision.lower() == "yes",
            semantic_enabled,
            semantic_similarity_threshold,
            semantic_percentile_threshold if semantic_enabled else None,
            semantic_min_chunk_size,
            semantic_max_chunk_size
        )
        chunking_mode = "semantic" if semantic_enabled else "multimodal"

    init_job(batch_id, rq_job_id=job.id)

    # Initialize detailed status tracking
    pdf_names = [Path(f.filename).stem for f in files]
    init_batch_status(batch_id, pdf_names)

    return JSONResponse({
        "batch_id": batch_id,
        "rq_job_id": job.id,
        "uploaded_files": [f.filename for f in files],
        "message": f"Files uploaded. {chunking_mode.title()} background processing started.",
        "mode": chunking_mode,
        "vision_enabled": use_vision.lower() == "yes",
        "semantic_chunking_enabled": semantic_enabled,
        "structure_chunking_enabled": structure_enabled,
        "preview_markdown": previews if preview_only.lower() == "yes" else None,
        "static_base_url": f"/static/{batch_id}",
        "status_url": f"/api/v2/status/{batch_id}",
        "status_summary_url": f"/api/v2/status/{batch_id}/summary",
        "status_timeline_url": f"/api/v2/status/{batch_id}/stages",
        "websocket_url": f"/ws/status/{batch_id}"
    })


@app.get("/api/v2/pdf/chunks/{batch_id}", tags=["PDFs - Multimodal"])
def get_chunks(batch_id: str, content_type: str = Query(None, description="Filter by content type: text, table, image")):
    """
    Get all chunks for a batch with optional content type filtering.
    """
    import json

    output_dir = Path("outputs") / batch_id
    if not output_dir.exists():
        return JSONResponse({"error": "Batch not found"}, status_code=404)

    all_chunks = []

    for pdf_dir in output_dir.iterdir():
        if pdf_dir.is_dir():
            chunks_file = pdf_dir / "chunks.json"
            if chunks_file.exists():
                with open(chunks_file, "r", encoding="utf-8") as f:
                    chunks = json.load(f)
                    # Add static URLs for links
                    for chunk in chunks:
                        if chunk.get("image_link"):
                            chunk["image_url"] = f"/static/{batch_id}/{pdf_dir.name}/images/{Path(chunk['image_link']).name}"
                        if chunk.get("table_link"):
                            chunk["table_url"] = f"/static/{batch_id}/{pdf_dir.name}/tables/{Path(chunk['table_link']).name}"
                    all_chunks.extend(chunks)

    # Filter by content type if specified
    if content_type:
        all_chunks = [c for c in all_chunks if c.get("content_type") == content_type]

    return JSONResponse({
        "batch_id": batch_id,
        "total_chunks": len(all_chunks),
        "chunks": all_chunks
    })


@app.get("/api/v2/pdf/summary/{batch_id}", tags=["PDFs - Multimodal"])
def get_batch_summary(batch_id: str):
    """
    Get batch processing summary including content type breakdown.
    """
    import json

    output_dir = Path("outputs") / batch_id
    summary_file = output_dir / "batch_summary.json"

    if not summary_file.exists():
        return JSONResponse({"error": "Batch summary not found"}, status_code=404)

    with open(summary_file, "r", encoding="utf-8") as f:
        summary = json.load(f)

    return JSONResponse(summary)


# =============================================================================
# Q&A Endpoints
# =============================================================================

def _get_qa_service(llm_provider: Optional[str] = None, llm_model: Optional[str] = None) -> QAService:
    """Helper to create QA service with specified or default LLM."""
    embedder = EmbeddingClient(EMBEDDING_SERVICE_URL)
    vector_store = MilvusVectorStore(host=MILVUS_HOST, port=MILVUS_PORT)
    retriever = Retriever(vector_store, embedder)

    if llm_provider:
        generator = create_generator(provider=llm_provider, model=llm_model)
    else:
        generator = get_default_generator()

    return QAService(retriever, generator)


@app.post("/api/v2/qa/ask/{batch_id}", tags=["Q&A"])
def ask_question(batch_id: str, request: QuestionRequest):
    """
    Ask a question about the ingested PDFs.

    The system will:
    1. Search for relevant chunks in Milvus using semantic similarity
    2. Pass the retrieved context to an LLM
    3. Generate an answer with citations [Source: PDF_NAME, Page X]

    LLM Providers:
    - ollama: Local inference (default, requires Ollama running)
    - openai: OpenAI API (requires OPENAI_API_KEY)
    - anthropic: Anthropic API (requires ANTHROPIC_API_KEY)
    - gemini: Google Gemini API (requires GOOGLE_API_KEY)
    """
    # Verify batch exists
    job = get_job(batch_id)
    if not job:
        return JSONResponse({"error": "Batch not found"}, status_code=404)

    if job.get("state") != "completed":
        return JSONResponse({
            "error": "Batch processing not complete",
            "status": job.get("state")
        }, status_code=400)

    try:
        qa_service = _get_qa_service(request.llm_provider, request.llm_model)
        result = qa_service.ask(
            session_id=batch_id,
            question=request.question,
            top_k=request.top_k,
            temperature=request.temperature,
            include_sources=request.include_sources
        )
        return JSONResponse({
            "batch_id": batch_id,
            "question": request.question,
            **result
        })
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.post("/api/v2/qa/summarize/{batch_id}", tags=["Q&A"])
def summarize_documents(batch_id: str, request: SummaryRequest):
    """
    Generate a summary of the ingested PDFs.

    Summary Types:
    - brief: 2-3 paragraph overview
    - detailed: Comprehensive coverage of all topics
    - bullets: Key points as bullet list
    """
    job = get_job(batch_id)
    if not job:
        return JSONResponse({"error": "Batch not found"}, status_code=404)

    if job.get("state") != "completed":
        return JSONResponse({
            "error": "Batch processing not complete",
            "status": job.get("state")
        }, status_code=400)

    try:
        qa_service = _get_qa_service(request.llm_provider, request.llm_model)
        result = qa_service.summarize(
            session_id=batch_id,
            summary_type=request.summary_type,
            max_chunks=request.max_chunks,
            temperature=request.temperature
        )
        return JSONResponse({
            "batch_id": batch_id,
            **result
        })
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.post("/api/v2/qa/summarize-advanced/{batch_id}", tags=["Q&A - Advanced"])
def advanced_summarize(batch_id: str, request: AdvancedSummaryRequest):
    """
    Generate summaries with different scopes and strategies.

    **Summary Scopes:**

    1. **topic** - Section/Topic-wise Summary
       - Clusters chunks by semantic similarity
       - Identifies distinct topics automatically
       - Generates summary for each topic with title
       - Best for: Understanding document structure

    2. **document** - Per-Document Summary
       - Generates separate summary for each PDF
       - Uses Map-Reduce strategy for full coverage
       - Best for: Comparing multiple documents

    3. **all** - Combined Summary
       - Single summary covering all documents
       - Hierarchical Map-Reduce strategy
       - Best for: Overall understanding of content

    **Summary Formats:**
    - brief: Concise 2-3 paragraphs
    - detailed: Comprehensive coverage
    - bullets: Organized bullet points
    """
    job = get_job(batch_id)
    if not job:
        return JSONResponse({"error": "Batch not found"}, status_code=404)

    if job.get("state") != "completed":
        return JSONResponse({
            "error": "Batch processing not complete",
            "status": job.get("state")
        }, status_code=400)

    try:
        qa_service = _get_qa_service(request.llm_provider, request.llm_model)
        result = qa_service.advanced_summarize(
            session_id=batch_id,
            scope=request.scope,
            summary_format=request.summary_format,
            num_topics=request.num_topics,
            max_chunks=request.max_chunks,
            temperature=request.temperature
        )
        return JSONResponse({
            "batch_id": batch_id,
            **result
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse({"error": str(e)}, status_code=500)


@app.post("/api/v2/qa/chat/{batch_id}", tags=["Q&A"])
def chat_with_documents(batch_id: str, request: ChatRequest):
    """
    Multi-turn conversation with the ingested PDFs.

    Send conversation history as messages array:
    [
        {"role": "user", "content": "What is this document about?"},
        {"role": "assistant", "content": "This document discusses..."},
        {"role": "user", "content": "Tell me more about section 2"}
    ]
    """
    job = get_job(batch_id)
    if not job:
        return JSONResponse({"error": "Batch not found"}, status_code=404)

    if job.get("state") != "completed":
        return JSONResponse({
            "error": "Batch processing not complete",
            "status": job.get("state")
        }, status_code=400)

    try:
        qa_service = _get_qa_service(request.llm_provider, request.llm_model)
        messages = [{"role": m.role, "content": m.content} for m in request.messages]
        result = qa_service.chat(
            session_id=batch_id,
            messages=messages,
            top_k=request.top_k,
            temperature=request.temperature
        )
        return JSONResponse({
            "batch_id": batch_id,
            **result
        })
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/api/v2/qa/search/{batch_id}", tags=["Q&A"])
def search_chunks(
    batch_id: str,
    query: str = Query(..., description="Search query"),
    top_k: int = Query(5, description="Number of results"),
    content_type: str = Query(None, description="Filter by content type")
):
    """
    Search for relevant chunks without LLM generation.

    Useful for:
    - Finding specific content in documents
    - Debugging retrieval quality
    - Building custom Q&A flows
    """
    job = get_job(batch_id)
    if not job:
        return JSONResponse({"error": "Batch not found"}, status_code=404)

    if job.get("state") != "completed":
        return JSONResponse({
            "error": "Batch processing not complete",
            "status": job.get("state")
        }, status_code=400)

    try:
        embedder = EmbeddingClient(EMBEDDING_SERVICE_URL)
        vector_store = MilvusVectorStore(host=MILVUS_HOST, port=MILVUS_PORT)
        retriever = Retriever(vector_store, embedder)

        chunks = retriever.search(
            session_id=batch_id,
            query=query,
            top_k=top_k,
            content_type_filter=content_type
        )

        return JSONResponse({
            "batch_id": batch_id,
            "query": query,
            "total_results": len(chunks),
            "results": [
                {
                    "pdf_name": c["pdf_name"],
                    "page_no": c["page_no"] + 1,
                    "content_type": c["content_type"],
                    "text": c["text"],
                    "score": round(c["score"], 4),
                    "image_link": c.get("image_link", ""),
                    "table_link": c.get("table_link", "")
                }
                for c in chunks
            ]
        })
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


# =============================================================================
# Summary Generation Endpoints (On-Demand)
# =============================================================================

@app.post("/api/v2/summary/document/{batch_id}/{pdf_name}", tags=["Summary"])
def generate_document_summary(
    batch_id: str,
    pdf_name: str,
    request: DocumentSummaryRequest = None
):
    """
    Generate a summary for a specific document on demand.

    Uses Gemma3 model to analyze all chunks from the document and generate
    a comprehensive summary.

    Summary Types:
    - brief: 3-5 sentence summary
    - detailed: Full structured summary with sections, findings, takeaways
    - bullets: Key points as bullet list

    Note: This is generated on-demand and may take 1-2 minutes for large documents.
    """
    if request is None:
        request = DocumentSummaryRequest()

    # Verify batch exists
    job = get_job(batch_id)
    if not job:
        return JSONResponse({"error": "Batch not found"}, status_code=404)

    if job.get("state") != "completed":
        return JSONResponse({
            "error": "Batch processing not complete",
            "status": job.get("state")
        }, status_code=400)

    try:
        # Get chunks for this document from Milvus
        vector_store = MilvusVectorStore(host=MILVUS_HOST, port=MILVUS_PORT)

        # Search for all chunks from this PDF
        # Use a broad query to get all chunks
        embedder = EmbeddingClient(EMBEDDING_SERVICE_URL)
        query_embedding = embedder.embed_texts(["document content summary overview"])[0]

        all_chunks = vector_store.search(
            session_id=batch_id,
            query_embedding=query_embedding,
            top_k=1000  # Get all chunks
        )

        # Filter chunks for this specific PDF
        doc_chunks = [c for c in all_chunks if c.get("pdf_name") == pdf_name]

        if not doc_chunks:
            return JSONResponse({
                "error": f"No chunks found for document: {pdf_name}",
                "batch_id": batch_id
            }, status_code=404)

        # Generate summary
        summary_service = get_summary_service()
        result = summary_service.generate_document_summary(
            chunks=doc_chunks,
            pdf_name=pdf_name,
            summary_type=request.summary_type
        )

        return JSONResponse({
            "batch_id": batch_id,
            **result
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse({"error": str(e)}, status_code=500)


@app.post("/api/v2/summary/corpus/{batch_id}", tags=["Summary"])
def generate_corpus_summary(
    batch_id: str,
    request: CorpusSummaryRequest = None
):
    """
    Generate a summary across all documents in a batch.

    First generates individual document summaries, then creates a corpus-wide
    summary that identifies common themes, patterns, and insights across all
    documents.

    Note: This is generated on-demand and may take several minutes for
    batches with many documents.
    """
    if request is None:
        request = CorpusSummaryRequest()

    # Verify batch exists
    job = get_job(batch_id)
    if not job:
        return JSONResponse({"error": "Batch not found"}, status_code=404)

    if job.get("state") != "completed":
        return JSONResponse({
            "error": "Batch processing not complete",
            "status": job.get("state")
        }, status_code=400)

    try:
        # Get all chunks from Milvus
        vector_store = MilvusVectorStore(host=MILVUS_HOST, port=MILVUS_PORT)
        embedder = EmbeddingClient(EMBEDDING_SERVICE_URL)
        query_embedding = embedder.embed_texts(["document content summary overview"])[0]

        all_chunks = vector_store.search(
            session_id=batch_id,
            query_embedding=query_embedding,
            top_k=5000  # Get all chunks
        )

        if not all_chunks:
            return JSONResponse({
                "error": "No chunks found for this batch",
                "batch_id": batch_id
            }, status_code=404)

        # Group chunks by PDF name
        chunks_by_pdf = {}
        for chunk in all_chunks:
            pdf_name = chunk.get("pdf_name", "Unknown")
            if pdf_name not in chunks_by_pdf:
                chunks_by_pdf[pdf_name] = []
            chunks_by_pdf[pdf_name].append(chunk)

        # Generate individual document summaries
        summary_service = get_summary_service()
        document_summaries = []

        for pdf_name, pdf_chunks in chunks_by_pdf.items():
            print(f"Generating summary for {pdf_name}...")
            doc_summary = summary_service.generate_document_summary(
                chunks=pdf_chunks,
                pdf_name=pdf_name,
                summary_type=request.summary_type
            )
            document_summaries.append(doc_summary)

        # Generate corpus summary
        corpus_result = summary_service.generate_corpus_summary(
            document_summaries=document_summaries,
            batch_id=batch_id
        )

        response = {
            "batch_id": batch_id,
            **corpus_result
        }

        # Include individual summaries if requested
        if request.include_individual:
            response["document_summaries"] = document_summaries

        return JSONResponse(response)

    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/api/v2/summary/documents/{batch_id}", tags=["Summary"])
def list_documents_for_summary(batch_id: str):
    """
    List all documents available for summary generation in a batch.
    """
    # Verify batch exists
    job = get_job(batch_id)
    if not job:
        return JSONResponse({"error": "Batch not found"}, status_code=404)

    try:
        # Get all chunks from Milvus
        vector_store = MilvusVectorStore(host=MILVUS_HOST, port=MILVUS_PORT)
        embedder = EmbeddingClient(EMBEDDING_SERVICE_URL)
        query_embedding = embedder.embed_texts(["document"])[0]

        all_chunks = vector_store.search(
            session_id=batch_id,
            query_embedding=query_embedding,
            top_k=5000
        )

        # Group by PDF name and count
        pdf_stats = {}
        for chunk in all_chunks:
            pdf_name = chunk.get("pdf_name", "Unknown")
            if pdf_name not in pdf_stats:
                pdf_stats[pdf_name] = {
                    "chunk_count": 0,
                    "content_types": {},
                    "has_table_summary": False,
                    "has_image_summary": False
                }
            pdf_stats[pdf_name]["chunk_count"] += 1

            ct = chunk.get("content_type", "text")
            pdf_stats[pdf_name]["content_types"][ct] = \
                pdf_stats[pdf_name]["content_types"].get(ct, 0) + 1

            if chunk.get("table_summary"):
                pdf_stats[pdf_name]["has_table_summary"] = True
            if chunk.get("image_summary"):
                pdf_stats[pdf_name]["has_image_summary"] = True

        return JSONResponse({
            "batch_id": batch_id,
            "document_count": len(pdf_stats),
            "documents": [
                {
                    "pdf_name": name,
                    **stats
                }
                for name, stats in pdf_stats.items()
            ]
        })

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/", tags=["UI"])
def root():
    """Redirect to UI."""
    from fastapi.responses import RedirectResponse
    return RedirectResponse(url="/ui/index.html")
