from typing import List, Optional, Dict, Any
from enum import Enum
import uuid
import shutil

from fastapi import FastAPI, UploadFile, WebSocket, WebSocketDisconnect, Query, HTTPException, status
from fastapi.openapi.utils import get_openapi
from pydantic import BaseModel, Field

# ---- logging ----
from doc_analysis.logging_config import get_api_logger, setup_all_loggers
setup_all_loggers()
logger = get_api_logger()


# ============================================================================
# API METADATA AND TAGS
# ============================================================================

API_TITLE = "Document Analysis API"
API_VERSION = "1.0.0"
API_DESCRIPTION = """
## Overview

The **Document Analysis API** is a comprehensive PDF processing and summarization system that enables:

- 📄 **PDF Upload & Processing**: Upload multiple PDFs for extraction, chunking, and indexing
- 🔍 **Semantic Search**: Store document chunks in Milvus vector database for similarity search
- 📝 **AI-Powered Summarization**: Generate brief, detailed, or bullet-point summaries
- 🔄 **Summary Refinement**: Iteratively refine summaries with user feedback
- 📡 **Real-time Updates**: WebSocket connections for live processing status

## Architecture

```
PDF Upload → Extract → Chunk → Embed (E5-large) → Index (Milvus)
                                    ↓
                            Vision Processing (Ollama)
                                    ↓
                            Summarization (LLM)
```

## Key Features

- **Batch Processing**: Upload up to 5 PDFs per batch
- **Page Range Selection**: Process specific pages only
- **Hierarchical Summarization**: Map-reduce pattern for large documents
- **Hybrid Storage**: Redis for caching, Milvus for persistence
- **Real-time Events**: WebSocket notifications for all processing stages

## Authentication

Currently, no authentication is required. All endpoints are publicly accessible.

## Rate Limits

- Maximum 5 PDFs per upload batch
- 30-minute timeout per processing job
- 24-hour data retention (configurable)
"""

tags_metadata = [
    {
        "name": "PDF Processing",
        "description": "Upload and process PDF documents. Handles extraction, chunking, embedding, and indexing.",
    },
    {
        "name": "Job Status",
        "description": "Monitor processing job status and progress. Supports filtering by module.",
    },
    {
        "name": "Summarization",
        "description": "Generate AI-powered summaries for individual PDFs or entire batches.",
    },
    {
        "name": "Summary Refinement",
        "description": "Iteratively refine summaries using user feedback with optional context retrieval.",
    },
    {
        "name": "WebSocket",
        "description": "Real-time event streams for PDF processing and summarization updates.",
    },
    {
        "name": "System",
        "description": "System health, statistics, and monitoring endpoints.",
    },
]


# ============================================================================
# RESPONSE MODELS
# ============================================================================

class PageRange(BaseModel):
    """Page range specification for PDF processing."""
    start: Optional[int] = Field(None, description="Start page (1-indexed, inclusive)")
    end: Optional[int] = Field(None, description="End page (1-indexed, inclusive)")

    class Config:
        json_schema_extra = {
            "example": {"start": 1, "end": 10}
        }


class UploadResponse(BaseModel):
    """Response returned after successful PDF upload."""
    batch_id: str = Field(..., description="Unique identifier for the batch of uploaded PDFs")
    pdf_count: int = Field(..., description="Number of PDFs uploaded in this batch")
    status: str = Field(..., description="Current processing status (queued, processing, completed, failed)")
    page_range: Optional[PageRange] = Field(None, description="Page range filter if specified")

    class Config:
        json_schema_extra = {
            "example": {
                "batch_id": "550e8400-e29b-41d4-a716-446655440000",
                "pdf_count": 2,
                "status": "queued",
                "page_range": {"start": 1, "end": 50}
            }
        }


class UploadErrorResponse(BaseModel):
    """Error response when upload fails."""
    error: str = Field(..., description="Error message describing the failure reason")

    class Config:
        json_schema_extra = {
            "example": {"error": "Maximum 5 PDFs allowed per batch"}
        }


class DocumentProcessingStatus(BaseModel):
    """Status details for document processing module."""
    stage: str = Field(..., description="Current processing stage")
    progress: float = Field(..., description="Progress percentage (0-100)")
    pdfs: Dict[str, Any] = Field(..., description="Per-PDF processing status")
    stats: Dict[str, Any] = Field(..., description="Processing statistics")
    error: Optional[str] = Field(None, description="Error message if failed")

    class Config:
        json_schema_extra = {
            "example": {
                "stage": "embedding",
                "progress": 75.0,
                "pdfs": {
                    "report.pdf": {"status": "completed", "pages": 25, "chunks": 48}
                },
                "stats": {
                    "total_pages": 25,
                    "total_chunks": 48,
                    "total_tables": 5,
                    "total_images": 12
                },
                "error": None
            }
        }


class SummarizationStatus(BaseModel):
    """Status details for summarization module."""
    stage: str = Field(..., description="Current summarization stage")
    progress: float = Field(..., description="Progress percentage (0-100)")
    summaries: Dict[str, Any] = Field(..., description="Per-summary status")
    error: Optional[str] = Field(None, description="Error message if failed")

    class Config:
        json_schema_extra = {
            "example": {
                "stage": "batch_summarizing",
                "progress": 50.0,
                "summaries": {
                    "report.pdf:brief": {"status": "completed", "cached": True}
                },
                "error": None
            }
        }


class JobStatusResponse(BaseModel):
    """Complete job status response."""
    batch_id: str = Field(..., description="Batch identifier")
    status: str = Field(..., description="Overall job status")
    created_at: Optional[str] = Field(None, description="Job creation timestamp (ISO 8601)")
    updated_at: Optional[str] = Field(None, description="Last update timestamp (ISO 8601)")
    document_processing: Optional[DocumentProcessingStatus] = Field(None, description="Document processing module status")
    summarization: Optional[SummarizationStatus] = Field(None, description="Summarization module status")

    class Config:
        json_schema_extra = {
            "example": {
                "batch_id": "550e8400-e29b-41d4-a716-446655440000",
                "status": "processing",
                "created_at": "2024-01-15T10:30:00Z",
                "updated_at": "2024-01-15T10:35:00Z",
                "document_processing": {
                    "stage": "completed",
                    "progress": 100.0,
                    "pdfs": {},
                    "stats": {}
                },
                "summarization": {
                    "stage": "pending",
                    "progress": 0.0,
                    "summaries": {}
                }
            }
        }


class PDFListResponse(BaseModel):
    """Response containing list of PDFs in a batch."""
    batch_id: str = Field(..., description="Batch identifier")
    pdf_names: List[str] = Field(..., description="List of PDF filenames in the batch")
    count: int = Field(..., description="Total number of PDFs")

    class Config:
        json_schema_extra = {
            "example": {
                "batch_id": "550e8400-e29b-41d4-a716-446655440000",
                "pdf_names": ["report.pdf", "analysis.pdf", "summary.pdf"],
                "count": 3
            }
        }


class SummaryResponse(BaseModel):
    """Response containing a generated summary."""
    batch_id: str = Field(..., description="Batch identifier")
    pdf_name: str = Field(..., description="Name of the summarized PDF")
    summary_type: str = Field(..., description="Type of summary generated")
    summary: str = Field(..., description="The generated summary text")
    method: str = Field(..., description="Summarization method used (direct, hierarchical)")
    chunk_count: int = Field(..., description="Number of chunks processed")
    cached: bool = Field(..., description="Whether the summary was retrieved from cache")
    word_count: Optional[int] = Field(None, description="Word count of the summary")

    class Config:
        json_schema_extra = {
            "example": {
                "batch_id": "550e8400-e29b-41d4-a716-446655440000",
                "pdf_name": "report.pdf",
                "summary_type": "brief",
                "summary": "This document presents a comprehensive analysis of market trends...",
                "method": "hierarchical",
                "chunk_count": 48,
                "cached": False,
                "word_count": 150
            }
        }


class BatchSummaryResponse(BaseModel):
    """Response for batch summary of all PDFs."""
    batch_id: str = Field(..., description="Batch identifier")
    summary_type: str = Field(..., description="Type of summary generated")
    total_pdfs: int = Field(..., description="Number of PDFs summarized")
    summaries: Dict[str, SummaryResponse] = Field(..., description="Individual PDF summaries")

    class Config:
        json_schema_extra = {
            "example": {
                "batch_id": "550e8400-e29b-41d4-a716-446655440000",
                "summary_type": "brief",
                "total_pdfs": 2,
                "summaries": {
                    "report.pdf": {"summary": "...", "cached": False},
                    "analysis.pdf": {"summary": "...", "cached": True}
                }
            }
        }


class RefinedSummaryResponse(BaseModel):
    """Response containing a refined summary."""
    batch_id: str = Field(..., description="Batch identifier")
    pdf_name: str = Field(..., description="Name of the PDF")
    refined_summary: str = Field(..., description="The refined summary text")
    refinement_type: str = Field(..., description="Type of refinement applied (simple, contextual)")
    user_id: Optional[str] = Field(None, description="User identifier if provided")

    class Config:
        json_schema_extra = {
            "example": {
                "batch_id": "550e8400-e29b-41d4-a716-446655440000",
                "pdf_name": "report.pdf",
                "refined_summary": "This concise document analysis reveals...",
                "refinement_type": "simple",
                "user_id": "user123"
            }
        }


class ContextualRefinedSummaryResponse(RefinedSummaryResponse):
    """Response for contextual refinement with chunk details."""
    chunks_used: int = Field(..., description="Number of source chunks used for context")
    chunks_detail: List[Dict[str, Any]] = Field(..., description="Details of chunks used")

    class Config:
        json_schema_extra = {
            "example": {
                "batch_id": "550e8400-e29b-41d4-a716-446655440000",
                "pdf_name": "report.pdf",
                "refined_summary": "Based on the financial data, the Q3 projections show...",
                "refinement_type": "contextual",
                "chunks_used": 5,
                "chunks_detail": [
                    {"page": 12, "type": "text", "relevance": 0.92, "preview": "Financial projections..."},
                    {"page": 15, "type": "table", "relevance": 0.88, "preview": "Q3 Revenue table..."}
                ],
                "user_id": "user123"
            }
        }


class WebSocketStatsResponse(BaseModel):
    """WebSocket connection statistics."""
    pdf_connections: int = Field(..., description="Active PDF update connections")
    summary_connections: int = Field(..., description="Active summary update connections")
    total_connections: int = Field(..., description="Total active WebSocket connections")
    batches_with_connections: List[str] = Field(..., description="Batch IDs with active connections")

    class Config:
        json_schema_extra = {
            "example": {
                "pdf_connections": 5,
                "summary_connections": 3,
                "total_connections": 8,
                "batches_with_connections": ["batch-1", "batch-2"]
            }
        }


class ErrorResponse(BaseModel):
    """Standard error response."""
    detail: str = Field(..., description="Error description")

    class Config:
        json_schema_extra = {
            "example": {"detail": "No PDFs found for batch: invalid-batch-id"}
        }


# ============================================================================
# FASTAPI APP INITIALIZATION
# ============================================================================

app = FastAPI(
    title=API_TITLE,
    version=API_VERSION,
    description=API_DESCRIPTION,
    openapi_tags=tags_metadata,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

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


# ============================================================================
# SYSTEM ENDPOINTS
# ============================================================================

class HealthResponse(BaseModel):
    """Health check response."""
    status: str = Field(..., description="Service health status")
    version: str = Field(..., description="API version")
    services: Dict[str, str] = Field(..., description="Status of dependent services")

    class Config:
        json_schema_extra = {
            "example": {
                "status": "healthy",
                "version": "1.0.0",
                "services": {
                    "redis": "connected",
                    "milvus": "connected",
                    "ollama": "available"
                }
            }
        }


class APIInfoResponse(BaseModel):
    """API information response."""
    name: str = Field(..., description="API name")
    version: str = Field(..., description="API version")
    description: str = Field(..., description="Brief API description")
    docs_url: str = Field(..., description="URL to Swagger UI documentation")
    redoc_url: str = Field(..., description="URL to ReDoc documentation")

    class Config:
        json_schema_extra = {
            "example": {
                "name": "Document Analysis API",
                "version": "1.0.0",
                "description": "PDF processing and AI-powered summarization system",
                "docs_url": "/docs",
                "redoc_url": "/redoc"
            }
        }


@app.get(
    "/",
    response_model=APIInfoResponse,
    tags=["System"],
    summary="API information",
    description="Get basic information about the API and links to documentation.",
)
def root():
    """Return API information and documentation links."""
    return {
        "name": API_TITLE,
        "version": API_VERSION,
        "description": "PDF processing and AI-powered summarization system",
        "docs_url": "/docs",
        "redoc_url": "/redoc"
    }


@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["System"],
    summary="Health check",
    description="""
Check the health status of the API and its dependent services.

## Services Checked

| Service | Description |
|---------|-------------|
| **Redis** | Job queue, caching, and pub/sub messaging |
| **Milvus** | Vector database for document chunks |
| **Ollama** | Vision and language models |

## Response Status Values

- `healthy`: All services operational
- `degraded`: Some services unavailable but core functionality works
- `unhealthy`: Critical services down

## Use Cases

- **Load balancer health probes**: Configure your load balancer to check `/health`
- **Kubernetes liveness/readiness probes**: Use for container orchestration
- **Monitoring dashboards**: Track service availability over time
""",
)
def health_check():
    """Check API and dependent services health."""
    import redis

    services = {}
    overall_healthy = True

    # Check Redis
    try:
        r = redis.Redis(host='localhost', port=6379, socket_timeout=2)
        r.ping()
        services["redis"] = "connected"
    except Exception:
        services["redis"] = "disconnected"
        overall_healthy = False

    # Check Milvus (basic connectivity)
    try:
        from pymilvus import connections
        connections.connect(alias="health_check", host="localhost", port="19530", timeout=2)
        connections.disconnect(alias="health_check")
        services["milvus"] = "connected"
    except Exception:
        services["milvus"] = "disconnected"
        overall_healthy = False

    # Check Ollama
    try:
        import requests
        resp = requests.get("http://localhost:11434/api/tags", timeout=2)
        if resp.status_code == 200:
            services["ollama"] = "available"
        else:
            services["ollama"] = "unavailable"
    except Exception:
        services["ollama"] = "unavailable"
        # Ollama is optional, don't mark as unhealthy

    return {
        "status": "healthy" if overall_healthy else "degraded",
        "version": API_VERSION,
        "services": services
    }


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


@app.post(
    "/upload-pdfs",
    response_model=UploadResponse,
    responses={
        200: {
            "description": "PDFs successfully uploaded and queued for processing",
            "model": UploadResponse
        },
        400: {
            "description": "Upload validation failed (too many files)",
            "model": UploadErrorResponse
        },
        422: {
            "description": "Validation error in request parameters"
        }
    },
    tags=["PDF Processing"],
    summary="Upload PDF documents for processing",
    description="""
Upload one or more PDF documents for extraction, chunking, embedding, and indexing.

## Processing Pipeline

1. **Extraction**: Text, tables, and images are extracted from each page
2. **Chunking**: Content is semantically chunked with overlap for context preservation
3. **Vision Processing**: Tables and images are analyzed using vision models
4. **Embedding**: E5-large model generates 1024-dimensional embeddings
5. **Indexing**: Chunks are stored in Milvus vector database

## Limits

- Maximum **5 PDFs** per batch
- Processing timeout: **30 minutes**
- Data retention: **24 hours** (configurable)

## Page Range Filtering

Optionally specify `start_page` and `end_page` to process only a subset of pages.
Pages are 1-indexed and inclusive on both ends.

## Real-time Updates

Connect to WebSocket `/ws/pdf/{batch_id}` to receive real-time processing updates.
""",
)
def upload_pdfs(
    files: List[UploadFile] = Field(..., description="PDF files to upload (max 5)"),
    start_page: Optional[int] = Query(
        None,
        description="Start page (1-indexed, inclusive). Process from this page onwards.",
        ge=1,
        example=1
    ),
    end_page: Optional[int] = Query(
        None,
        description="End page (1-indexed, inclusive). Process up to and including this page.",
        ge=1,
        example=50
    )
):
    """
    Upload PDF documents for processing.

    Returns a batch_id that can be used to track processing status
    and retrieve results.
    """
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

@app.get(
    "/job-status/{batch_id}",
    response_model=JobStatusResponse,
    responses={
        200: {
            "description": "Job status retrieved successfully",
            "model": JobStatusResponse
        },
        404: {
            "description": "Batch not found",
            "model": ErrorResponse
        }
    },
    tags=["Job Status"],
    summary="Get processing job status",
    description="""
Retrieve the current status of a document processing job.

## Modules

The system tracks two independent processing modules:

### 1. Document Processing (`document_processing`)
Tracks PDF extraction, chunking, and indexing with stages:
- `queued` → `extraction` → `chunking` → `vision_tables` → `vision_images` → `embedding` → `indexing` → `completed`

### 2. Summarization (`summarization`)
Tracks summary generation with stages:
- `pending` → `fetching_chunks` → `batch_summarizing` → `combining` → `storing` → `completed`

## Filtering

Use the `module` query parameter to retrieve status for a specific module only,
reducing response payload size.

## Progress Tracking

Each module includes:
- **stage**: Current processing stage
- **progress**: Percentage completion (0-100)
- **stats**: Detailed statistics (pages, chunks, tables, images)
- **error**: Error message if processing failed
""",
)
def job_status(
    batch_id: str = Field(
        ...,
        description="Unique batch identifier returned from /upload-pdfs",
        example="550e8400-e29b-41d4-a716-446655440000"
    ),
    module: Optional[str] = Query(
        None,
        description="Filter by module: 'document_processing', 'summarization', or None for complete status",
        enum=["document_processing", "summarization"],
        example="document_processing"
    )
):
    """
    Get job status with optional module filter.

    Returns processing progress, statistics, and any errors encountered.
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
    """
    **[DEPRECATED]** Legacy WebSocket endpoint for job status.

    Use `/ws/pdf/{batch_id}` instead for PDF processing updates.

    This endpoint is maintained for backward compatibility and internally
    maps to the PDF updates channel.
    """
    await ws_manager.connect_pdf(batch_id, websocket)

    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        await ws_manager.disconnect_pdf(batch_id, websocket)


@app.websocket("/ws/pdf/{batch_id}")
async def pdf_status_ws(websocket: WebSocket, batch_id: str):
    """
    **WebSocket endpoint for real-time PDF processing updates.**

    Connect to this endpoint to receive live updates during PDF processing.

    ## Connection

    ```javascript
    const ws = new WebSocket('ws://localhost:8000/ws/pdf/{batch_id}');
    ws.onmessage = (event) => {
        const data = JSON.parse(event.data);
        console.log('Event:', data.event, 'Data:', data);
    };
    ```

    ## Events

    | Event | Description | Data Fields |
    |-------|-------------|-------------|
    | `pdf.started` | Processing started | `batch_id`, `pdf_count` |
    | `pdf.extracting` | Extracting content from pages | `pdf_name`, `page`, `total_pages` |
    | `pdf.extracted` | Extraction complete | `pdf_name`, `pages`, `tables`, `images` |
    | `pdf.chunking` | Creating semantic chunks | `pdf_name`, `progress` |
    | `pdf.chunked` | Chunking complete | `pdf_name`, `chunk_count` |
    | `pdf.vision_processing` | Processing tables/images | `pdf_name`, `type`, `current`, `total` |
    | `pdf.embedding` | Generating embeddings | `pdf_name`, `progress` |
    | `pdf.embedded` | Embedding complete | `pdf_name`, `chunk_count` |
    | `pdf.completed` | All processing done | `batch_id`, `total_chunks`, `duration_ms` |
    | `pdf.failed` | Error occurred | `batch_id`, `error`, `pdf_name` |

    ## Example Event Payload

    ```json
    {
        "event": "pdf.extracting",
        "timestamp": "2024-01-15T10:30:00Z",
        "batch_id": "550e8400-e29b-41d4-a716-446655440000",
        "pdf_name": "report.pdf",
        "page": 5,
        "total_pages": 25
    }
    ```
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
    **WebSocket endpoint for real-time summarization updates.**

    Connect to this endpoint to receive live updates during summary generation.

    ## Connection

    ```javascript
    const ws = new WebSocket('ws://localhost:8000/ws/summary/{batch_id}');
    ws.onmessage = (event) => {
        const data = JSON.parse(event.data);
        console.log('Event:', data.event, 'Data:', data);
    };
    ```

    ## Events

    | Event | Description | Data Fields |
    |-------|-------------|-------------|
    | `summary.started` | Summarization started | `pdf_name`, `summary_type`, `chunk_count` |
    | `summary.cache_hit` | Using cached summary | `pdf_name`, `summary_type` |
    | `summary.method_selected` | Method chosen | `method` (direct/hierarchical), `reason` |
    | `summary.batch_started` | Processing batch N/M | `batch_num`, `total_batches`, `word_count` |
    | `summary.batch_completed` | Batch completed | `batch_num`, `total_batches` |
    | `summary.reduce_started` | Combining summaries | `total_summaries` |
    | `summary.reduce_level` | Reduce level progress | `level`, `total_levels`, `summaries_at_level` |
    | `summary.llm_call_started` | LLM API call started | `purpose`, `input_tokens` |
    | `summary.llm_call_completed` | LLM API call done | `purpose`, `output_tokens`, `duration_ms` |
    | `summary.completed` | Summary ready | `pdf_name`, `summary_type`, `word_count` |
    | `summary.failed` | Error occurred | `error`, `pdf_name` |

    ## Multi-PDF Summary Events

    | Event | Description | Data Fields |
    |-------|-------------|-------------|
    | `multi_pdf.started` | Batch summary started | `batch_id`, `pdf_count` |
    | `multi_pdf.pdf_started` | Single PDF processing | `pdf_name`, `index`, `total` |
    | `multi_pdf.pdf_completed` | Single PDF done | `pdf_name`, `index`, `total` |
    | `multi_pdf.combining` | Combining all summaries | `pdf_count` |
    | `multi_pdf.completed` | All done | `batch_id`, `total_word_count` |

    ## Example Event Payload

    ```json
    {
        "event": "summary.batch_started",
        "timestamp": "2024-01-15T10:35:00Z",
        "batch_id": "550e8400-e29b-41d4-a716-446655440000",
        "pdf_name": "report.pdf",
        "batch_num": 2,
        "total_batches": 5,
        "word_count": 3000
    }
    ```
    """
    await ws_manager.connect_summary(batch_id, websocket)

    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        await ws_manager.disconnect_summary(batch_id, websocket)


@app.get(
    "/ws/stats",
    response_model=WebSocketStatsResponse,
    tags=["System"],
    summary="Get WebSocket connection statistics",
    description="""
Retrieve statistics about active WebSocket connections.

Useful for monitoring and debugging real-time update connections.

## Response Fields

- **pdf_connections**: Number of clients listening for PDF processing updates
- **summary_connections**: Number of clients listening for summarization updates
- **total_connections**: Combined total of all active connections
- **batches_with_connections**: List of batch IDs that have active listeners
""",
)
def get_websocket_stats():
    """Get WebSocket connection statistics for monitoring."""
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


@app.get(
    "/summary/pdfs/{batch_id}",
    response_model=PDFListResponse,
    responses={
        200: {
            "description": "List of PDFs retrieved successfully",
            "model": PDFListResponse
        },
        404: {
            "description": "Batch not found or no PDFs indexed",
            "model": ErrorResponse
        },
        500: {
            "description": "Internal server error",
            "model": ErrorResponse
        }
    },
    tags=["Summarization"],
    summary="List PDFs in a batch",
    description="""
Retrieve the list of PDF filenames that have been processed and indexed for a given batch.

## Use Case

Call this endpoint before generating summaries to:
- Verify which PDFs are available for summarization
- Get the exact filenames needed for the `/summary/pdf` endpoint
- Check if all expected PDFs have been indexed

## Prerequisites

The batch must have completed the document processing stage. PDFs that are still
being processed will not appear in this list.

## Example Response

```json
{
    "batch_id": "550e8400-e29b-41d4-a716-446655440000",
    "pdf_names": ["annual_report.pdf", "quarterly_analysis.pdf"],
    "count": 2
}
```
""",
)
def get_pdfs_in_batch(
    batch_id: str = Field(
        ...,
        description="Batch identifier from the upload response",
        example="550e8400-e29b-41d4-a716-446655440000"
    )
):
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


@app.get(
    "/summary/pdf",
    response_model=None,  # Dynamic response based on pdf_name parameter
    responses={
        200: {
            "description": "Summary generated successfully",
            "content": {
                "application/json": {
                    "examples": {
                        "single_pdf": {
                            "summary": "Single PDF summary response",
                            "value": {
                                "batch_id": "550e8400-e29b-41d4-a716-446655440000",
                                "pdf_name": "report.pdf",
                                "summary_type": "brief",
                                "summary": "This document provides a comprehensive analysis...",
                                "method": "hierarchical",
                                "chunk_count": 48,
                                "cached": False
                            }
                        },
                        "all_pdfs": {
                            "summary": "All PDFs individual summaries response",
                            "value": {
                                "batch_id": "550e8400-e29b-41d4-a716-446655440000",
                                "summary_type": "brief",
                                "total_pdfs": 2,
                                "summaries": {
                                    "report.pdf": {"summary": "...", "cached": False},
                                    "analysis.pdf": {"summary": "...", "cached": True}
                                }
                            }
                        }
                    }
                }
            }
        },
        404: {
            "description": "Batch or PDF not found",
            "model": ErrorResponse
        },
        500: {
            "description": "Summary generation failed",
            "model": ErrorResponse
        }
    },
    tags=["Summarization"],
    summary="Generate PDF summary",
    description="""
Generate an AI-powered summary for one or more PDF documents.

## Modes

### Single PDF Mode
When `pdf_name` is provided, generates and returns a summary for that specific PDF.

### All PDFs Mode
When `pdf_name` is omitted, generates individual summaries for **all** PDFs in the batch.

## Summary Types

| Type | Description | Typical Length |
|------|-------------|----------------|
| `brief` | Concise overview of key points | 100-200 words |
| `bulletwise` | Bullet-point format with main takeaways | 10-20 bullets |
| `detailed` | Comprehensive summary with sections | 500-1000 words |
| `executive` | Executive summary for decision makers | 200-400 words |

## Summarization Methods

The system automatically selects the optimal method:

### Direct Method
- Used for small documents (< 4000 words)
- Single LLM call with all content
- Faster processing

### Hierarchical Method (Map-Reduce)
- Used for large documents (> 4000 words)
- Batches content into groups
- Summarizes each batch (MAP phase)
- Combines batch summaries (REDUCE phase)
- Handles documents of any size

## Caching

Summaries are cached for performance:
- **Redis**: Intermediate results (1 hour TTL)
- **Milvus**: Final summaries (persistent)

The `cached` field in the response indicates if a cached summary was returned.

## Real-time Updates

Connect to `/ws/summary/{batch_id}` to receive real-time progress updates
during summary generation.
""",
)
def summarize_pdf(
    batch_id: str = Query(
        ...,
        description="Batch identifier from the upload response",
        example="550e8400-e29b-41d4-a716-446655440000"
    ),
    pdf_name: Optional[str] = Query(
        None,
        description="PDF filename to summarize. If omitted, all PDFs in the batch are summarized individually.",
        example="annual_report.pdf"
    ),
    summary_type: SummaryType = Query(
        SummaryType.BRIEF,
        description="Type of summary to generate: brief (concise), bulletwise (bullet points), detailed (comprehensive), or executive"
    )
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


@app.get(
    "/summary/all",
    response_model=SummaryResponse,
    responses={
        200: {
            "description": "Combined summary generated successfully",
            "model": SummaryResponse
        },
        404: {
            "description": "Batch not found or no PDFs indexed",
            "model": ErrorResponse
        },
        500: {
            "description": "Summary generation failed",
            "model": ErrorResponse
        }
    },
    tags=["Summarization"],
    summary="Generate combined summary for all PDFs",
    description="""
Generate a **unified summary** that combines content from all PDFs in the batch.

## Difference from /summary/pdf

| Endpoint | Behavior |
|----------|----------|
| `/summary/pdf` | Individual summaries for each PDF |
| `/summary/all` | Single combined summary across all PDFs |

## Use Cases

- **Multi-document analysis**: Summarize a collection of related documents
- **Research synthesis**: Combine findings from multiple papers
- **Report generation**: Create executive summary from multiple source files

## Process

1. Retrieves all chunks from all PDFs in the batch
2. Generates individual PDF summaries (if not cached)
3. Combines all PDF summaries into a unified document summary
4. Identifies common themes and cross-document insights

## Real-time Updates

Connect to `/ws/summary/{batch_id}` to receive:
- `multi_pdf.started` - Batch summary started
- `multi_pdf.pdf_started` / `multi_pdf.pdf_completed` - Per-PDF progress
- `multi_pdf.combining` - Final combination phase
- `multi_pdf.completed` - Summary ready

## Example Response

```json
{
    "batch_id": "550e8400-e29b-41d4-a716-446655440000",
    "pdf_name": "_combined_",
    "summary_type": "brief",
    "summary": "Across the three uploaded documents, the key findings indicate...",
    "method": "multi_pdf_combine",
    "chunk_count": 156,
    "cached": false,
    "pdf_count": 3
}
```
""",
)
def summarize_all_pdfs(
    batch_id: str = Query(
        ...,
        description="Batch identifier from the upload response",
        example="550e8400-e29b-41d4-a716-446655440000"
    ),
    summary_type: SummaryType = Query(
        SummaryType.BRIEF,
        description="Type of combined summary to generate"
    )
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


# ============================================================================
# REFINEMENT REQUEST MODELS
# ============================================================================

class RefinementRequest(BaseModel):
    """
    Request body for simple summary refinement.

    Used to iteratively improve a summary based on user feedback without
    accessing the original source documents.
    """
    batch_id: str = Field(
        ...,
        description="Batch identifier containing the PDF",
        example="550e8400-e29b-41d4-a716-446655440000"
    )
    pdf_name: str = Field(
        ...,
        description="Name of the PDF whose summary should be refined",
        example="annual_report.pdf"
    )
    summary_type: SummaryType = Field(
        SummaryType.DETAILED,
        description="Type of the original summary to retrieve from cache if original_summary not provided"
    )
    user_feedback: str = Field(
        ...,
        description="Instructions for how to refine the summary. Be specific about desired changes.",
        min_length=1,
        max_length=2000,
        example="Make the summary more concise and focus on financial metrics. Use bullet points."
    )
    original_summary: Optional[str] = Field(
        None,
        description="The original summary text to refine. If not provided, the cached summary will be retrieved."
    )
    user_id: Optional[str] = Field(
        None,
        description="Optional user identifier for audit logging and analytics",
        example="user_12345"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "batch_id": "550e8400-e29b-41d4-a716-446655440000",
                "pdf_name": "annual_report.pdf",
                "summary_type": "detailed",
                "user_feedback": "Make it more concise and use bullet points for key metrics",
                "user_id": "user_12345"
            }
        }


class ContextualRefinementRequest(RefinementRequest):
    """
    Request body for contextual summary refinement with source chunk retrieval.

    Extends simple refinement by using vector search to find relevant source
    chunks that can provide additional context for the refinement.
    """
    top_k: int = Field(
        10,
        description="Number of relevant source chunks to retrieve via vector search. Higher values provide more context but may slow down processing.",
        ge=1,
        le=50,
        example=15
    )

    class Config:
        json_schema_extra = {
            "example": {
                "batch_id": "550e8400-e29b-41d4-a716-446655440000",
                "pdf_name": "annual_report.pdf",
                "summary_type": "detailed",
                "user_feedback": "Include more details about Q3 financial projections and risk factors",
                "top_k": 15,
                "user_id": "user_12345"
            }
        }


@app.post(
    "/summary/refine/simple",
    response_model=RefinedSummaryResponse,
    responses={
        200: {
            "description": "Summary refined successfully",
            "model": RefinedSummaryResponse
        },
        404: {
            "description": "Cached summary not found (provide original_summary or generate summary first)",
            "model": ErrorResponse
        },
        422: {
            "description": "Validation error in request body"
        },
        500: {
            "description": "Refinement failed",
            "model": ErrorResponse
        }
    },
    tags=["Summary Refinement"],
    summary="Refine summary using feedback only",
    description="""
Refine a summary using **only** the original summary text and user feedback.

## Overview

This is a fast refinement option that doesn't access source documents. The LLM
modifies the existing summary based solely on the user's instructions.

## Best For

| Use Case | Example Feedback |
|----------|------------------|
| **Style changes** | "Make it more formal and professional" |
| **Format changes** | "Convert to bullet points" |
| **Length adjustment** | "Condense to 3 sentences" |
| **Tone modification** | "Make it more accessible for non-technical readers" |
| **Focus shift** | "Emphasize the conclusions more" |

## Limitations

⚠️ Cannot add information not present in the original summary. For adding
missing details, use `/summary/refine/contextual` instead.

## Process

```
Original Summary + User Feedback → LLM → Refined Summary
```

## Cache Behavior

If `original_summary` is not provided in the request, the system will attempt
to retrieve it from cache using `batch_id`, `pdf_name`, and `summary_type`.

## Example

**Request:**
```json
{
    "batch_id": "550e8400-e29b-41d4-a716-446655440000",
    "pdf_name": "report.pdf",
    "summary_type": "detailed",
    "user_feedback": "Shorten to 100 words and highlight key metrics"
}
```

**Response:**
```json
{
    "batch_id": "550e8400-e29b-41d4-a716-446655440000",
    "pdf_name": "report.pdf",
    "refined_summary": "The Q3 report shows 15% revenue growth...",
    "refinement_type": "simple"
}
```
""",
)
def refine_summary_simple_endpoint(request: RefinementRequest):
    """
    Refine a summary using ONLY the original summary and user feedback.

    Fast option that doesn't require access to source chunks.
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


@app.post(
    "/summary/refine/contextual",
    response_model=ContextualRefinedSummaryResponse,
    responses={
        200: {
            "description": "Summary refined successfully with context",
            "model": ContextualRefinedSummaryResponse
        },
        404: {
            "description": "Cached summary not found or no relevant chunks found",
            "model": ErrorResponse
        },
        422: {
            "description": "Validation error in request body"
        },
        500: {
            "description": "Refinement failed",
            "model": ErrorResponse
        }
    },
    tags=["Summary Refinement"],
    summary="Refine summary using feedback and source context",
    description="""
Refine a summary using the original summary, user feedback, **and** relevant source chunks
retrieved via semantic search.

## Overview

This is the most powerful refinement option. It can add information that was not
in the original summary by retrieving relevant passages from the source document.

## How It Works

```
User Feedback
     ↓
Generate Embedding (E5-large)
     ↓
Vector Search in Milvus
     ↓
Filter chunks for target PDF
     ↓
Original Summary + Feedback + Context Chunks → LLM → Refined Summary
```

## Best For

| Use Case | Example Feedback |
|----------|------------------|
| **Adding details** | "Include more about the Q3 financial projections" |
| **Topic expansion** | "Elaborate on the methodology section" |
| **Missing info** | "Add information about risk factors" |
| **Specific questions** | "What were the key recommendations?" |
| **Factual additions** | "Include the specific metrics mentioned" |

## Parameters

### top_k (default: 10, max: 50)

Number of source chunks to retrieve via vector search. Higher values provide
more context but may:
- Include less relevant information
- Increase processing time
- Increase LLM token usage

**Recommendations:**
- `5-10`: Focused refinement on specific topics
- `10-20`: General enhancement with more context
- `20-50`: Comprehensive refinement for detailed requests

## Response Details

The response includes information about which source chunks were used:

```json
{
    "refined_summary": "The refined text...",
    "chunks_used": 8,
    "chunks_detail": [
        {
            "page": 12,
            "type": "text",
            "relevance": 0.92,
            "preview": "Financial projections for Q3..."
        },
        {
            "page": 15,
            "type": "table",
            "relevance": 0.88,
            "preview": "Revenue breakdown table..."
        }
    ]
}
```

## Example

**Request:**
```json
{
    "batch_id": "550e8400-e29b-41d4-a716-446655440000",
    "pdf_name": "annual_report.pdf",
    "summary_type": "detailed",
    "user_feedback": "Include specific revenue figures and growth percentages from the financial section",
    "top_k": 15
}
```

**Response:**
```json
{
    "batch_id": "550e8400-e29b-41d4-a716-446655440000",
    "pdf_name": "annual_report.pdf",
    "refined_summary": "The company achieved $42.5M in Q3 revenue, representing 18% YoY growth...",
    "refinement_type": "contextual",
    "chunks_used": 8,
    "chunks_detail": [...]
}
```
""",
)
def refine_summary_contextual_endpoint(request: ContextualRefinementRequest):
    """
    Refine a summary using original summary, user feedback, AND relevant source chunks.

    Uses vector search to find passages relevant to the user's feedback.
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
