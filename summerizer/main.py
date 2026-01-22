import uuid
from pathlib import Path
import shutil
from typing import List

from fastapi import FastAPI, UploadFile, File, Query
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import redis
from rq import Queue

from pdf.pdf_utils import process_page, page_to_markdown
from jobs.jobs import process_pdf_batch, process_pdf_batch_multimodal
from jobs.job_state import init_job, get_job

app = FastAPI(
    title="PDF Analysis API",
    description="Upload PDFs for async batch processing with multimodal support",
    version="2.0.0"
)

# Mount static files for serving images/tables
app.mount("/static", StaticFiles(directory="outputs"), name="static")

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


@app.post("/api/v2/pdf/analyze", tags=["PDFs - Multimodal"])
def analyze_pdfs_multimodal(
    files: List[UploadFile] = File(...),
    use_vision: str = Query("yes", description="Use vision model for image/table descriptions"),
    use_semantic_chunking: str = Query("no", description="Use embedding-based semantic chunking for better topic boundaries"),
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
    - Indexes all content into Milvus

    Semantic Chunking:
    - When enabled, uses sentence embeddings to detect topic shifts
    - Splits text at semantic boundaries rather than fixed word counts
    - Produces more coherent chunks that preserve topical context

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

    # Use multimodal processing job
    semantic_enabled = use_semantic_chunking.lower() == "yes"
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
    init_job(batch_id, rq_job_id=job.id)

    return JSONResponse({
        "batch_id": batch_id,
        "rq_job_id": job.id,
        "uploaded_files": [f.filename for f in files],
        "message": "Files uploaded. Multimodal background processing started.",
        "mode": "multimodal",
        "vision_enabled": use_vision.lower() == "yes",
        "semantic_chunking_enabled": semantic_enabled,
        "preview_markdown": previews if preview_only.lower() == "yes" else None,
        "static_base_url": f"/static/{batch_id}"
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


@app.get("/health")
def health():
    return {"status": "ok"}
