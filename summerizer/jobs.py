import json
import os
import traceback
from pathlib import Path
from typing import Optional
import fitz

from pdf_utils import process_page, process_page_with_positions
from chunking_utils import (
    semantic_chunk_text,
    create_multimodal_chunks,
    create_semantic_multimodal_chunks,
    SemanticChunker
)
from embedding_client import EmbeddingClient
from vector_store.milvus_store import MilvusVectorStore
from chunk_indexer import ChunkIndexer
from vision_utils import VisionProcessor
from job_state import (
    update_job,
    init_pdf,
    update_pdf
)

SESSION_TTL_SECONDS = 3600
EMBEDDING_DIM = 1024
EMBEDDING_SERVICE_URL = "http://localhost:8000"
MILVUS_HOST = "localhost"
MILVUS_PORT = 19530

# Vision API configuration (optional)
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")


def process_pdf_batch(batch_id: str, input_dir: str, output_dir: str):
    print("JOB STARTED:", batch_id)

    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    print("yes JOB STARTED:", batch_id)

    # Initialize embedding client
    embedder = EmbeddingClient(EMBEDDING_SERVICE_URL)

    # Initialize Milvus vector store with error handling
    try:
        vector_store = MilvusVectorStore(
            host=MILVUS_HOST,
            port=MILVUS_PORT
        )
        print(f"Connected to Milvus at {MILVUS_HOST}:{MILVUS_PORT}")
    except Exception as e:
        print(f"ERROR: Failed to connect to Milvus at {MILVUS_HOST}:{MILVUS_PORT}: {e}")
        update_job(batch_id, state="failed", error=f"Milvus connection failed: {str(e)}")
        raise

    indexer = ChunkIndexer(embedder, vector_store, EMBEDDING_DIM)

    # -------- Count total pages (batch level) --------
    total_pages = 0
    for pdf_file in input_path.glob("*.pdf"):
        doc = fitz.open(pdf_file)
        total_pages += doc.page_count

    update_job(batch_id, state="running", total_pages=total_pages)

    processed_pages_global = 0
    total_chunks = 0

    for pdf_file in input_path.glob("*.pdf"):
        pdf_name = pdf_file.stem
        pdf_out_dir = output_path / pdf_name
        pdf_out_dir.mkdir(parents=True, exist_ok=True)

        doc = fitz.open(pdf_file)
        init_pdf(batch_id, pdf_name, doc.page_count)

        pages_result = []
        pdf_processed_pages = 0

        for i, page in enumerate(doc):
            page_result = process_page(page, i, pdf_file.name, pdf_out_dir)
            pages_result.append(page_result)

            pdf_processed_pages += 1
            processed_pages_global += 1

            update_pdf(
                batch_id,
                pdf_name,
                processed_pages=pdf_processed_pages
            )

            update_job(
                batch_id,
                processed_pages=processed_pages_global
            )

        # -------- Save page results --------
        with open(pdf_out_dir / "result.json", "w", encoding="utf-8") as f:
            json.dump(pages_result, f, ensure_ascii=False, indent=2)

        # -------- Chunking --------
        all_chunks = []
        for page in pages_result:
            if page["scanned"]:
                continue
            if not page.get("text_markdown"):
                continue

            all_chunks.extend(
                semantic_chunk_text(
                    text=page["text_markdown"],
                    pdf_name=page["metadata"]["pdf_name"],
                    page_no=page["metadata"]["page_no"]
                )
            )

        total_chunks += len(all_chunks)

        update_pdf(
            batch_id,
            pdf_name,
            chunk_count=len(all_chunks)
        )

        update_job(batch_id, chunk_count=total_chunks)

        with open(pdf_out_dir / "chunks.json", "w", encoding="utf-8") as f:
            json.dump(all_chunks, f, ensure_ascii=False, indent=2)

        # -------- Milvus indexing --------
        if not all_chunks:
            print(f"WARNING: No chunks generated for {pdf_name}. Skipping indexing.")
        else:
            try:
                print(f"Indexing {len(all_chunks)} chunks for {pdf_name}...")
                indexer.index_chunks(
                    session_id=batch_id,
                    chunks=all_chunks,
                    ttl_seconds=SESSION_TTL_SECONDS
                )
                print(f"Successfully indexed {len(all_chunks)} chunks for {pdf_name}")
            except Exception as e:
                print(f"ERROR: Failed to index chunks for {pdf_name}: {e}")
                print(traceback.format_exc())
                update_pdf(
                    batch_id,
                    pdf_name,
                    status="failed",
                    error=f"Indexing failed: {str(e)}"
                )
                continue

        update_pdf(
            batch_id,
            pdf_name,
            status="completed",
            progress=100
        )

    update_job(
        batch_id,
        state="completed",
        progress=100,
        milvus_indexed=True
    )


def process_pdf_batch_multimodal(
    batch_id: str,
    input_dir: str,
    output_dir: str,
    use_vision: bool = True,
    use_semantic_chunking: bool = False,
    semantic_similarity_threshold: float = 0.5,
    semantic_percentile_threshold: Optional[float] = 25,
    semantic_min_chunk_size: int = 50,
    semantic_max_chunk_size: int = 500
):
    """
    Process PDF batch with multimodal support (text, tables, images).

    This function:
    1. Extracts content with position tracking
    2. Saves tables as images for vision model processing
    3. Uses vision models (if available) to describe images and tables
    4. Creates chunks with image_link and table_link for explainability
    5. Optionally uses embedding-based semantic chunking for better topic boundaries
    6. Indexes all chunks into Milvus

    Args:
        batch_id: Unique batch identifier
        input_dir: Directory containing PDF files
        output_dir: Directory for output files
        use_vision: Whether to use vision model for image/table descriptions
        use_semantic_chunking: Whether to use embedding-based semantic chunking
        semantic_similarity_threshold: Similarity threshold for semantic chunking (0-1)
        semantic_percentile_threshold: Use bottom N percentile as breakpoints
        semantic_min_chunk_size: Minimum words per chunk for semantic chunking
        semantic_max_chunk_size: Maximum words per chunk for semantic chunking
    """
    print(f"MULTIMODAL JOB STARTED: {batch_id}")

    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Initialize embedding client
    embedder = EmbeddingClient(EMBEDDING_SERVICE_URL)

    # Initialize Milvus vector store
    try:
        vector_store = MilvusVectorStore(host=MILVUS_HOST, port=MILVUS_PORT)
        print(f"Connected to Milvus at {MILVUS_HOST}:{MILVUS_PORT}")
    except Exception as e:
        print(f"ERROR: Failed to connect to Milvus: {e}")
        update_job(batch_id, state="failed", error=f"Milvus connection failed: {str(e)}")
        raise

    indexer = ChunkIndexer(embedder, vector_store, EMBEDDING_DIM)

    # Initialize vision processor (optional)
    vision_processor = None
    if use_vision:
        vision_processor = VisionProcessor(
            openai_api_key=OPENAI_API_KEY,
            anthropic_api_key=ANTHROPIC_API_KEY
        )

    # Count total pages
    total_pages = 0
    for pdf_file in input_path.glob("*.pdf"):
        doc = fitz.open(pdf_file)
        total_pages += doc.page_count
        doc.close()

    update_job(batch_id, state="running", total_pages=total_pages)

    processed_pages_global = 0
    total_chunks = 0
    all_batch_results = []

    for pdf_file in input_path.glob("*.pdf"):
        pdf_name = pdf_file.stem
        pdf_out_dir = output_path / pdf_name
        pdf_out_dir.mkdir(parents=True, exist_ok=True)

        doc = fitz.open(pdf_file)
        init_pdf(batch_id, pdf_name, doc.page_count)

        pages_result = []
        all_chunks = []
        pdf_processed_pages = 0
        chunk_number = 0

        print(f"Processing {pdf_name} ({doc.page_count} pages)...")

        for i, page in enumerate(doc):
            # Use new multimodal extraction
            page_result = process_page_with_positions(
                page, i, pdf_file.name, pdf_out_dir
            )
            pages_result.append(page_result)

            # Process blocks with vision model
            blocks = page_result.get("blocks", [])
            if vision_processor and blocks:
                print(f"  Page {i+1}: Processing {len(blocks)} blocks with vision model...")
                blocks = vision_processor.process_blocks(blocks)
                page_result["blocks"] = blocks

            # Create chunks from blocks
            if use_semantic_chunking:
                # Use embedding-based semantic chunking
                page_chunks = create_semantic_multimodal_chunks(
                    blocks=blocks,
                    pdf_name=pdf_file.name,
                    page_no=page_result["metadata"]["page_no"],
                    embedding_client=embedder,
                    similarity_threshold=semantic_similarity_threshold,
                    percentile_threshold=semantic_percentile_threshold,
                    min_chunk_size=semantic_min_chunk_size,
                    max_chunk_size=semantic_max_chunk_size,
                    start_chunk_number=chunk_number
                )
            else:
                # Use simple word-count based chunking
                page_chunks = create_multimodal_chunks(
                    blocks=blocks,
                    pdf_name=pdf_file.name,
                    page_no=page_result["metadata"]["page_no"],
                    max_words=250,
                    start_chunk_number=chunk_number
                )

            all_chunks.extend(page_chunks)
            chunk_number += len(page_chunks)

            # Update progress
            pdf_processed_pages += 1
            processed_pages_global += 1

            update_pdf(batch_id, pdf_name, processed_pages=pdf_processed_pages)
            update_job(batch_id, processed_pages=processed_pages_global)

        doc.close()

        # Save results
        with open(pdf_out_dir / "result.json", "w", encoding="utf-8") as f:
            json.dump(pages_result, f, ensure_ascii=False, indent=2)

        # Remove metadata field from chunks before saving (not JSON serializable issues)
        chunks_to_save = []
        for chunk in all_chunks:
            chunk_copy = {k: v for k, v in chunk.items() if k != "metadata"}
            chunks_to_save.append(chunk_copy)

        with open(pdf_out_dir / "chunks.json", "w", encoding="utf-8") as f:
            json.dump(chunks_to_save, f, ensure_ascii=False, indent=2)

        total_chunks += len(all_chunks)
        update_pdf(batch_id, pdf_name, chunk_count=len(all_chunks))
        update_job(batch_id, chunk_count=total_chunks)

        # Index chunks
        if not all_chunks:
            print(f"WARNING: No chunks generated for {pdf_name}. Skipping indexing.")
        else:
            try:
                print(f"Indexing {len(all_chunks)} multimodal chunks for {pdf_name}...")
                indexer.index_multimodal_chunks(
                    session_id=batch_id,
                    chunks=all_chunks,
                    ttl_seconds=SESSION_TTL_SECONDS
                )
                print(f"Successfully indexed {len(all_chunks)} chunks for {pdf_name}")
            except Exception as e:
                print(f"ERROR: Failed to index chunks for {pdf_name}: {e}")
                print(traceback.format_exc())
                update_pdf(batch_id, pdf_name, status="failed", error=f"Indexing failed: {str(e)}")
                continue

        update_pdf(batch_id, pdf_name, status="completed", progress=100)

        # Add to batch results
        all_batch_results.append({
            "pdf_name": pdf_name,
            "page_count": len(pages_result),
            "chunk_count": len(all_chunks),
            "content_summary": _summarize_content_types(all_chunks)
        })

    # Save batch summary
    batch_summary = {
        "batch_id": batch_id,
        "total_pdfs": len(all_batch_results),
        "total_pages": total_pages,
        "total_chunks": total_chunks,
        "pdfs": all_batch_results
    }

    with open(output_path / "batch_summary.json", "w", encoding="utf-8") as f:
        json.dump(batch_summary, f, ensure_ascii=False, indent=2)

    update_job(
        batch_id,
        state="completed",
        progress=100,
        milvus_indexed=True
    )

    print(f"MULTIMODAL JOB COMPLETED: {batch_id}")
    print(f"  Total PDFs: {len(all_batch_results)}")
    print(f"  Total Pages: {total_pages}")
    print(f"  Total Chunks: {total_chunks}")


def _summarize_content_types(chunks: list) -> dict:
    """Summarize content types in chunks."""
    summary = {"text": 0, "table": 0, "image": 0}
    for chunk in chunks:
        ct = chunk.get("content_type", "text")
        if ct in summary:
            summary[ct] += 1
    return summary
