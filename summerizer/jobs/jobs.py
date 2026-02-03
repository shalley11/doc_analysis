import json
import os
import traceback
from pathlib import Path
from typing import Optional
import fitz

from pdf.pdf_utils import process_page, process_page_with_positions, process_page_with_structure
from pdf.structure_detector import analyze_document_fonts, detect_document_structure
from chunking.chunking_utils import (
    semantic_chunk_text,
    create_multimodal_chunks,
    create_semantic_multimodal_chunks,
    SemanticChunker,
    process_document_with_structure
)
from embedding.embedding_client import EmbeddingClient
from vector_store.milvus_store import MilvusVectorStore
from chunking.chunk_indexer import ChunkIndexer
from pdf.vision_utils import VisionProcessor
from jobs.job_state import (
    update_job,
    init_pdf,
    update_pdf
)
from jobs.processing_status import (
    ProcessingStage,
    StageStatus,
    init_batch_status,
    start_batch,
    set_batch_totals,
    start_pdf,
    update_stage,
    update_extraction_progress,
    update_chunking_progress,
    complete_pdf,
    complete_batch,
    fail_batch,
    fail_pdf
)

SESSION_TTL_SECONDS = 3600
EMBEDDING_DIM = 1024
EMBEDDING_SERVICE_URL = "http://localhost:8000"
MILVUS_HOST = "localhost"
MILVUS_PORT = 19530

# Vision API configuration (optional)
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")

# Gemma 3 4B configuration (recommended for vision)
# Set USE_GEMMA3=true and GEMMA3_MODE=local or GEMMA3_MODE=api
USE_GEMMA3 = os.environ.get("USE_GEMMA3", "false").lower() == "true"
GEMMA3_MODE = os.environ.get("GEMMA3_MODE", "local")  # "local" or "api"

# Ollama configuration (for local models)
OLLAMA_MODEL = os.environ.get("OLLAMA_VISION_MODEL")  # e.g., "llava", "moondream"
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")

# Vision model batch processing configuration
VISION_BATCH_SIZE = int(os.environ.get("VISION_BATCH_SIZE", "5"))  # Batch size for image/table processing


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

    # Get list of PDFs for status initialization
    pdf_files = list(input_path.glob("*.pdf"))
    pdf_names = [f.stem for f in pdf_files]

    # Initialize detailed status tracking
    init_batch_status(batch_id, pdf_names)
    start_batch(batch_id)

    # Initialize embedding client
    update_stage(batch_id, "", ProcessingStage.INITIALIZING, StageStatus.RUNNING,
                 message="Connecting to services...")
    embedder = EmbeddingClient(EMBEDDING_SERVICE_URL)

    # Initialize Milvus vector store
    try:
        vector_store = MilvusVectorStore(host=MILVUS_HOST, port=MILVUS_PORT)
        print(f"Connected to Milvus at {MILVUS_HOST}:{MILVUS_PORT}")
    except Exception as e:
        print(f"ERROR: Failed to connect to Milvus: {e}")
        update_job(batch_id, state="failed", error=f"Milvus connection failed: {str(e)}")
        fail_batch(batch_id, f"Milvus connection failed: {str(e)}")
        raise

    indexer = ChunkIndexer(embedder, vector_store, EMBEDDING_DIM)

    # Initialize vision processor (optional)
    # Priority: Gemma3 > Anthropic > OpenAI > Gemini > Ollama > Fallback
    vision_processor = None
    if use_vision:
        vision_processor = VisionProcessor(
            use_gemma3=USE_GEMMA3,
            gemma3_mode=GEMMA3_MODE,
            openai_api_key=OPENAI_API_KEY,
            anthropic_api_key=ANTHROPIC_API_KEY,
            google_api_key=GOOGLE_API_KEY,
            ollama_model=OLLAMA_MODEL,
            ollama_base_url=OLLAMA_BASE_URL
        )

    # Count total pages
    total_pages = 0
    for pdf_file in pdf_files:
        doc = fitz.open(pdf_file)
        total_pages += doc.page_count
        doc.close()

    update_job(batch_id, state="running", total_pages=total_pages)
    set_batch_totals(batch_id, total_pages)

    processed_pages_global = 0
    total_chunks = 0
    all_batch_results = []

    for pdf_file in pdf_files:
        pdf_name = pdf_file.stem
        pdf_out_dir = output_path / pdf_name
        pdf_out_dir.mkdir(parents=True, exist_ok=True)

        doc = fitz.open(pdf_file)
        init_pdf(batch_id, pdf_name, doc.page_count)
        start_pdf(batch_id, pdf_name, doc.page_count)

        pages_result = []
        all_chunks = []
        pdf_processed_pages = 0
        chunk_number = 0

        print(f"Processing {pdf_name} ({doc.page_count} pages)...")

        # Stage: Extraction
        update_stage(batch_id, pdf_name, ProcessingStage.EXTRACTING, StageStatus.RUNNING,
                     message=f"Extracting content from {pdf_name}")

        for i, page in enumerate(doc):
            # Use new multimodal extraction
            page_result = process_page_with_positions(
                page, i, pdf_file.name, pdf_out_dir
            )
            pages_result.append(page_result)

            # Update extraction progress
            update_extraction_progress(batch_id, pdf_name, i, doc.page_count)

            # Process blocks with vision model
            blocks = page_result.get("blocks", [])
            if vision_processor and blocks:
                # Stage: Vision processing
                update_stage(batch_id, pdf_name, ProcessingStage.VISION_PROCESSING, StageStatus.RUNNING,
                             progress=int((i + 1) / doc.page_count * 100),
                             current_item=f"Page {i+1} of {doc.page_count}",
                             message=f"Processing {len(blocks)} blocks with vision model (batch_size={VISION_BATCH_SIZE})")
                print(f"  Page {i+1}: Processing {len(blocks)} blocks with vision model (batch_size={VISION_BATCH_SIZE})...")
                # Use process_blocks_with_metadata to populate table_summary, image_caption, image_summary
                blocks = vision_processor.process_blocks_with_metadata(blocks, batch_size=VISION_BATCH_SIZE)
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

        # Mark extraction complete
        update_stage(batch_id, pdf_name, ProcessingStage.EXTRACTING, StageStatus.COMPLETED,
                     progress=100, message=f"Extracted {len(pages_result)} pages")

        if vision_processor:
            update_stage(batch_id, pdf_name, ProcessingStage.VISION_PROCESSING, StageStatus.COMPLETED,
                         progress=100, message="Vision processing complete")

        # Stage: Chunking
        update_stage(batch_id, pdf_name, ProcessingStage.CHUNKING, StageStatus.RUNNING,
                     message="Creating text chunks...")

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
        update_chunking_progress(batch_id, pdf_name, len(all_chunks))

        update_stage(batch_id, pdf_name, ProcessingStage.CHUNKING, StageStatus.COMPLETED,
                     progress=100, message=f"Created {len(all_chunks)} chunks")

        # Stage: Indexing
        if not all_chunks:
            print(f"WARNING: No chunks generated for {pdf_name}. Skipping indexing.")
            update_stage(batch_id, pdf_name, ProcessingStage.INDEXING, StageStatus.SKIPPED,
                         message="No chunks to index")
        else:
            try:
                update_stage(batch_id, pdf_name, ProcessingStage.INDEXING, StageStatus.RUNNING,
                             message=f"Indexing {len(all_chunks)} chunks into Milvus...")

                print(f"Indexing {len(all_chunks)} multimodal chunks for {pdf_name}...")
                indexer.index_multimodal_chunks(
                    session_id=batch_id,
                    chunks=all_chunks,
                    ttl_seconds=SESSION_TTL_SECONDS
                )
                print(f"Successfully indexed {len(all_chunks)} chunks for {pdf_name}")

                update_stage(batch_id, pdf_name, ProcessingStage.INDEXING, StageStatus.COMPLETED,
                             progress=100, message=f"Indexed {len(all_chunks)} chunks")

            except Exception as e:
                print(f"ERROR: Failed to index chunks for {pdf_name}: {e}")
                print(traceback.format_exc())
                update_pdf(batch_id, pdf_name, status="failed", error=f"Indexing failed: {str(e)}")
                update_stage(batch_id, pdf_name, ProcessingStage.INDEXING, StageStatus.FAILED,
                             error=str(e))
                fail_pdf(batch_id, pdf_name, str(e))
                continue

        update_pdf(batch_id, pdf_name, status="completed", progress=100)
        complete_pdf(batch_id, pdf_name, len(all_chunks))

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

    # Mark batch as completed
    complete_batch(batch_id)

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


def process_pdf_batch_structured(
    batch_id: str,
    input_dir: str,
    output_dir: str,
    use_vision: bool = True,
    max_words: int = 500,
    min_words: int = 50
):
    """
    Process PDF batch with structure-aware chunking.

    This function:
    1. Analyzes document fonts to detect heading hierarchy
    2. Extracts content with section hierarchy tracking
    3. Creates chunks with [Section: ...] prefix for context
    4. Preserves headers in split tables/lists
    5. Indexes all chunks with section_hierarchy field

    Args:
        batch_id: Unique batch identifier
        input_dir: Directory containing PDF files
        output_dir: Directory for output files
        use_vision: Whether to use vision model for image/table descriptions
        max_words: Maximum words per chunk (default 500)
        min_words: Minimum words per chunk (default 50)
    """
    print(f"STRUCTURED JOB STARTED: {batch_id}")

    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Get list of PDFs
    pdf_files = list(input_path.glob("*.pdf"))
    pdf_names = [f.stem for f in pdf_files]

    # Initialize status tracking
    init_batch_status(batch_id, pdf_names)
    start_batch(batch_id)

    # Initialize services
    update_stage(batch_id, "", ProcessingStage.INITIALIZING, StageStatus.RUNNING,
                 message="Connecting to services...")
    embedder = EmbeddingClient(EMBEDDING_SERVICE_URL)

    try:
        vector_store = MilvusVectorStore(host=MILVUS_HOST, port=MILVUS_PORT)
        print(f"Connected to Milvus at {MILVUS_HOST}:{MILVUS_PORT}")
    except Exception as e:
        print(f"ERROR: Failed to connect to Milvus: {e}")
        update_job(batch_id, state="failed", error=f"Milvus connection failed: {str(e)}")
        fail_batch(batch_id, f"Milvus connection failed: {str(e)}")
        raise

    indexer = ChunkIndexer(embedder, vector_store, EMBEDDING_DIM)

    # Initialize vision processor if requested
    vision_processor = None
    if use_vision:
        vision_processor = VisionProcessor(
            use_gemma3=USE_GEMMA3,
            gemma3_mode=GEMMA3_MODE,
            openai_api_key=OPENAI_API_KEY,
            anthropic_api_key=ANTHROPIC_API_KEY,
            google_api_key=GOOGLE_API_KEY,
            ollama_model=OLLAMA_MODEL,
            ollama_base_url=OLLAMA_BASE_URL
        )

    # Count total pages
    total_pages = 0
    for pdf_file in pdf_files:
        doc = fitz.open(pdf_file)
        total_pages += doc.page_count
        doc.close()

    update_job(batch_id, state="running", total_pages=total_pages)
    set_batch_totals(batch_id, total_pages)

    processed_pages_global = 0
    total_chunks = 0
    all_batch_results = []

    for pdf_file in pdf_files:
        pdf_name = pdf_file.stem
        pdf_out_dir = output_path / pdf_name
        pdf_out_dir.mkdir(parents=True, exist_ok=True)

        doc = fitz.open(pdf_file)
        init_pdf(batch_id, pdf_name, doc.page_count)
        start_pdf(batch_id, pdf_name, doc.page_count)

        print(f"Processing {pdf_name} ({doc.page_count} pages) with structure detection...")

        # Stage: Font Analysis
        update_stage(batch_id, pdf_name, ProcessingStage.EXTRACTING, StageStatus.RUNNING,
                     message=f"Analyzing document structure for {pdf_name}")

        # Analyze document fonts and detect structure
        font_profile, page_block_paths = detect_document_structure(doc)
        print(f"  Font analysis: body_size={font_profile.body_size}, headings={len(font_profile.heading_sizes)}")

        pages_result = []
        pdf_processed_pages = 0

        # Stage: Extraction with structure
        for i, page in enumerate(doc):
            # Get section paths for this page
            block_paths = page_block_paths.get(i, {})

            # Extract page content with structure info
            page_result = process_page_with_structure(
                page, i, pdf_file.name, pdf_out_dir,
                font_profile=font_profile,
                page_block_paths=block_paths
            )
            pages_result.append(page_result)

            # Update progress
            update_extraction_progress(batch_id, pdf_name, i, doc.page_count)

            # Process blocks with vision model if available
            blocks = page_result.get("blocks", [])
            if vision_processor and blocks:
                update_stage(batch_id, pdf_name, ProcessingStage.VISION_PROCESSING, StageStatus.RUNNING,
                             progress=int((i + 1) / doc.page_count * 100),
                             current_item=f"Page {i+1} of {doc.page_count}",
                             message=f"Processing {len(blocks)} blocks with vision model (batch_size={VISION_BATCH_SIZE})")
                print(f"  Page {i+1}: Processing {len(blocks)} blocks with vision model (batch_size={VISION_BATCH_SIZE})...")
                # Use process_blocks_with_metadata to populate table_summary, image_caption, image_summary
                blocks = vision_processor.process_blocks_with_metadata(blocks, batch_size=VISION_BATCH_SIZE)
                page_result["blocks"] = blocks

            pdf_processed_pages += 1
            processed_pages_global += 1
            update_pdf(batch_id, pdf_name, processed_pages=pdf_processed_pages)
            update_job(batch_id, processed_pages=processed_pages_global)

        doc.close()

        update_stage(batch_id, pdf_name, ProcessingStage.EXTRACTING, StageStatus.COMPLETED,
                     progress=100, message=f"Extracted {len(pages_result)} pages with structure")

        if vision_processor:
            update_stage(batch_id, pdf_name, ProcessingStage.VISION_PROCESSING, StageStatus.COMPLETED,
                         progress=100, message="Vision processing complete")

        # Stage: Structure-aware chunking
        update_stage(batch_id, pdf_name, ProcessingStage.CHUNKING, StageStatus.RUNNING,
                     message="Creating structure-aware chunks...")

        # Create chunks with section hierarchy
        all_chunks = process_document_with_structure(
            pages=pages_result,
            pdf_name=pdf_file.name,
            max_words=max_words,
            min_words=min_words
        )

        # Save results
        with open(pdf_out_dir / "result.json", "w", encoding="utf-8") as f:
            json.dump(pages_result, f, ensure_ascii=False, indent=2)

        # Save chunks (excluding non-serializable fields)
        chunks_to_save = []
        for chunk in all_chunks:
            chunk_copy = {k: v for k, v in chunk.items() if k != "metadata" or isinstance(v, (dict, list, str, int, float, bool, type(None)))}
            chunks_to_save.append(chunk_copy)

        with open(pdf_out_dir / "chunks.json", "w", encoding="utf-8") as f:
            json.dump(chunks_to_save, f, ensure_ascii=False, indent=2)

        total_chunks += len(all_chunks)
        update_pdf(batch_id, pdf_name, chunk_count=len(all_chunks))
        update_job(batch_id, chunk_count=total_chunks)
        update_chunking_progress(batch_id, pdf_name, len(all_chunks))

        update_stage(batch_id, pdf_name, ProcessingStage.CHUNKING, StageStatus.COMPLETED,
                     progress=100, message=f"Created {len(all_chunks)} structure-aware chunks")

        # Stage: Indexing
        if not all_chunks:
            print(f"WARNING: No chunks generated for {pdf_name}. Skipping indexing.")
            update_stage(batch_id, pdf_name, ProcessingStage.INDEXING, StageStatus.SKIPPED,
                         message="No chunks to index")
        else:
            try:
                update_stage(batch_id, pdf_name, ProcessingStage.INDEXING, StageStatus.RUNNING,
                             message=f"Indexing {len(all_chunks)} chunks into Milvus...")

                print(f"Indexing {len(all_chunks)} structure-aware chunks for {pdf_name}...")
                indexer.index_multimodal_chunks(
                    session_id=batch_id,
                    chunks=all_chunks,
                    ttl_seconds=SESSION_TTL_SECONDS
                )
                print(f"Successfully indexed {len(all_chunks)} chunks for {pdf_name}")

                update_stage(batch_id, pdf_name, ProcessingStage.INDEXING, StageStatus.COMPLETED,
                             progress=100, message=f"Indexed {len(all_chunks)} chunks")

            except Exception as e:
                print(f"ERROR: Failed to index chunks for {pdf_name}: {e}")
                print(traceback.format_exc())
                update_pdf(batch_id, pdf_name, status="failed", error=f"Indexing failed: {str(e)}")
                update_stage(batch_id, pdf_name, ProcessingStage.INDEXING, StageStatus.FAILED,
                             error=str(e))
                fail_pdf(batch_id, pdf_name, str(e))
                continue

        update_pdf(batch_id, pdf_name, status="completed", progress=100)
        complete_pdf(batch_id, pdf_name, len(all_chunks))

        # Count chunks with section hierarchy
        chunks_with_sections = sum(1 for c in all_chunks if c.get("section_hierarchy"))

        all_batch_results.append({
            "pdf_name": pdf_name,
            "page_count": len(pages_result),
            "chunk_count": len(all_chunks),
            "chunks_with_sections": chunks_with_sections,
            "content_summary": _summarize_content_types(all_chunks)
        })

    # Save batch summary
    batch_summary = {
        "batch_id": batch_id,
        "total_pdfs": len(all_batch_results),
        "total_pages": total_pages,
        "total_chunks": total_chunks,
        "chunking_method": "structure_based",
        "max_words": max_words,
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

    complete_batch(batch_id)

    print(f"STRUCTURED JOB COMPLETED: {batch_id}")
    print(f"  Total PDFs: {len(all_batch_results)}")
    print(f"  Total Pages: {total_pages}")
    print(f"  Total Chunks: {total_chunks}")
    print(f"  Chunking Method: structure_based")
