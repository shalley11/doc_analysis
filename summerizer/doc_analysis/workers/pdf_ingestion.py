"""
Simple PDF ingestion pipeline.

Flow: Extract → Chunk → Index
"""
import json
import time
from typing import Optional
from pathlib import Path

from doc_analysis.config import ChunkingConfig, DEFAULT_CHUNKING_CONFIG
from doc_analysis.logging_config import get_ingestion_logger, set_batch_id, clear_batch_id
from doc_analysis.realtime import (
    publish_pdf_event,
    pdf_started,
    pdf_extracting,
    pdf_extracted,
    pdf_chunking,
    pdf_chunked,
    pdf_completed,
    pdf_failed
)

logger = get_ingestion_logger()


def _publish_event(batch_id: str, event):
    """Safely publish event, logging any errors."""
    try:
        publish_pdf_event(batch_id, event)
    except Exception as e:
        logger.warning(f"[EVENT] Failed to publish event: {e}")

# POC: Path to save chunks for debugging
CHUNKING_JSON_PATH = Path(__file__).parent.parent / "chunking.json"

from doc_analysis.jobs.job_store import (
    update_batch_status,
    update_pdf_status,
    start_stage,
    complete_stage,
    fail_stage,
    update_stats
)
from doc_analysis.pdf.scan_detector import is_scanned_pdf
from doc_analysis.pdf.text_table_extractor import extract_text_and_tables
from doc_analysis.pdf.image_extractor import extract_images
from doc_analysis.chunking.chunk_builder import build_chunks
from doc_analysis.workers.ingestion_worker import ingest_batch


def _save_chunks_to_json(batch_id: str, chunks: list):
    """
    POC: Save chunks to chunking.json for debugging.
    Remove in PROD deployment.
    """
    try:
        serializable_chunks = []
        for chunk in chunks:
            chunk_copy = chunk.copy()
            chunk_copy.pop("embedding", None)
            serializable_chunks.append(chunk_copy)

        output = {
            "_comment": "POC Debug Output - Remove in PROD",
            "batch_id": batch_id,
            "total_chunks": len(serializable_chunks),
            "chunks": serializable_chunks,
        }

        with open(CHUNKING_JSON_PATH, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)

        logger.info(f"POC: Saved {len(serializable_chunks)} chunks to chunking.json")

    except Exception as e:
        logger.warning(f"POC: Failed to save chunks to JSON: {e}")


def ingest_batch_pdfs(
    batch_id: str,
    filenames: list,
    paths: list,
    embedding_dim: int,
    ttl_seconds: int,
    start_page: int = None,
    end_page: int = None,
    config: Optional[ChunkingConfig] = None,
    **kwargs,
):
    """
    Process a batch of PDFs with simple semantic chunking.

    Args:
        batch_id: Unique batch identifier
        filenames: List of PDF filenames (strings)
        paths: List of file paths
        embedding_dim: Embedding dimension for vector store
        ttl_seconds: TTL for Milvus collection
        start_page: Start page (1-indexed, inclusive). None means page 1.
        end_page: End page (1-indexed, inclusive). None means last page.
        config: Chunking configuration (uses default if not provided)
    """
    set_batch_id(batch_id)
    start_time = time.time()

    try:
        logger.info(f"Starting batch ingestion with {len(filenames)} PDF(s)")

        # Publish started event
        _publish_event(batch_id, pdf_started(batch_id, len(filenames), filenames))

        if config is None:
            config = DEFAULT_CHUNKING_CONFIG

        update_batch_status(batch_id, "processing")
        logger.info("Status: processing")

        all_chunks = []
        total_elements = 0

        # ===== EXTRACTION STAGE =====
        start_stage(batch_id, "extraction", f"Extracting from {len(filenames)} PDF(s)")

        for idx, (pdf_name, path) in enumerate(zip(filenames, paths)):
            pdf_id = f"pdf_{idx + 1}"

            logger.info(f"Processing {pdf_id}: {pdf_name}")

            try:
                # -------- extracting --------
                update_pdf_status(batch_id, pdf_id, "extracting")

                # Publish extracting event (page 1 initially)
                _publish_event(batch_id, pdf_extracting(batch_id, pdf_name, 1, 1))

                if is_scanned_pdf(path):
                    logger.warning(f"[{pdf_id}] Rejected: Scanned PDF detected")
                    update_pdf_status(batch_id, pdf_id, "failed", "Scanned PDF")
                    _publish_event(batch_id, pdf_failed(batch_id, "Scanned PDF detected", pdf_name))
                    continue

                # Extract text and tables
                elements = extract_text_and_tables(
                    path, pdf_id, pdf_name,
                    include_bbox=config.include_bbox,
                    start_page=start_page,
                    end_page=end_page
                )

                # Extract images
                image_elements = extract_images(
                    path, pdf_id, pdf_name,
                    start_page=start_page,
                    end_page=end_page
                )

                # Merge elements and sort by page number and position
                elements.extend(image_elements)
                elements.sort(key=lambda e: (e.get("page_no", 0), e.get("bbox", [0])[1] if e.get("bbox") else 0))

                # Count pages and content types
                pages = set(e.get("page_no", 0) for e in elements)
                total_pages = len(pages) if pages else 1
                tables_count = sum(1 for e in elements if e.get("content_type") == "table")
                images_count = len(image_elements)

                total_elements += len(elements)
                logger.info(f"[{pdf_id}] Extracted {len(elements)} elements (text/tables: {len(elements) - len(image_elements)}, images: {len(image_elements)})")

                update_pdf_status(batch_id, pdf_id, "extracted", elements=len(elements))

                # Publish extracted event
                _publish_event(batch_id, pdf_extracted(batch_id, pdf_name, total_pages, tables_count, images_count))

                # -------- chunking --------
                update_pdf_status(batch_id, pdf_id, "chunking")

                # Publish chunking event
                _publish_event(batch_id, pdf_chunking(batch_id, pdf_name, 0))

                chunks = build_chunks(elements, pdf_name, pdf_id, config)
                logger.info(f"[{pdf_id}] Created {len(chunks)} chunks")

                all_chunks.extend(chunks)
                update_pdf_status(batch_id, pdf_id, "completed", chunks=len(chunks))

                # Publish chunked event
                _publish_event(batch_id, pdf_chunked(batch_id, pdf_name, len(chunks)))

            except Exception as e:
                logger.error(f"[{pdf_id}] Failed: {str(e)}", exc_info=True)
                update_pdf_status(batch_id, pdf_id, "failed", str(e))
                _publish_event(batch_id, pdf_failed(batch_id, str(e), pdf_name))

        complete_stage(batch_id, "extraction", f"Extracted {total_elements} elements from {len(filenames)} PDFs")
        update_stats(batch_id, total_elements=total_elements)

        if not all_chunks:
            logger.error("No chunks generated from any PDF")
            fail_stage(batch_id, "chunking", "No chunks generated from any PDF")
            _publish_event(batch_id, pdf_failed(batch_id, "No chunks generated from any PDF", None))
            return

        # ===== CHUNKING STAGE =====
        complete_stage(batch_id, "chunking", f"Created {len(all_chunks)} chunks")
        update_stats(batch_id, total_chunks=len(all_chunks))

        # POC: Save chunks to JSON
        _save_chunks_to_json(batch_id, all_chunks)

        # ===== INDEXING (handled in ingest_batch) =====
        logger.info(f"Status: indexing ({len(all_chunks)} chunks)")

        ingest_batch(
            batch_id=batch_id,
            all_chunks=all_chunks,
            embedding_dim=embedding_dim,
            ttl_seconds=ttl_seconds
        )

        # Publish completed event
        elapsed = time.time() - start_time
        _publish_event(batch_id, pdf_completed(batch_id, len(filenames), len(all_chunks), elapsed))

        logger.info("Batch completed successfully")

    except Exception as e:
        # Publish failed event for any unhandled error
        _publish_event(batch_id, pdf_failed(batch_id, str(e), None))
        raise

    finally:
        clear_batch_id()
