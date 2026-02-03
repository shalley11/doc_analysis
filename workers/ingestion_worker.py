"""
Simple batch ingestion worker for Milvus.
"""
import json
from typing import List

from pymilvus.exceptions import MilvusException

from doc_analysis.logging_config import get_embedding_logger, set_batch_id
from doc_analysis.jobs.job_store import (
    update_batch_status,
    start_stage,
    complete_stage,
    skip_stage,
    fail_stage,
    update_stats
)
from doc_analysis.embedding.e5_embedder import embed_passages
from doc_analysis.vector_store.milvus_utils import create_temp_collection
from doc_analysis.vector_store.milvus_store import MilvusStore
from doc_analysis.vision.vision_worker import batch_table_summaries_parallel, batch_image_summaries_parallel
from doc_analysis.config import ENABLE_TABLE_VISION, ENABLE_IMAGE_VISION
from doc_analysis.realtime import (
    publish_pdf_event,
    pdf_embedding,
    pdf_embedded,
    pdf_failed
)

logger = get_embedding_logger()


def _publish_event(batch_id: str, event):
    """Safely publish event, logging any errors."""
    try:
        publish_pdf_event(batch_id, event)
    except Exception as e:
        logger.warning(f"[EVENT] Failed to publish event: {e}")


def _safe_collection_name(batch_id: str) -> str:
    return f"batch_{batch_id.replace('-', '_')}"


def ingest_batch(
    batch_id: str,
    all_chunks: List[dict],
    embedding_dim: int,
    ttl_seconds: int,
):
    """
    Ingest chunks into Milvus vector store.

    Args:
        batch_id: Unique batch identifier
        all_chunks: List of chunk dictionaries
        embedding_dim: Embedding dimension
        ttl_seconds: TTL for collection
    """
    set_batch_id(batch_id)

    try:
        logger.info(f"Starting batch indexing with {len(all_chunks)} chunks")

        if not all_chunks:
            logger.error("No chunks to index")
            fail_stage(batch_id, "chunking", "No chunks generated")
            return

        # Update stats
        update_stats(batch_id, total_chunks=len(all_chunks))

        # Create Milvus collection
        collection_name = _safe_collection_name(batch_id)
        collection = create_temp_collection(
            collection_name=collection_name,
            dim=embedding_dim,
            ttl_seconds=ttl_seconds,
        )
        logger.info(f"Milvus collection created: {collection_name}")

        store = MilvusStore(collection)

        # ===== VISION: TABLES =====
        table_summaries = {}
        image_summaries = {}

        # Collect all table and image paths
        table_images_to_process = []
        image_paths_to_process = []

        for i, chunk in enumerate(all_chunks):
            table_image_path = chunk.get("table_image_path", "")
            if table_image_path:
                table_images_to_process.append((i, table_image_path))

            if chunk.get("content_type") == "image":
                image_path = chunk.get("image_path", "")
                if image_path:
                    image_paths_to_process.append((i, image_path))

        # ===== VISION: TABLES =====
        if ENABLE_TABLE_VISION:
            if table_images_to_process:
                start_stage(batch_id, "vision_tables", f"Processing {len(table_images_to_process)} tables")
                logger.info(f"Processing {len(table_images_to_process)} table images with vision model...")

                table_summaries = batch_table_summaries_parallel(table_images_to_process)

                for idx, summary in table_summaries.items():
                    all_chunks[idx]["table_summary"] = summary

                complete_stage(batch_id, "vision_tables", f"Processed {len(table_summaries)} tables")
                update_stats(batch_id, tables_processed=len(table_summaries))
                logger.info(f"Completed vision processing for {len(table_summaries)} tables")
            else:
                skip_stage(batch_id, "vision_tables", "No table images found")
                logger.info("No table images to process")
        else:
            skip_stage(batch_id, "vision_tables", "Table vision processing disabled")
            logger.info("Table vision processing DISABLED (ENABLE_TABLE_VISION=False)")

        # ===== VISION: IMAGES =====
        if ENABLE_IMAGE_VISION:
            if image_paths_to_process:
                start_stage(batch_id, "vision_images", f"Processing {len(image_paths_to_process)} images")
                logger.info(f"Processing {len(image_paths_to_process)} document images with vision model...")

                image_summaries = batch_image_summaries_parallel(image_paths_to_process)

                for idx, summary in image_summaries.items():
                    all_chunks[idx]["image_summary"] = summary

                complete_stage(batch_id, "vision_images", f"Processed {len(image_summaries)} images")
                update_stats(batch_id, images_processed=len(image_summaries))
                logger.info(f"Completed vision processing for {len(image_summaries)} images")
            else:
                skip_stage(batch_id, "vision_images", "No document images found")
                logger.info("No document images to process")
        else:
            skip_stage(batch_id, "vision_images", "Image vision processing disabled")
            logger.info("Image vision processing DISABLED (ENABLE_IMAGE_VISION=False)")

        # ===== EMBEDDING GENERATION =====
        start_stage(batch_id, "embedding", f"Generating embeddings for {len(all_chunks)} chunks")
        logger.info(f"Generating embeddings for {len(all_chunks)} chunks...")

        # Publish embedding started event
        _publish_event(batch_id, pdf_embedding(batch_id, "all_chunks", 0, len(all_chunks)))

        texts = [f"passage: {c['text']}" for c in all_chunks]
        embeddings = embed_passages(texts, batch_size=8)
        logger.info(f"Generated {len(embeddings)} embeddings")

        # Publish embedding progress
        _publish_event(batch_id, pdf_embedding(batch_id, "all_chunks", len(embeddings), len(all_chunks)))

        if len(embeddings) != len(all_chunks):
            fail_stage(batch_id, "embedding", "Embedding count mismatch")
            _publish_event(batch_id, pdf_failed(batch_id, "Embedding count mismatch", None))
            raise RuntimeError("Embedding count mismatch")

        total_embeddings = len(embeddings)

        # Merge table and image summaries into unified summary dict
        # Key: chunk index, Value: summary text
        all_summaries = {}
        for idx, summary in table_summaries.items():
            if summary:
                all_summaries[idx] = summary
        for idx, summary in image_summaries.items():
            if summary:
                all_summaries[idx] = summary

        # Generate embeddings for all summaries (table + image)
        logger.info(f"Generating embeddings for {len(all_summaries)} summaries...")
        summary_embeddings = {}
        if all_summaries:
            summary_texts = []
            summary_indices = []
            for idx, summary in all_summaries.items():
                summary_texts.append(f"passage: {summary}")
                summary_indices.append(idx)

            if summary_texts:
                summary_embs = embed_passages(summary_texts, batch_size=8)
                for idx, emb in zip(summary_indices, summary_embs):
                    summary_embeddings[idx] = emb
                total_embeddings += len(summary_embeddings)
                logger.info(f"Generated {len(summary_embeddings)} summary embeddings")

        complete_stage(batch_id, "embedding", f"Generated {total_embeddings} total embeddings")
        update_stats(batch_id, embeddings_generated=total_embeddings)

        # Publish embedded event
        _publish_event(batch_id, pdf_embedded(batch_id, "all_chunks", total_embeddings))

        # Zero vector for chunks without summary
        zero_vector = [0.0] * embedding_dim

        # Build chunk_id list for linking prev/next
        chunk_ids = [chunk["chunk_id"] for chunk in all_chunks]

        # Prepare records with prev/next chunk linking
        records = []
        for i, (chunk, emb) in enumerate(zip(all_chunks, embeddings)):
            # Determine prev and next chunk IDs
            prev_chunk_id = chunk_ids[i - 1] if i > 0 else ""
            next_chunk_id = chunk_ids[i + 1] if i < len(chunk_ids) - 1 else ""

            # Get summary (from table or image, based on content_type)
            summary = all_summaries.get(i, "")

            # Get image_path (could be from table_image_path or image_path)
            image_path = chunk.get("image_path", "") or chunk.get("table_image_path", "") or ""

            record = {
                "chunk_id": chunk["chunk_id"],
                "embedding": emb,
                "text": chunk["text"],
                "pdf_id": chunk["pdf_id"],
                "pdf_name": chunk["pdf_name"],
                "page_no": chunk["page_no"],
                "chunk_seq": chunk["chunk_seq"],
                "prev_chunk_id": prev_chunk_id,
                "next_chunk_id": next_chunk_id,
                "content_type": chunk["content_type"],
                "word_count": chunk.get("word_count", 0),
                "image_path": image_path,
                "table_id": chunk.get("table_id", 0),
                "table_part": chunk.get("table_part", 0),
                "table_total_parts": chunk.get("table_total_parts", 0),
                "summary": summary,
                "summary_embedding": summary_embeddings.get(i, zero_vector),
            }
            records.append(record)

        # ===== INDEXING (MILVUS INSERT) =====
        start_stage(batch_id, "indexing", f"Inserting {len(records)} records into Milvus")
        logger.info(f"Inserting {len(records)} records into Milvus...")

        store.insert_chunks(records)

        complete_stage(batch_id, "indexing", f"Inserted {len(records)} records")
        logger.info(f"Successfully inserted {len(records)} records")

        # ===== COMPLETED =====
        complete_stage(batch_id, "completed", "All processing completed successfully")
        update_batch_status(batch_id, "completed")
        logger.info("Batch indexing completed successfully")

    except MilvusException as e:
        logger.error(f"Milvus error: {str(e)}", exc_info=True)
        fail_stage(batch_id, "indexing", str(e))
        _publish_event(batch_id, pdf_failed(batch_id, f"Milvus error: {str(e)}", None))
        raise

    except Exception as e:
        logger.error(f"Indexing failed: {str(e)}", exc_info=True)
        update_batch_status(batch_id, "failed", str(e))
        _publish_event(batch_id, pdf_failed(batch_id, str(e), None))
        raise
