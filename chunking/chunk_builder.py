"""
Simple semantic chunking for PDF documents.

Chunk types: text, table, image
Strategy: Accumulate text until max_words, then start new chunk with overlap.
"""
import hashlib
from typing import List, Dict, Optional

from doc_analysis.config import ChunkingConfig, DEFAULT_CHUNKING_CONFIG


def _word_count(text: str) -> int:
    """Count words in text."""
    return len(text.split())


def _chunk_id(pdf_id: str, page_no: int, seq: int) -> str:
    """Generate deterministic chunk ID."""
    base = f"{pdf_id}|{page_no}|{seq}"
    return hashlib.sha256(base.encode("utf-8")).hexdigest()[:16]


def _merge_bbox(bbox_list: List[List[float]]) -> List[float]:
    """Merge multiple bboxes into one encompassing bbox."""
    valid = [b for b in bbox_list if b and len(b) >= 4]
    if not valid:
        return []

    return [
        min(b[0] for b in valid),  # x0
        min(b[1] for b in valid),  # y0
        max(b[2] for b in valid),  # x1
        max(b[3] for b in valid),  # y1
    ]


def _get_last_n_words(text: str, n: int) -> str:
    """Get the last n words from text."""
    words = text.split()
    if len(words) <= n:
        return text
    return " ".join(words[-n:])


def _split_table_by_rows(table_text: str, max_words: int) -> List[str]:
    """
    Split a large table into chunks by row groups, preserving header in each chunk.

    Args:
        table_text: Pipe-delimited table text with newline-separated rows
        max_words: Maximum words per chunk

    Returns:
        List of table text chunks, each with header row preserved
    """
    rows = table_text.strip().split("\n")

    if len(rows) <= 1:
        return [table_text]

    header = rows[0]
    data_rows = rows[1:]
    header_wc = _word_count(header)

    # If header alone exceeds limit, return as-is (edge case)
    if header_wc >= max_words:
        return [table_text]

    chunks = []
    current_rows = []
    current_wc = header_wc

    for row in data_rows:
        row_wc = _word_count(row)

        # If adding this row exceeds limit, flush current chunk
        if current_wc + row_wc > max_words and current_rows:
            chunk_text = header + "\n" + "\n".join(current_rows)
            chunks.append(chunk_text)
            current_rows = []
            current_wc = header_wc

        current_rows.append(row)
        current_wc += row_wc

    # Flush remaining rows
    if current_rows:
        chunk_text = header + "\n" + "\n".join(current_rows)
        chunks.append(chunk_text)

    return chunks if chunks else [table_text]


def build_chunks(
    elements: List[Dict],
    pdf_name: str,
    pdf_id: str,
    config: Optional[ChunkingConfig] = None
) -> List[Dict]:
    """
    Build chunks from document elements with overlap.

    Args:
        elements: List of elements with 'type', 'text', 'page_no', 'bbox'
        pdf_name: PDF filename
        pdf_id: Unique PDF identifier
        config: Chunking configuration

    Returns:
        List of chunks with metadata
    """
    if config is None:
        config = DEFAULT_CHUNKING_CONFIG

    chunks: List[Dict] = []
    chunk_seq = 0

    # Buffer for accumulating text
    buffer_texts: List[str] = []
    buffer_word_count = 0
    buffer_page: Optional[int] = None
    buffer_bboxes: List[List[float]] = []

    # Track previous chunk text for overlap
    prev_chunk_text: str = ""

    def flush_buffer(force: bool = False):
        """Create chunk from buffer with overlap from previous chunk."""
        nonlocal chunk_seq, buffer_texts, buffer_word_count, buffer_page, buffer_bboxes, prev_chunk_text

        if not buffer_texts:
            return

        # Skip if below minimum (unless forced)
        if buffer_word_count < config.min_words and not force:
            return

        text = " ".join(buffer_texts)

        # Add overlap from previous chunk (skip for first chunk)
        if prev_chunk_text and config.overlap_words > 0:
            overlap_text = _get_last_n_words(prev_chunk_text, config.overlap_words)
            text = overlap_text + " " + text

        chunk_seq += 1

        chunk = {
            "chunk_id": _chunk_id(pdf_id, buffer_page or 1, chunk_seq),
            "text": text,
            "page_no": buffer_page or 1,
            "pdf_id": pdf_id,
            "pdf_name": pdf_name,
            "content_type": "text",
            "chunk_seq": chunk_seq,
            "word_count": _word_count(text),
        }

        if config.include_bbox and buffer_bboxes:
            chunk["bbox"] = _merge_bbox(buffer_bboxes)

        chunks.append(chunk)

        # Store current chunk text (without overlap) for next chunk's overlap
        prev_chunk_text = " ".join(buffer_texts)

        # Clear buffer
        buffer_texts = []
        buffer_word_count = 0
        buffer_page = None
        buffer_bboxes = []

    for el in elements:
        el_type = el.get("type", "paragraph")
        text = el.get("text", "").strip()
        page_no = el.get("page_no", 1)
        bbox = el.get("bbox", [])

        if not text:
            continue

        # ----- IMAGE: standalone chunk -----
        if el_type == "image":
            flush_buffer(force=True)
            chunk_seq += 1

            chunk = {
                "chunk_id": _chunk_id(pdf_id, page_no, chunk_seq),
                "text": text or "[Image]",
                "page_no": page_no,
                "pdf_id": pdf_id,
                "pdf_name": pdf_name,
                "content_type": "image",
                "chunk_seq": chunk_seq,
                "word_count": _word_count(text),
            }
            if el.get("image_path"):
                chunk["image_path"] = el["image_path"]
                chunk["image_summary"] = ""  # Will be populated by vision model
            if config.include_bbox and bbox:
                chunk["bbox"] = bbox

            chunks.append(chunk)
            prev_chunk_text = ""  # Reset overlap after image
            continue

        # ----- TABLE: standalone chunk (split if exceeds max_words) -----
        if el_type == "table":
            flush_buffer(force=True)

            table_wc = _word_count(text)
            table_id = el.get("table_id", 0)  # Get table ID from element
            table_image_path = el.get("table_image_path", "")  # Image path for vision

            # Split large tables by row groups, preserving header
            if table_wc > config.max_words:
                table_chunks = _split_table_by_rows(text, config.max_words)
            else:
                table_chunks = [text]

            for i, table_text in enumerate(table_chunks):
                chunk_seq += 1

                chunk = {
                    "chunk_id": _chunk_id(pdf_id, page_no, chunk_seq),
                    "text": table_text,
                    "page_no": page_no,
                    "pdf_id": pdf_id,
                    "pdf_name": pdf_name,
                    "content_type": "table",
                    "chunk_seq": chunk_seq,
                    "word_count": _word_count(table_text),
                    "table_id": table_id,  # Which table in the PDF
                }

                # Add table part indicator for split tables
                if len(table_chunks) > 1:
                    chunk["table_part"] = i + 1
                    chunk["table_total_parts"] = len(table_chunks)

                # Add table image path and summary placeholder to FIRST chunk only
                if i == 0 and table_image_path:
                    chunk["table_image_path"] = table_image_path
                    chunk["table_summary"] = ""  # Will be populated by vision model

                if config.include_bbox and bbox:
                    chunk["bbox"] = bbox

                chunks.append(chunk)

            prev_chunk_text = ""  # Reset overlap after table
            continue

        # ----- TEXT: accumulate into buffer -----
        text_wc = _word_count(text)

        # If adding this text exceeds max, flush first
        if buffer_word_count + text_wc > config.max_words and buffer_texts:
            flush_buffer(force=True)

        # Add to buffer
        buffer_texts.append(text)
        buffer_word_count += text_wc
        buffer_page = buffer_page or page_no
        if bbox:
            buffer_bboxes.append(bbox)

        # Flush if at or above max
        if buffer_word_count >= config.max_words:
            flush_buffer(force=True)

    # Final flush
    flush_buffer(force=True)

    return chunks
