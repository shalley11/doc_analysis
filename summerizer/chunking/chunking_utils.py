# chunking_utils.py
import re
import hashlib
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


# =============================================================================
# Constants for improved sentence splitting
# =============================================================================

# Common abbreviations that should not trigger sentence splits
ABBREVIATIONS = {
    'e.g.', 'i.e.', 'etc.', 'vs.', 'viz.', 'cf.', 'al.',
    'Dr.', 'Mr.', 'Mrs.', 'Ms.', 'Prof.', 'Sr.', 'Jr.',
    'Ph.D.', 'M.D.', 'B.A.', 'M.A.', 'D.D.S.',
    'Inc.', 'Ltd.', 'Corp.', 'Co.',
    'Jan.', 'Feb.', 'Mar.', 'Apr.', 'Jun.', 'Jul.', 'Aug.', 'Sep.', 'Oct.', 'Nov.', 'Dec.',
    'Fig.', 'fig.', 'Figs.', 'figs.', 'No.', 'no.', 'Vol.', 'vol.', 'pp.', 'p.',
    'approx.', 'est.', 'ca.', 'c.'
}

# Patterns for detecting structured content
LIST_BULLET_PATTERNS = [
    r'^[\-\•\*\‣\◦\▪\▸]\s+',           # Bullet points
    r'^\d+[\.\)]\s+',                    # Numbered lists (1. or 1))
    r'^[a-z][\.\)]\s+',                  # Letter lists (a. or a))
    r'^[ivxIVX]+[\.\)]\s+',              # Roman numeral lists
    r'^\([a-z\d]+\)\s+',                 # Parenthetical lists ((a) or (1))
]

# Pattern for detecting section headings
HEADING_PATTERNS = [
    r'^(I{1,3}V?|V?I{0,3}|IV|IX|VI{0,3})\.\s+[A-Z]',  # Roman numeral headings
    r'^\d+\.\s+[A-Z][a-z]+',                            # Numbered headings
    r'^(Executive Summary|Table of Contents|References|Acknowledgements|Abstract)\s*$',
    r'^(Figure|Table)\s+\d+[\.:]',                      # Figure/Table captions
]

# Pattern for detecting tables and figures
TABLE_FIGURE_PATTERNS = [
    r'^(Figure|Table)\s+\d+[\.\:]',
    r'^\|(.*\|)+\s*$',                                  # Markdown table rows
    r'^[\+\-]{3,}',                                     # Table borders
]


class ContentType(Enum):
    TEXT = "text"
    LIST = "list"
    TABLE = "table"
    FIGURE = "figure"
    HEADING = "heading"


@dataclass
class ContentBlock:
    """Represents a structured content block."""
    content_type: ContentType
    text: str
    position: int
    items: List[str] = None  # For lists
    caption: str = None      # For tables/figures

    def __post_init__(self):
        if self.items is None:
            self.items = []


# =============================================================================
# Improved Text Processing Functions
# =============================================================================

def normalize_text(text: str) -> str:
    """
    Normalize text for deterministic hashing.
    - collapse whitespace
    - lowercase
    """
    return " ".join(text.split()).lower()


def deterministic_chunk_id(pdf_name: str, page_no: int, text: str, content_type: str = "text") -> str:
    """
    Generate deterministic chunk ID.
    Same content => same ID (idempotent).
    """
    normalized = normalize_text(text)
    raw = f"{pdf_name}:{page_no}:{content_type}:{normalized}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def protect_abbreviations(text: str) -> str:
    """Protect abbreviations from being split at sentence boundaries."""
    protected = text
    for abbr in ABBREVIATIONS:
        # Replace the period in abbreviation with a placeholder
        protected_abbr = abbr.replace('.', '<ABBR_DOT>')
        protected = protected.replace(abbr, protected_abbr)
    return protected


def restore_abbreviations(text: str) -> str:
    """Restore protected abbreviations."""
    return text.replace('<ABBR_DOT>', '.')


def protect_special_patterns(text: str) -> str:
    """Protect special patterns from sentence splitting."""
    protected = text

    # Protect footnote references (e.g., "contexts.1" or "tools2'3")
    protected = re.sub(r'\.(\d+)', r'<FN_DOT>\1', protected)

    # Protect URLs
    protected = re.sub(r'(https?://[^\s]+)', lambda m: m.group(1).replace('.', '<URL_DOT>'), protected)

    # Protect decimal numbers
    protected = re.sub(r'(\d)\.(\d)', r'\1<DEC_DOT>\2', protected)

    return protected


def restore_special_patterns(text: str) -> str:
    """Restore protected special patterns."""
    text = text.replace('<FN_DOT>', '.')
    text = text.replace('<URL_DOT>', '.')
    text = text.replace('<DEC_DOT>', '.')
    return text


def sentence_split(text: str) -> List[str]:
    """
    Improved sentence splitter with abbreviation handling.
    Handles common abbreviations, footnotes, URLs, and decimal numbers.
    """
    if not text or not text.strip():
        return []

    # Protect special patterns
    protected = protect_abbreviations(text)
    protected = protect_special_patterns(protected)

    # Split on sentence boundaries
    sentences = re.split(r'(?<=[.!?])\s+', protected.strip())

    # Restore protected patterns and filter empty
    restored = []
    for s in sentences:
        s = restore_abbreviations(s)
        s = restore_special_patterns(s)
        if s.strip():
            restored.append(s.strip())

    return restored


def is_list_item(line: str) -> bool:
    """Check if a line is a list item."""
    line = line.strip()
    for pattern in LIST_BULLET_PATTERNS:
        if re.match(pattern, line):
            return True
    return False


def is_heading(line: str) -> bool:
    """Check if a line is a section heading."""
    line = line.strip()
    for pattern in HEADING_PATTERNS:
        if re.match(pattern, line, re.IGNORECASE):
            return True
    return False


def is_table_or_figure(text: str) -> Tuple[bool, str]:
    """
    Check if text is a table or figure.
    Returns (is_table_or_figure, type).
    """
    text = text.strip()

    # Check for Figure caption
    if re.match(r'^Figure\s+\d+', text, re.IGNORECASE):
        return True, "figure"

    # Check for Table caption
    if re.match(r'^Table\s+\d+', text, re.IGNORECASE):
        return True, "table"

    # Check for table-like structure (multiple | characters)
    if text.count('|') >= 2:
        return True, "table"

    return False, None


# =============================================================================
# Structure Detection and Parsing
# =============================================================================

def detect_content_structure(text: str) -> List[ContentBlock]:
    """
    Parse text into structured content blocks.
    Detects headings, lists, tables/figures, and regular text.
    """
    if not text:
        return []

    blocks = []
    lines = text.split('\n')
    current_block_lines = []
    current_type = ContentType.TEXT
    position = 0

    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        if not stripped:
            # Empty line - might signal end of block
            if current_block_lines:
                blocks.append(_create_block(current_type, current_block_lines, position))
                position += 1
                current_block_lines = []
                current_type = ContentType.TEXT
            i += 1
            continue

        # Check for table/figure
        is_tf, tf_type = is_table_or_figure(stripped)
        if is_tf:
            # Save current block if any
            if current_block_lines:
                blocks.append(_create_block(current_type, current_block_lines, position))
                position += 1
                current_block_lines = []

            # Collect entire table/figure block
            table_lines = [stripped]
            i += 1
            # Continue collecting until empty line or new section
            while i < len(lines):
                next_line = lines[i].strip()
                if not next_line:
                    break
                # Check if this is a new heading (end of table)
                if is_heading(next_line):
                    break
                table_lines.append(next_line)
                i += 1

            content_type = ContentType.TABLE if tf_type == "table" else ContentType.FIGURE
            blocks.append(_create_block(content_type, table_lines, position))
            position += 1
            current_type = ContentType.TEXT
            continue

        # Check for heading
        if is_heading(stripped):
            # Save current block if any
            if current_block_lines:
                blocks.append(_create_block(current_type, current_block_lines, position))
                position += 1
                current_block_lines = []

            blocks.append(_create_block(ContentType.HEADING, [stripped], position))
            position += 1
            current_type = ContentType.TEXT
            i += 1
            continue

        # Check for list item
        if is_list_item(stripped):
            # If current block is not a list, save it and start a list
            if current_type != ContentType.LIST:
                if current_block_lines:
                    blocks.append(_create_block(current_type, current_block_lines, position))
                    position += 1
                current_block_lines = []
                current_type = ContentType.LIST

            current_block_lines.append(stripped)
            i += 1
            continue

        # Regular text
        if current_type == ContentType.LIST:
            # End of list - save it
            if current_block_lines:
                blocks.append(_create_block(current_type, current_block_lines, position))
                position += 1
            current_block_lines = []
            current_type = ContentType.TEXT

        current_block_lines.append(stripped)
        i += 1

    # Save final block
    if current_block_lines:
        blocks.append(_create_block(current_type, current_block_lines, position))

    return blocks


def _create_block(content_type: ContentType, lines: List[str], position: int) -> ContentBlock:
    """Create a ContentBlock from lines."""
    text = ' '.join(lines) if content_type != ContentType.LIST else '\n'.join(lines)

    block = ContentBlock(
        content_type=content_type,
        text=text,
        position=position,
        items=lines if content_type == ContentType.LIST else None
    )

    return block


# =============================================================================
# Cross-Page Merging
# =============================================================================

def merge_cross_page_content(pages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Merge content that spans page boundaries.

    Detects incomplete sentences at page end and merges with beginning of next page.
    Also merges split lists and tables.

    Args:
        pages: List of page results, each with 'text', 'page_no', etc.

    Returns:
        List of merged page results with adjusted content.
    """
    if not pages or len(pages) <= 1:
        return pages

    merged_pages = []
    carry_over_text = ""
    carry_over_type = None  # Track if we're in the middle of a list

    for i, page in enumerate(pages):
        page_copy = page.copy()
        text = page.get("text", "") or ""
        blocks = page.get("blocks", [])

        # If we have carry-over text from previous page, prepend it
        if carry_over_text:
            if blocks:
                # Prepend to first text block
                for block in blocks:
                    if block.get("type") == "text":
                        block["content"] = carry_over_text + " " + block.get("content", "")
                        break
                else:
                    # No text block found, insert one at the beginning
                    blocks.insert(0, {
                        "type": "text",
                        "content": carry_over_text,
                        "position": 0
                    })
            else:
                text = carry_over_text + " " + text

            page_copy["text"] = text
            page_copy["blocks"] = blocks
            page_copy["metadata"] = page_copy.get("metadata", {})
            page_copy["metadata"]["merged_from_previous"] = True
            carry_over_text = ""

        # Check if this page ends with an incomplete sentence
        if i < len(pages) - 1:  # Not the last page
            incomplete, remaining = _check_incomplete_ending(text, blocks)

            if incomplete and remaining:
                carry_over_text = remaining
                # Remove the incomplete text from this page
                if blocks:
                    for block in reversed(blocks):
                        if block.get("type") == "text":
                            content = block.get("content", "")
                            if content.endswith(remaining):
                                block["content"] = content[:-len(remaining)].rstrip()
                            break
                else:
                    page_copy["text"] = text[:-len(remaining)].rstrip()

                page_copy["metadata"] = page_copy.get("metadata", {})
                page_copy["metadata"]["continues_to_next"] = True

        merged_pages.append(page_copy)

    return merged_pages


def _check_incomplete_ending(text: str, blocks: List[Dict]) -> Tuple[bool, str]:
    """
    Check if page ends with an incomplete sentence.

    Returns:
        (is_incomplete, text_to_carry_over)
    """
    # Get the last text content
    last_text = ""
    if blocks:
        for block in reversed(blocks):
            if block.get("type") == "text" and block.get("content"):
                last_text = block.get("content", "").strip()
                break
    else:
        last_text = text.strip() if text else ""

    if not last_text:
        return False, ""

    # Check if ends with sentence-ending punctuation
    if last_text[-1] in '.!?:':
        return False, ""

    # Check if ends mid-word or mid-sentence
    # Find the last complete sentence
    sentences = sentence_split(last_text)
    if not sentences:
        return True, last_text

    # If the last "sentence" doesn't end with proper punctuation
    last_sentence = sentences[-1]
    if last_sentence and last_sentence[-1] not in '.!?:':
        return True, last_sentence

    return False, ""


def merge_split_lists(pages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Merge lists that are split across page boundaries.

    Detects when a page ends with a list item and next page starts with one,
    and merges them into a single list.
    """
    if not pages or len(pages) <= 1:
        return pages

    merged_pages = []

    for i, page in enumerate(pages):
        page_copy = page.copy()
        blocks = page.get("blocks", [])

        if i > 0 and blocks:
            # Check if previous page ended with a list and this one starts with list items
            prev_page = merged_pages[-1]
            prev_blocks = prev_page.get("blocks", [])

            if prev_blocks and blocks:
                # Check if previous ends with list and current starts with list
                prev_last = prev_blocks[-1] if prev_blocks else None
                curr_first = blocks[0] if blocks else None

                if (prev_last and curr_first and
                    prev_last.get("type") == "list" and
                    (curr_first.get("type") == "list" or
                     is_list_item(str(curr_first.get("content", "")).split('\n')[0]))):

                    # Merge the lists
                    prev_content = prev_last.get("content", "")
                    curr_content = curr_first.get("content", "")
                    prev_last["content"] = prev_content + "\n" + curr_content
                    prev_last["metadata"] = prev_last.get("metadata", {})
                    prev_last["metadata"]["merged_list"] = True

                    # Remove the first block from current page
                    page_copy["blocks"] = blocks[1:]

        merged_pages.append(page_copy)

    return merged_pages


# =============================================================================
# List-Aware Chunking
# =============================================================================

def extract_lists_from_text(text: str) -> List[Dict[str, Any]]:
    """
    Extract lists from text, keeping related items together.

    Returns:
        List of blocks with type 'text' or 'list'.
    """
    if not text:
        return []

    lines = text.split('\n')
    blocks = []
    current_text_lines = []
    current_list_items = []
    position = 0

    for line in lines:
        stripped = line.strip()

        if not stripped:
            # Empty line - might end current block
            if current_list_items:
                blocks.append({
                    "type": "list",
                    "content": '\n'.join(current_list_items),
                    "items": current_list_items.copy(),
                    "position": position,
                    "metadata": {"item_count": len(current_list_items)}
                })
                position += 1
                current_list_items = []
            elif current_text_lines:
                blocks.append({
                    "type": "text",
                    "content": ' '.join(current_text_lines),
                    "position": position,
                    "metadata": {}
                })
                position += 1
                current_text_lines = []
            continue

        if is_list_item(stripped):
            # End current text block if any
            if current_text_lines:
                blocks.append({
                    "type": "text",
                    "content": ' '.join(current_text_lines),
                    "position": position,
                    "metadata": {}
                })
                position += 1
                current_text_lines = []

            current_list_items.append(stripped)
        else:
            # End current list block if any
            if current_list_items:
                blocks.append({
                    "type": "list",
                    "content": '\n'.join(current_list_items),
                    "items": current_list_items.copy(),
                    "position": position,
                    "metadata": {"item_count": len(current_list_items)}
                })
                position += 1
                current_list_items = []

            current_text_lines.append(stripped)

    # Handle remaining content
    if current_list_items:
        blocks.append({
            "type": "list",
            "content": '\n'.join(current_list_items),
            "items": current_list_items.copy(),
            "position": position,
            "metadata": {"item_count": len(current_list_items)}
        })
    elif current_text_lines:
        blocks.append({
            "type": "text",
            "content": ' '.join(current_text_lines),
            "position": position,
            "metadata": {}
        })

    return blocks


def chunk_list_block(
    list_block: Dict[str, Any],
    pdf_name: str,
    page_no: int,
    max_words: int = 500,
    start_chunk_number: int = 0
) -> List[Dict[str, Any]]:
    """
    Chunk a list block while keeping related items together.

    Lists are kept as single chunks unless they exceed max_words,
    in which case they are split at logical boundaries (between items).
    """
    items = list_block.get("items", [])
    content = list_block.get("content", "")
    position = list_block.get("position", 0)

    if not items:
        items = content.split('\n')

    total_words = len(content.split())

    # If list fits in one chunk, keep it together
    if total_words <= max_words:
        return [{
            "chunk_id": deterministic_chunk_id(pdf_name, page_no, content, "list"),
            "content_type": "list",
            "text": content,
            "pdf_name": pdf_name,
            "page_no": page_no,
            "position": position,
            "chunk_number": start_chunk_number,
            "image_link": "",
            "table_link": "",
            "context_before_id": "",
            "context_after_id": "",
            "metadata": {
                "chunking_method": "list_aware",
                "item_count": len(items),
                "is_complete_list": True
            }
        }]

    # Split list into multiple chunks
    chunks = []
    current_items = []
    current_word_count = 0
    chunk_number = start_chunk_number
    sub_position = 0

    for item in items:
        item_words = len(item.split())

        if current_word_count + item_words > max_words and current_items:
            # Save current chunk
            chunk_text = '\n'.join(current_items)
            chunks.append({
                "chunk_id": deterministic_chunk_id(pdf_name, page_no, chunk_text, "list"),
                "content_type": "list",
                "text": chunk_text,
                "pdf_name": pdf_name,
                "page_no": page_no,
                "position": position + (sub_position * 0.01),
                "chunk_number": chunk_number,
                "image_link": "",
                "table_link": "",
                "context_before_id": "",
                "context_after_id": "",
                "metadata": {
                    "chunking_method": "list_aware",
                    "item_count": len(current_items),
                    "is_complete_list": False,
                    "list_part": sub_position + 1
                }
            })
            chunk_number += 1
            sub_position += 1
            current_items = []
            current_word_count = 0

        current_items.append(item)
        current_word_count += item_words

    # Save final chunk
    if current_items:
        chunk_text = '\n'.join(current_items)
        chunks.append({
            "chunk_id": deterministic_chunk_id(pdf_name, page_no, chunk_text, "list"),
            "content_type": "list",
            "text": chunk_text,
            "pdf_name": pdf_name,
            "page_no": page_no,
            "position": position + (sub_position * 0.01),
            "chunk_number": chunk_number,
            "image_link": "",
            "table_link": "",
            "context_before_id": "",
            "context_after_id": "",
            "metadata": {
                "chunking_method": "list_aware",
                "item_count": len(current_items),
                "is_complete_list": False,
                "list_part": sub_position + 1
            }
        })

    return chunks


# =============================================================================
# Table/Figure Extraction
# =============================================================================

def extract_tables_and_figures(text: str) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Extract tables and figures from text as separate chunks.

    Returns:
        (remaining_text, list_of_table_figure_blocks)
    """
    if not text:
        return "", []

    extracted = []
    remaining_lines = []
    lines = text.split('\n')

    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        # Check for Figure caption
        figure_match = re.match(r'^(Figure\s+\d+[\.\:]?\s*.*?)$', stripped, re.IGNORECASE)
        if figure_match:
            figure_lines = [stripped]
            i += 1
            # Collect figure content until empty line or new section
            while i < len(lines):
                next_line = lines[i].strip()
                if not next_line:
                    break
                if is_heading(next_line) or re.match(r'^(Figure|Table)\s+\d+', next_line, re.IGNORECASE):
                    break
                figure_lines.append(next_line)
                i += 1

            extracted.append({
                "type": "figure",
                "content": '\n'.join(figure_lines),
                "caption": figure_match.group(1),
                "position": len(extracted),
                "metadata": {"is_figure": True}
            })
            continue

        # Check for Table caption
        table_match = re.match(r'^(Table\s+\d+[\.\:]?\s*.*?)$', stripped, re.IGNORECASE)
        if table_match:
            table_lines = [stripped]
            i += 1
            # Collect table content until empty line or new section
            while i < len(lines):
                next_line = lines[i].strip()
                if not next_line:
                    # Check if next non-empty line is still part of table
                    lookahead = i + 1
                    while lookahead < len(lines) and not lines[lookahead].strip():
                        lookahead += 1
                    if lookahead < len(lines):
                        next_content = lines[lookahead].strip()
                        # If it looks like table data, continue
                        if '|' in next_content or next_content.startswith(('-', '+')):
                            table_lines.append('')
                            i += 1
                            continue
                    break
                if is_heading(next_line):
                    break
                if re.match(r'^(Figure|Table)\s+\d+', next_line, re.IGNORECASE) and next_line != stripped:
                    break
                table_lines.append(next_line)
                i += 1

            extracted.append({
                "type": "table",
                "content": '\n'.join(table_lines),
                "caption": table_match.group(1),
                "position": len(extracted),
                "metadata": {"is_table": True}
            })
            continue

        remaining_lines.append(line)
        i += 1

    remaining_text = '\n'.join(remaining_lines)
    return remaining_text, extracted


def chunk_table_or_figure(
    block: Dict[str, Any],
    pdf_name: str,
    page_no: int,
    chunk_number: int
) -> Dict[str, Any]:
    """
    Create a chunk for a table or figure.
    Tables and figures are always kept as single chunks.
    """
    content = block.get("content", "")
    block_type = block.get("type", "table")
    caption = block.get("caption", "")
    position = block.get("position", 0)

    return {
        "chunk_id": deterministic_chunk_id(pdf_name, page_no, content, block_type),
        "content_type": block_type,
        "text": content,
        "pdf_name": pdf_name,
        "page_no": page_no,
        "position": position,
        "chunk_number": chunk_number,
        "image_link": block.get("image_link", ""),
        "table_link": block.get("table_link", ""),
        "context_before_id": "",
        "context_after_id": "",
        "metadata": {
            "chunking_method": "table_figure_extraction",
            "caption": caption,
            f"is_{block_type}": True
        }
    }


# =============================================================================
# Enhanced Document Processor
# =============================================================================

def merge_small_chunks_final(
    chunks: List[Dict[str, Any]],
    min_words: int = 20
) -> List[Dict[str, Any]]:
    """
    Merge chunks that are smaller than min_words with adjacent chunks.

    This is a final pass to clean up very small chunks (like page numbers,
    headers, footers) that may result from PDF extraction.

    Args:
        chunks: List of chunk dictionaries
        min_words: Minimum words per chunk (default 20)

    Returns:
        List of merged chunks
    """
    if not chunks or len(chunks) <= 1:
        return chunks

    merged = []
    pending_small = None

    for chunk in chunks:
        word_count = len(chunk.get("text", "").split())

        if word_count < min_words:
            # This is a small chunk
            if pending_small is None:
                pending_small = chunk
            else:
                # Merge with pending small chunk
                pending_small["text"] = pending_small.get("text", "") + " " + chunk.get("text", "")
                pending_small["metadata"] = pending_small.get("metadata", {})
                pending_small["metadata"]["merged_small_chunks"] = True
        else:
            # Normal-sized chunk
            if pending_small is not None:
                # Merge pending small with this chunk
                chunk["text"] = pending_small.get("text", "") + " " + chunk.get("text", "")
                chunk["metadata"] = chunk.get("metadata", {})
                chunk["metadata"]["absorbed_small_chunk"] = True
                # Recalculate chunk_id
                chunk["chunk_id"] = deterministic_chunk_id(
                    chunk.get("pdf_name", ""),
                    chunk.get("page_no", 0),
                    chunk.get("text", ""),
                    chunk.get("content_type", "text")
                )
                pending_small = None
            merged.append(chunk)

    # Handle any remaining small chunk
    if pending_small is not None:
        if merged:
            # Merge with last chunk
            merged[-1]["text"] = merged[-1].get("text", "") + " " + pending_small.get("text", "")
            merged[-1]["metadata"] = merged[-1].get("metadata", {})
            merged[-1]["metadata"]["absorbed_trailing_chunk"] = True
            merged[-1]["chunk_id"] = deterministic_chunk_id(
                merged[-1].get("pdf_name", ""),
                merged[-1].get("page_no", 0),
                merged[-1].get("text", ""),
                merged[-1].get("content_type", "text")
            )
        else:
            # Only small chunks, keep it
            merged.append(pending_small)

    # Renumber chunks
    for i, chunk in enumerate(merged):
        chunk["chunk_number"] = i

    return merged


def process_document_pages(
    pages: List[Dict[str, Any]],
    pdf_name: str,
    max_words: int = 500,
    min_words: int = 20,
    enable_cross_page_merge: bool = True,
    enable_list_aware: bool = True,
    enable_table_figure_extraction: bool = True
) -> List[Dict[str, Any]]:
    """
    Process all pages of a document with enhanced chunking.

    This is the main entry point for improved document chunking with:
    - Cross-page content merging
    - List-aware chunking
    - Table/figure extraction
    - Minimum chunk size enforcement

    Args:
        pages: List of page dictionaries with 'text', 'page_no', 'blocks', etc.
        pdf_name: Name of the PDF document
        max_words: Maximum words per chunk (default 500)
        min_words: Minimum words per chunk - smaller chunks are merged (default 20)
        enable_cross_page_merge: Enable cross-page content merging
        enable_list_aware: Enable list-aware chunking
        enable_table_figure_extraction: Enable table/figure extraction

    Returns:
        List of chunk dictionaries ready for embedding and indexing
    """
    # Step 1: Cross-page merging
    if enable_cross_page_merge:
        pages = merge_cross_page_content(pages)
        pages = merge_split_lists(pages)

    all_chunks = []
    chunk_number = 0

    for page in pages:
        page_no = page.get("page_no", page.get("metadata", {}).get("page_no", 0))
        text = page.get("text", "")
        blocks = page.get("blocks", [])

        # If blocks are provided, use them
        if blocks:
            for block in blocks:
                block_type = block.get("type", "text")
                content = block.get("content", "")
                position = block.get("position", 0)

                if block_type in ("table", "figure"):
                    # Table/figure as single chunk
                    chunk = chunk_table_or_figure(block, pdf_name, page_no, chunk_number)
                    all_chunks.append(chunk)
                    chunk_number += 1

                elif block_type == "list" and enable_list_aware:
                    # List-aware chunking
                    list_chunks = chunk_list_block(
                        block, pdf_name, page_no, max_words, chunk_number
                    )
                    all_chunks.extend(list_chunks)
                    chunk_number += len(list_chunks)

                elif block_type == "text" and content:
                    # Process text with structure detection
                    text_chunks = _chunk_text_with_structure(
                        content, pdf_name, page_no, position, max_words,
                        chunk_number, enable_list_aware, enable_table_figure_extraction
                    )
                    all_chunks.extend(text_chunks)
                    chunk_number += len(text_chunks)

                elif block_type == "image":
                    # Image as single chunk
                    chunk = chunk_table_or_figure(
                        {**block, "type": "image"},
                        pdf_name, page_no, chunk_number
                    )
                    all_chunks.append(chunk)
                    chunk_number += 1

        elif text:
            # Process raw text
            text_chunks = _chunk_text_with_structure(
                text, pdf_name, page_no, 0, max_words,
                chunk_number, enable_list_aware, enable_table_figure_extraction
            )
            all_chunks.extend(text_chunks)
            chunk_number += len(text_chunks)

    # Step: Merge small chunks (remove artifacts like page numbers)
    all_chunks = merge_small_chunks_final(all_chunks, min_words)

    # Link context
    all_chunks = _link_chunk_context(all_chunks)

    return all_chunks


def _chunk_text_with_structure(
    text: str,
    pdf_name: str,
    page_no: int,
    base_position: int,
    max_words: int,
    start_chunk_number: int,
    enable_list_aware: bool = True,
    enable_table_figure_extraction: bool = True
) -> List[Dict[str, Any]]:
    """
    Chunk text with structure awareness (lists, tables, figures).
    """
    chunks = []
    chunk_number = start_chunk_number
    position = base_position

    # Step 1: Extract tables and figures
    remaining_text = text
    table_figure_blocks = []

    if enable_table_figure_extraction:
        remaining_text, table_figure_blocks = extract_tables_and_figures(text)

    # Step 2: Extract lists from remaining text
    if enable_list_aware:
        content_blocks = extract_lists_from_text(remaining_text)
    else:
        content_blocks = [{
            "type": "text",
            "content": remaining_text,
            "position": 0,
            "metadata": {}
        }] if remaining_text.strip() else []

    # Step 3: Process each content block
    for block in content_blocks:
        block_type = block.get("type", "text")
        content = block.get("content", "")

        if not content.strip():
            continue

        if block_type == "list":
            list_chunks = chunk_list_block(
                block, pdf_name, page_no, max_words, chunk_number
            )
            for chunk in list_chunks:
                chunk["position"] = position + (chunk.get("position", 0) * 0.001)
            chunks.extend(list_chunks)
            chunk_number += len(list_chunks)
            position += 1

        else:
            # Regular text chunking
            text_chunks = _chunk_text_content_enhanced(
                content, pdf_name, page_no, position, max_words, chunk_number
            )
            chunks.extend(text_chunks)
            chunk_number += len(text_chunks)
            position += 1

    # Step 4: Add table/figure chunks
    for tf_block in table_figure_blocks:
        chunk = chunk_table_or_figure(tf_block, pdf_name, page_no, chunk_number)
        chunk["position"] = position
        chunks.append(chunk)
        chunk_number += 1
        position += 1

    return chunks


def _chunk_text_content_enhanced(
    text: str,
    pdf_name: str,
    page_no: int,
    position: int,
    max_words: int,
    start_chunk_number: int
) -> List[Dict[str, Any]]:
    """
    Enhanced text chunking with improved sentence splitting.
    """
    sentences = sentence_split(text)
    chunks = []

    if not sentences:
        return chunks

    current_chunk = []
    current_word_count = 0
    chunk_number = start_chunk_number
    sub_position = 0

    for sentence in sentences:
        words = sentence.split()
        word_count = len(words)

        # If single sentence exceeds max, split it
        if word_count > max_words:
            # Save current chunk if any
            if current_chunk:
                chunk_text = " ".join(current_chunk)
                chunks.append(_create_text_chunk(
                    chunk_text, pdf_name, page_no,
                    position + (sub_position * 0.01),
                    chunk_number, "enhanced"
                ))
                chunk_number += 1
                sub_position += 1
                current_chunk = []
                current_word_count = 0

            # Split long sentence
            sentence_chunks = _split_long_sentence(sentence, max_words)
            for sc in sentence_chunks:
                chunks.append(_create_text_chunk(
                    sc, pdf_name, page_no,
                    position + (sub_position * 0.01),
                    chunk_number, "enhanced_split"
                ))
                chunk_number += 1
                sub_position += 1
            continue

        if current_word_count + word_count > max_words and current_chunk:
            # Save current chunk
            chunk_text = " ".join(current_chunk)
            chunks.append(_create_text_chunk(
                chunk_text, pdf_name, page_no,
                position + (sub_position * 0.01),
                chunk_number, "enhanced"
            ))
            chunk_number += 1
            sub_position += 1
            current_chunk = []
            current_word_count = 0

        current_chunk.append(sentence)
        current_word_count += word_count

    # Save final chunk
    if current_chunk:
        chunk_text = " ".join(current_chunk)
        chunks.append(_create_text_chunk(
            chunk_text, pdf_name, page_no,
            position + (sub_position * 0.01),
            chunk_number, "enhanced"
        ))

    return chunks


def _split_long_sentence(sentence: str, max_words: int) -> List[str]:
    """Split a long sentence into smaller parts at natural boundaries."""
    words = sentence.split()
    if len(words) <= max_words:
        return [sentence]

    chunks = []
    current = []

    # Try to split at punctuation first
    for word in words:
        current.append(word)
        if len(current) >= max_words:
            # Try to find a good split point
            text = ' '.join(current)
            # Split at comma, semicolon, or colon if possible
            split_match = re.search(r'(.*?[,;:])\s+(\S.*)$', text)
            if split_match and len(split_match.group(1).split()) >= max_words // 2:
                chunks.append(split_match.group(1))
                current = split_match.group(2).split()
            else:
                chunks.append(text)
                current = []

    if current:
        chunks.append(' '.join(current))

    return chunks


def _create_text_chunk(
    text: str,
    pdf_name: str,
    page_no: int,
    position: float,
    chunk_number: int,
    method: str
) -> Dict[str, Any]:
    """Create a text chunk dictionary."""
    return {
        "chunk_id": deterministic_chunk_id(pdf_name, page_no, text, "text"),
        "content_type": "text",
        "text": text,
        "pdf_name": pdf_name,
        "page_no": page_no,
        "position": position,
        "chunk_number": chunk_number,
        "image_link": "",
        "table_link": "",
        "context_before_id": "",
        "context_after_id": "",
        "metadata": {"chunking_method": method}
    }


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Calculate cosine similarity between two vectors."""
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return float(np.dot(vec1, vec2) / (norm1 * norm2))


def calculate_sentence_similarities(embeddings: List[List[float]]) -> List[float]:
    """
    Calculate cosine similarity between consecutive sentence embeddings.

    Returns list of similarities where similarities[i] is the similarity
    between sentence i and sentence i+1.
    """
    if len(embeddings) < 2:
        return []

    similarities = []
    embeddings_np = [np.array(emb) for emb in embeddings]

    for i in range(len(embeddings_np) - 1):
        sim = cosine_similarity(embeddings_np[i], embeddings_np[i + 1])
        similarities.append(sim)

    return similarities


def find_semantic_breakpoints(
    similarities: List[float],
    threshold: float = 0.5,
    percentile_threshold: Optional[float] = None
) -> List[int]:
    """
    Find breakpoints where semantic similarity drops below threshold.

    Args:
        similarities: List of similarities between consecutive sentences
        threshold: Absolute similarity threshold (default 0.5)
        percentile_threshold: If set, use this percentile of similarities as threshold
                            (e.g., 25 means break at bottom 25% of similarities)

    Returns:
        List of indices where breaks should occur (break AFTER sentence at index)
    """
    if not similarities:
        return []

    # Use percentile-based threshold if specified
    if percentile_threshold is not None:
        threshold = float(np.percentile(similarities, percentile_threshold))

    breakpoints = []
    for i, sim in enumerate(similarities):
        if sim < threshold:
            breakpoints.append(i)

    return breakpoints


class SemanticChunker:
    """
    Embedding-based semantic chunker that detects topic boundaries.

    Uses sentence embeddings to identify where semantic shifts occur,
    creating chunks that preserve topical coherence.
    """

    def __init__(
        self,
        embedding_client: Optional[Any] = None,
        similarity_threshold: float = 0.5,
        percentile_threshold: Optional[float] = 25,
        min_chunk_size: int = 50,
        max_chunk_size: int = 500,
        combine_threshold: float = 0.7,
        buffer_size: int = 1
    ):
        """
        Initialize the semantic chunker.

        Args:
            embedding_client: EmbeddingClient instance for generating embeddings
            similarity_threshold: Minimum similarity to keep sentences together (0-1)
            percentile_threshold: If set, use bottom N percentile as breakpoints
            min_chunk_size: Minimum words per chunk (will merge small chunks)
            max_chunk_size: Maximum words per chunk (will split large chunks)
            combine_threshold: Similarity threshold for combining small chunks
            buffer_size: Number of sentences to consider for context (sliding window)
        """
        self.embedding_client = embedding_client
        self.similarity_threshold = similarity_threshold
        self.percentile_threshold = percentile_threshold
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
        self.combine_threshold = combine_threshold
        self.buffer_size = buffer_size

    def _get_sentence_embeddings(self, sentences: List[str]) -> Optional[List[List[float]]]:
        """Get embeddings for sentences, with buffering for context."""
        if not self.embedding_client or not sentences:
            return None

        try:
            # Create buffered sentences for better context
            if self.buffer_size > 1 and len(sentences) > 1:
                buffered = []
                for i in range(len(sentences)):
                    start = max(0, i - self.buffer_size // 2)
                    end = min(len(sentences), i + self.buffer_size // 2 + 1)
                    buffered.append(" ".join(sentences[start:end]))
                return self.embedding_client.embed(buffered)
            else:
                return self.embedding_client.embed(sentences)
        except Exception as e:
            logger.warning(f"Failed to get embeddings for semantic chunking: {e}")
            return None

    def _merge_small_chunks(
        self,
        chunks: List[List[str]],
        embeddings: Optional[List[List[float]]] = None
    ) -> List[List[str]]:
        """Merge chunks that are too small with their most similar neighbor."""
        if not chunks:
            return chunks

        merged = []
        i = 0

        while i < len(chunks):
            current_chunk = chunks[i]
            current_words = sum(len(s.split()) for s in current_chunk)

            # If chunk is too small and not the last one, try to merge
            if current_words < self.min_chunk_size and i < len(chunks) - 1:
                # Merge with next chunk
                next_chunk = chunks[i + 1]
                merged_chunk = current_chunk + next_chunk
                merged.append(merged_chunk)
                i += 2
            elif current_words < self.min_chunk_size and merged:
                # Merge with previous chunk if this is the last small chunk
                merged[-1] = merged[-1] + current_chunk
                i += 1
            else:
                merged.append(current_chunk)
                i += 1

        return merged

    def _split_large_chunks(self, chunks: List[List[str]]) -> List[List[str]]:
        """Split chunks that exceed max_chunk_size."""
        result = []

        for chunk in chunks:
            total_words = sum(len(s.split()) for s in chunk)

            if total_words <= self.max_chunk_size:
                result.append(chunk)
                continue

            # Split the chunk
            current_split = []
            current_words = 0

            for sentence in chunk:
                sentence_words = len(sentence.split())

                if current_words + sentence_words > self.max_chunk_size and current_split:
                    result.append(current_split)
                    current_split = []
                    current_words = 0

                current_split.append(sentence)
                current_words += sentence_words

            if current_split:
                result.append(current_split)

        return result

    def chunk_text(
        self,
        text: str,
        pdf_name: str,
        page_no: int,
        start_chunk_number: int = 0
    ) -> List[Dict[str, Any]]:
        """
        Perform semantic chunking on text.

        Args:
            text: The text to chunk
            pdf_name: Name of the source PDF
            page_no: Page number in the PDF
            start_chunk_number: Starting chunk number

        Returns:
            List of chunk dictionaries with metadata
        """
        sentences = sentence_split(text)

        if not sentences:
            return []

        # If only one sentence or embedding client unavailable, use simple chunking
        if len(sentences) <= 1 or not self.embedding_client:
            return self._fallback_chunking(
                text, pdf_name, page_no, start_chunk_number
            )

        # Get embeddings for sentences
        embeddings = self._get_sentence_embeddings(sentences)

        if embeddings is None:
            logger.info("Falling back to simple chunking (no embeddings)")
            return self._fallback_chunking(
                text, pdf_name, page_no, start_chunk_number
            )

        # Calculate similarities between consecutive sentences
        similarities = calculate_sentence_similarities(embeddings)

        # Find breakpoints
        breakpoints = find_semantic_breakpoints(
            similarities,
            threshold=self.similarity_threshold,
            percentile_threshold=self.percentile_threshold
        )

        # Group sentences into chunks based on breakpoints
        chunks_sentences = self._group_by_breakpoints(sentences, breakpoints)

        # Apply size constraints
        chunks_sentences = self._merge_small_chunks(chunks_sentences, embeddings)
        chunks_sentences = self._split_large_chunks(chunks_sentences)

        # Convert to chunk dictionaries
        return self._sentences_to_chunks(
            chunks_sentences, pdf_name, page_no, start_chunk_number
        )

    def _group_by_breakpoints(
        self,
        sentences: List[str],
        breakpoints: List[int]
    ) -> List[List[str]]:
        """Group sentences into chunks based on breakpoint indices."""
        if not breakpoints:
            return [sentences]

        chunks = []
        prev_idx = 0

        for bp in sorted(breakpoints):
            # Break AFTER the sentence at index bp
            chunk = sentences[prev_idx:bp + 1]
            if chunk:
                chunks.append(chunk)
            prev_idx = bp + 1

        # Add remaining sentences
        if prev_idx < len(sentences):
            chunks.append(sentences[prev_idx:])

        return chunks

    def _sentences_to_chunks(
        self,
        chunks_sentences: List[List[str]],
        pdf_name: str,
        page_no: int,
        start_chunk_number: int
    ) -> List[Dict[str, Any]]:
        """Convert grouped sentences to chunk dictionaries."""
        chunks = []
        chunk_number = start_chunk_number

        for sentence_group in chunks_sentences:
            chunk_text = " ".join(sentence_group)

            chunks.append({
                "chunk_id": deterministic_chunk_id(
                    pdf_name, page_no, chunk_text, "text"
                ),
                "content_type": "text",
                "text": chunk_text,
                "pdf_name": pdf_name,
                "page_no": page_no,
                "position": 0,
                "chunk_number": chunk_number,
                "image_link": "",
                "table_link": "",
                "context_before_id": "",
                "context_after_id": "",
                "metadata": {
                    "chunking_method": "semantic",
                    "sentence_count": len(sentence_group)
                }
            })
            chunk_number += 1

        return chunks

    def _fallback_chunking(
        self,
        text: str,
        pdf_name: str,
        page_no: int,
        start_chunk_number: int
    ) -> List[Dict[str, Any]]:
        """Fallback to simple word-count based chunking."""
        sentences = sentence_split(text)
        chunks = []

        current_chunk = []
        current_word_count = 0
        chunk_number = start_chunk_number

        for sentence in sentences:
            words = sentence.split()

            if current_word_count + len(words) > self.max_chunk_size and current_chunk:
                chunk_text = " ".join(current_chunk)
                chunks.append({
                    "chunk_id": deterministic_chunk_id(
                        pdf_name, page_no, chunk_text, "text"
                    ),
                    "content_type": "text",
                    "text": chunk_text,
                    "pdf_name": pdf_name,
                    "page_no": page_no,
                    "position": 0,
                    "chunk_number": chunk_number,
                    "image_link": "",
                    "table_link": "",
                    "context_before_id": "",
                    "context_after_id": "",
                    "metadata": {"chunking_method": "fallback"}
                })
                chunk_number += 1
                current_chunk = []
                current_word_count = 0

            current_chunk.append(sentence)
            current_word_count += len(words)

        # Last chunk
        if current_chunk:
            chunk_text = " ".join(current_chunk)
            chunks.append({
                "chunk_id": deterministic_chunk_id(
                    pdf_name, page_no, chunk_text, "text"
                ),
                "content_type": "text",
                "text": chunk_text,
                "pdf_name": pdf_name,
                "page_no": page_no,
                "position": 0,
                "chunk_number": chunk_number,
                "image_link": "",
                "table_link": "",
                "context_before_id": "",
                "context_after_id": "",
                "metadata": {"chunking_method": "fallback"}
            })

        return chunks


def semantic_chunk_text(
    text: str,
    pdf_name: str,
    page_no: int,
    max_words: int = 250,
    start_chunk_number: int = 0
) -> List[Dict]:
    """
    Create semantic chunks with deterministic IDs and metadata.

    Args:
        text: The text to chunk
        pdf_name: Name of the source PDF
        page_no: Page number in the PDF
        max_words: Maximum words per chunk (default 250)
        start_chunk_number: Starting chunk number for this page (default 0)

    Returns:
        List of chunk dictionaries with chunk_id, pdf_name, page_no, chunk_number, and text
    """
    sentences = sentence_split(text)
    chunks = []

    current_chunk = []
    current_word_count = 0
    chunk_number = start_chunk_number

    for sentence in sentences:
        words = sentence.split()

        if current_word_count + len(words) > max_words:
            chunk_text = " ".join(current_chunk)

            chunks.append({
                "chunk_id": deterministic_chunk_id(
                    pdf_name, page_no, chunk_text
                ),
                "pdf_name": pdf_name,
                "page_no": page_no,
                "chunk_number": chunk_number,
                "text": chunk_text
            })

            chunk_number += 1
            current_chunk = []
            current_word_count = 0

        current_chunk.append(sentence)
        current_word_count += len(words)

    # last chunk
    if current_chunk:
        chunk_text = " ".join(current_chunk)

        chunks.append({
            "chunk_id": deterministic_chunk_id(
                pdf_name, page_no, chunk_text
            ),
            "pdf_name": pdf_name,
            "page_no": page_no,
            "chunk_number": chunk_number,
            "text": chunk_text
        })

    return chunks


def create_multimodal_chunks(
    blocks: List[Dict[str, Any]],
    pdf_name: str,
    page_no: int,
    max_words: int = 250,
    start_chunk_number: int = 0
) -> List[Dict[str, Any]]:
    """
    Create chunks from multimodal content blocks (text, tables, images).

    Each block becomes one or more chunks:
    - Text blocks: Split by sentence/word count
    - Table blocks: Single chunk with vision-generated summary
    - Image blocks: Single chunk with vision-generated description

    Args:
        blocks: List of content blocks with type, content, position, links
        pdf_name: Name of the source PDF
        page_no: Page number in the PDF
        max_words: Maximum words per text chunk
        start_chunk_number: Starting chunk number

    Returns:
        List of chunk dictionaries ready for embedding and indexing
    """
    chunks = []
    chunk_number = start_chunk_number

    for block in blocks:
        block_type = block.get("type", "text")
        content = block.get("content", "")
        position = block.get("position", 0)
        image_link = block.get("image_link")
        table_link = block.get("table_link")

        if block_type == "text" and content:
            # Split text into smaller chunks if needed
            text_chunks = _chunk_text_content(
                content, pdf_name, page_no, position, max_words, chunk_number
            )
            chunks.extend(text_chunks)
            chunk_number += len(text_chunks)

        elif block_type == "table":
            # Table as single chunk with vision summary
            chunk_text = content if content else "Table content"
            chunks.append({
                "chunk_id": deterministic_chunk_id(pdf_name, page_no, chunk_text, "table"),
                "content_type": "table",
                "text": chunk_text,
                "pdf_name": pdf_name,
                "page_no": page_no,
                "position": position,
                "chunk_number": chunk_number,
                "image_link": "",
                "table_link": table_link or "",
                "context_before_id": "",
                "context_after_id": "",
                "metadata": block.get("metadata", {})
            })
            chunk_number += 1

        elif block_type == "image":
            # Image as single chunk with vision description
            chunk_text = content if content else "Image content"
            chunks.append({
                "chunk_id": deterministic_chunk_id(pdf_name, page_no, chunk_text, "image"),
                "content_type": "image",
                "text": chunk_text,
                "pdf_name": pdf_name,
                "page_no": page_no,
                "position": position,
                "chunk_number": chunk_number,
                "image_link": image_link or "",
                "table_link": "",
                "context_before_id": "",
                "context_after_id": "",
                "metadata": block.get("metadata", {})
            })
            chunk_number += 1

    # Link context (previous and next chunks)
    chunks = _link_chunk_context(chunks)

    return chunks


def _chunk_text_content(
    text: str,
    pdf_name: str,
    page_no: int,
    position: int,
    max_words: int,
    start_chunk_number: int
) -> List[Dict[str, Any]]:
    """Split text content into chunks respecting word limits."""
    sentences = sentence_split(text)
    chunks = []

    current_chunk = []
    current_word_count = 0
    chunk_number = start_chunk_number
    sub_position = 0

    for sentence in sentences:
        words = sentence.split()

        if current_word_count + len(words) > max_words and current_chunk:
            chunk_text = " ".join(current_chunk)
            chunks.append({
                "chunk_id": deterministic_chunk_id(pdf_name, page_no, chunk_text, "text"),
                "content_type": "text",
                "text": chunk_text,
                "pdf_name": pdf_name,
                "page_no": page_no,
                "position": position + (sub_position * 0.01),  # Sub-position for text splits
                "chunk_number": chunk_number,
                "image_link": "",
                "table_link": "",
                "context_before_id": "",
                "context_after_id": "",
                "metadata": {}
            })

            chunk_number += 1
            sub_position += 1
            current_chunk = []
            current_word_count = 0

        current_chunk.append(sentence)
        current_word_count += len(words)

    # Last chunk
    if current_chunk:
        chunk_text = " ".join(current_chunk)
        chunks.append({
            "chunk_id": deterministic_chunk_id(pdf_name, page_no, chunk_text, "text"),
            "content_type": "text",
            "text": chunk_text,
            "pdf_name": pdf_name,
            "page_no": page_no,
            "position": position + (sub_position * 0.01),
            "chunk_number": chunk_number,
            "image_link": "",
            "table_link": "",
            "context_before_id": "",
            "context_after_id": "",
            "metadata": {}
        })

    return chunks


def _link_chunk_context(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Link each chunk to its previous and next chunk for context."""
    for i, chunk in enumerate(chunks):
        if i > 0:
            chunk["context_before_id"] = chunks[i - 1]["chunk_id"]
        if i < len(chunks) - 1:
            chunk["context_after_id"] = chunks[i + 1]["chunk_id"]

    return chunks


def process_page_to_chunks(
    page_result: Dict[str, Any],
    vision_processor: Optional[Any] = None,
    max_words: int = 250,
    start_chunk_number: int = 0
) -> List[Dict[str, Any]]:
    """
    Process a page result (from pdf_utils.process_page_with_positions) into chunks.

    Args:
        page_result: Result from process_page_with_positions()
        vision_processor: Optional VisionProcessor for image/table descriptions
        max_words: Maximum words per text chunk
        start_chunk_number: Starting chunk number

    Returns:
        List of chunks ready for embedding and indexing
    """
    pdf_name = page_result["metadata"]["pdf_name"]
    page_no = page_result["metadata"]["page_no"]
    blocks = page_result.get("blocks", [])

    # Process blocks with vision model if available
    if vision_processor and blocks:
        from vision_utils import VisionProcessor
        if isinstance(vision_processor, VisionProcessor):
            blocks = vision_processor.process_blocks(blocks)

    # Create chunks from blocks
    chunks = create_multimodal_chunks(
        blocks=blocks,
        pdf_name=pdf_name,
        page_no=page_no,
        max_words=max_words,
        start_chunk_number=start_chunk_number
    )

    return chunks


def create_semantic_multimodal_chunks(
    blocks: List[Dict[str, Any]],
    pdf_name: str,
    page_no: int,
    embedding_client: Optional[Any] = None,
    similarity_threshold: float = 0.5,
    percentile_threshold: Optional[float] = 25,
    min_chunk_size: int = 50,
    max_chunk_size: int = 500,
    start_chunk_number: int = 0
) -> List[Dict[str, Any]]:
    """
    Create semantically-aware chunks from multimodal content blocks.

    Uses embedding-based semantic chunking for text blocks to detect
    topic boundaries, while handling tables and images as single chunks.

    Args:
        blocks: List of content blocks with type, content, position, links
        pdf_name: Name of the source PDF
        page_no: Page number in the PDF
        embedding_client: EmbeddingClient for semantic chunking (optional)
        similarity_threshold: Minimum similarity to keep sentences together (0-1)
        percentile_threshold: Use bottom N percentile as breakpoints
        min_chunk_size: Minimum words per chunk
        max_chunk_size: Maximum words per chunk
        start_chunk_number: Starting chunk number

    Returns:
        List of chunk dictionaries ready for embedding and indexing
    """
    # Initialize semantic chunker
    semantic_chunker = SemanticChunker(
        embedding_client=embedding_client,
        similarity_threshold=similarity_threshold,
        percentile_threshold=percentile_threshold,
        min_chunk_size=min_chunk_size,
        max_chunk_size=max_chunk_size
    )

    chunks = []
    chunk_number = start_chunk_number

    for block in blocks:
        block_type = block.get("type", "text")
        content = block.get("content", "")
        position = block.get("position", 0)
        image_link = block.get("image_link")
        table_link = block.get("table_link")

        if block_type == "text" and content:
            # Use semantic chunking for text blocks
            text_chunks = semantic_chunker.chunk_text(
                text=content,
                pdf_name=pdf_name,
                page_no=page_no,
                start_chunk_number=chunk_number
            )

            # Update positions for each chunk
            for i, chunk in enumerate(text_chunks):
                chunk["position"] = position + (i * 0.01)

            chunks.extend(text_chunks)
            chunk_number += len(text_chunks)

        elif block_type == "table":
            # Table as single chunk with vision summary
            chunk_text = content if content else "Table content"
            chunks.append({
                "chunk_id": deterministic_chunk_id(pdf_name, page_no, chunk_text, "table"),
                "content_type": "table",
                "text": chunk_text,
                "pdf_name": pdf_name,
                "page_no": page_no,
                "position": position,
                "chunk_number": chunk_number,
                "image_link": "",
                "table_link": table_link or "",
                "context_before_id": "",
                "context_after_id": "",
                "metadata": block.get("metadata", {})
            })
            chunk_number += 1

        elif block_type == "image":
            # Image as single chunk with vision description
            chunk_text = content if content else "Image content"
            chunks.append({
                "chunk_id": deterministic_chunk_id(pdf_name, page_no, chunk_text, "image"),
                "content_type": "image",
                "text": chunk_text,
                "pdf_name": pdf_name,
                "page_no": page_no,
                "position": position,
                "chunk_number": chunk_number,
                "image_link": image_link or "",
                "table_link": "",
                "context_before_id": "",
                "context_after_id": "",
                "metadata": block.get("metadata", {})
            })
            chunk_number += 1

    # Link context (previous and next chunks)
    chunks = _link_chunk_context(chunks)

    return chunks


def process_page_to_semantic_chunks(
    page_result: Dict[str, Any],
    embedding_client: Optional[Any] = None,
    vision_processor: Optional[Any] = None,
    similarity_threshold: float = 0.5,
    percentile_threshold: Optional[float] = 25,
    min_chunk_size: int = 50,
    max_chunk_size: int = 500,
    start_chunk_number: int = 0
) -> List[Dict[str, Any]]:
    """
    Process a page result into semantically-aware chunks.

    This is the main entry point for semantic chunking of PDF pages.

    Args:
        page_result: Result from process_page_with_positions()
        embedding_client: EmbeddingClient for semantic chunking
        vision_processor: Optional VisionProcessor for image/table descriptions
        similarity_threshold: Minimum similarity to keep sentences together
        percentile_threshold: Use bottom N percentile as breakpoints
        min_chunk_size: Minimum words per chunk
        max_chunk_size: Maximum words per chunk
        start_chunk_number: Starting chunk number

    Returns:
        List of chunks ready for embedding and indexing
    """
    pdf_name = page_result["metadata"]["pdf_name"]
    page_no = page_result["metadata"]["page_no"]
    blocks = page_result.get("blocks", [])

    # Process blocks with vision model if available
    if vision_processor and blocks:
        from vision_utils import VisionProcessor
        if isinstance(vision_processor, VisionProcessor):
            blocks = vision_processor.process_blocks(blocks)

    # Create semantic chunks from blocks
    chunks = create_semantic_multimodal_chunks(
        blocks=blocks,
        pdf_name=pdf_name,
        page_no=page_no,
        embedding_client=embedding_client,
        similarity_threshold=similarity_threshold,
        percentile_threshold=percentile_threshold,
        min_chunk_size=min_chunk_size,
        max_chunk_size=max_chunk_size,
        start_chunk_number=start_chunk_number
    )

    return chunks
