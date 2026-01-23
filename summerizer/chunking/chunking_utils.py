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
    PARAGRAPH = "paragraph"      # Regular text paragraphs (max 500 words)
    LIST = "list"                # Bullet/numbered lists as separate chunks
    TABLE = "table"              # Table summary as separate chunk
    FIGURE = "figure"            # Image/figure summary as separate chunk
    TABLE_OF_CONTENTS = "toc"    # Table of contents as separate chunk
    HEADING = "heading"          # Section headings
    TEXT = "text"                # Generic text (for backward compatibility)


# =============================================================================
# Page Number and Artifact Removal
# =============================================================================

# Patterns for page numbers and artifacts to remove
PAGE_NUMBER_PATTERNS = [
    r'^\s*\d+\s*$',                          # Standalone page numbers: "5", " 12 "
    r'^\s*[-–—]\s*\d+\s*[-–—]\s*$',          # Dash-surrounded: "- 5 -", "– 12 –"
    r'^\s*Page\s+\d+\s*$',                   # "Page 5"
    r'^\s*Page\s+\d+\s+of\s+\d+\s*$',        # "Page 5 of 10"
    r'^\s*\d+\s*/\s*\d+\s*$',                # "5/10"
    r'^\s*p\.\s*\d+\s*$',                    # "p. 5"
    r'^\s*\[\d+\]\s*$',                      # "[5]"
]

# Patterns for Table of Contents detection
TOC_PATTERNS = [
    r'^Table\s+of\s+Contents?\s*$',
    r'^Contents?\s*$',
    r'^INDEX\s*$',
    r'^CONTENTS\s*$',
]

# Pattern for TOC entries (title followed by page number)
TOC_ENTRY_PATTERN = r'^(.+?)\s+\.{2,}\s*\d+\s*$|^(.+?)\s{2,}\d+\s*$'


def is_page_number(text: str) -> bool:
    """Check if text is a page number or page artifact."""
    text = text.strip()
    if not text:
        return False

    for pattern in PAGE_NUMBER_PATTERNS:
        if re.match(pattern, text, re.IGNORECASE):
            return True

    # Also check for single digits or small numbers at line boundaries
    if re.match(r'^\d{1,3}$', text) and int(text) < 1000:
        return True

    return False


def remove_page_numbers(text: str) -> str:
    """Remove page numbers and artifacts from text."""
    if not text:
        return ""

    lines = text.split('\n')
    cleaned_lines = []

    for line in lines:
        stripped = line.strip()

        # Skip page numbers
        if is_page_number(stripped):
            continue

        # Skip empty lines that might be artifacts
        if not stripped:
            # Keep one empty line for paragraph separation
            if cleaned_lines and cleaned_lines[-1].strip():
                cleaned_lines.append('')
            continue

        cleaned_lines.append(line)

    return '\n'.join(cleaned_lines)


def is_table_of_contents_header(text: str) -> bool:
    """Check if text is a Table of Contents header."""
    text = text.strip()
    for pattern in TOC_PATTERNS:
        if re.match(pattern, text, re.IGNORECASE):
            return True
    return False


def is_toc_entry(text: str) -> bool:
    """Check if text looks like a TOC entry (title ... page_number)."""
    text = text.strip()
    # Check for dotted leaders or multiple spaces followed by number
    if re.match(TOC_ENTRY_PATTERN, text):
        return True
    # Check for section-like entries with page numbers
    if re.match(r'^[IVX\d]+\.\s+.+\s+\d+\s*$', text):
        return True
    return False


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


# =============================================================================
# NEW CHUNKING STRATEGY - User Requirements
# =============================================================================
# Strategy:
# - Paragraph chunks (max 500 words, min 50 words)
# - Table summary as separate chunk
# - Image summary as separate chunk
# - List as separate chunk
# - Table of Contents as separate chunk
# - Remove page numbers
# - Keep cross-page references
# =============================================================================

def extract_table_of_contents(text: str) -> Tuple[str, Optional[Dict[str, Any]]]:
    """
    Extract Table of Contents from text as a separate block.

    Returns:
        (remaining_text, toc_block or None)
    """
    if not text:
        return "", None

    lines = text.split('\n')
    toc_block = None
    toc_lines = []
    in_toc = False
    toc_start_idx = -1
    toc_end_idx = -1

    for i, line in enumerate(lines):
        stripped = line.strip()

        # Check for TOC header
        if is_table_of_contents_header(stripped):
            in_toc = True
            toc_start_idx = i
            toc_lines.append(stripped)
            continue

        if in_toc:
            # Check if this looks like a TOC entry
            if is_toc_entry(stripped) or not stripped:
                toc_lines.append(stripped)
                toc_end_idx = i
            elif stripped and not is_toc_entry(stripped):
                # Check if next few lines are TOC entries (might be a section header in TOC)
                is_still_toc = False
                for j in range(i, min(i + 3, len(lines))):
                    if is_toc_entry(lines[j].strip()):
                        is_still_toc = True
                        break

                if is_still_toc:
                    toc_lines.append(stripped)
                    toc_end_idx = i
                else:
                    # End of TOC
                    break

    if toc_lines and len(toc_lines) > 2:  # At least header + 2 entries
        toc_text = '\n'.join(toc_lines)
        toc_block = {
            "type": "toc",
            "content": toc_text,
            "position": 0,
            "metadata": {
                "is_table_of_contents": True,
                "entry_count": len([l for l in toc_lines if l.strip() and not is_table_of_contents_header(l)])
            }
        }

        # Remove TOC from original text
        remaining_lines = lines[:toc_start_idx] + lines[toc_end_idx + 1:]
        return '\n'.join(remaining_lines), toc_block

    return text, None


def chunk_paragraph(
    text: str,
    pdf_name: str,
    page_no: int,
    position: float,
    chunk_number: int,
    max_words: int = 500,
    min_words: int = 50,
    heading: str = None
) -> List[Dict[str, Any]]:
    """
    Chunk text into paragraph chunks (max 500 words each).

    If heading is provided, it will be prepended to each chunk to maintain context.
    Paragraphs are split at sentence boundaries to respect max_words.
    """
    sentences = sentence_split(text)
    chunks = []

    if not sentences:
        return chunks

    # Calculate heading words (will be added to each chunk)
    heading_words = len(heading.split()) if heading else 0
    effective_max = max_words - heading_words  # Reserve space for heading

    current_sentences = []
    current_word_count = 0
    sub_position = 0
    total_parts = 0

    # First pass: count how many parts we'll create
    temp_sentences = []
    temp_word_count = 0
    for sentence in sentences:
        word_count = len(sentence.split())
        if temp_word_count + word_count > effective_max and temp_sentences:
            total_parts += 1
            temp_sentences = []
            temp_word_count = 0
        temp_sentences.append(sentence)
        temp_word_count += word_count
    if temp_sentences:
        total_parts += 1

    # Second pass: create chunks
    part_num = 1
    for sentence in sentences:
        word_count = len(sentence.split())

        # If adding this sentence exceeds max, save current chunk
        if current_word_count + word_count > effective_max and current_sentences:
            chunk_text = ' '.join(current_sentences)
            # Prepend heading if provided
            if heading:
                chunk_text = f"{heading}\n\n{chunk_text}"

            chunks.append({
                "chunk_id": deterministic_chunk_id(pdf_name, page_no, chunk_text, "paragraph"),
                "content_type": "paragraph",
                "text": chunk_text,
                "pdf_name": pdf_name,
                "page_no": page_no,
                "position": position + (sub_position * 0.001),
                "chunk_number": chunk_number + len(chunks),
                "image_link": "",
                "table_link": "",
                "context_before_id": "",
                "context_after_id": "",
                "metadata": {
                    "chunking_method": "paragraph",
                    "word_count": current_word_count + heading_words,
                    "sentence_count": len(current_sentences),
                    "heading": heading if heading else None,
                    "part": part_num,
                    "total_parts": total_parts,
                    "is_split": total_parts > 1
                }
            })
            part_num += 1
            sub_position += 1
            current_sentences = []
            current_word_count = 0

        current_sentences.append(sentence)
        current_word_count += word_count

    # Save final chunk
    if current_sentences:
        chunk_text = ' '.join(current_sentences)
        # Prepend heading if provided
        if heading:
            chunk_text = f"{heading}\n\n{chunk_text}"

        chunks.append({
            "chunk_id": deterministic_chunk_id(pdf_name, page_no, chunk_text, "paragraph"),
            "content_type": "paragraph",
            "text": chunk_text,
            "pdf_name": pdf_name,
            "page_no": page_no,
            "position": position + (sub_position * 0.001),
            "chunk_number": chunk_number + len(chunks),
            "image_link": "",
            "table_link": "",
            "context_before_id": "",
            "context_after_id": "",
            "metadata": {
                "chunking_method": "paragraph",
                "word_count": current_word_count + heading_words,
                "sentence_count": len(current_sentences),
                "heading": heading if heading else None,
                "part": part_num,
                "total_parts": total_parts,
                "is_split": total_parts > 1
            }
        })

    return chunks


def chunk_table_with_structure(
    content: str,
    pdf_name: str,
    page_no: int,
    position: float,
    chunk_number: int,
    max_words: int = 500,
    table_link: str = "",
    caption: str = ""
) -> List[Dict[str, Any]]:
    """
    Chunk table content while preserving table link and header in all split chunks.

    If table > 500 words, split but keep caption/header and table_link in each chunk.
    """
    total_words = len(content.split())

    # Extract header (first line or caption)
    lines = content.strip().split('\n')
    header = caption if caption else (lines[0] if lines else "")
    header_words = len(header.split())

    # If fits in one chunk
    if total_words <= max_words:
        return [{
            "chunk_id": deterministic_chunk_id(pdf_name, page_no, content, "table"),
            "content_type": "table",
            "text": content,
            "pdf_name": pdf_name,
            "page_no": page_no,
            "position": position,
            "chunk_number": chunk_number,
            "image_link": "",
            "table_link": table_link,
            "context_before_id": "",
            "context_after_id": "",
            "metadata": {
                "chunking_method": "table_summary",
                "caption": caption,
                "is_table": True,
                "is_split": False,
                "part": 1,
                "total_parts": 1
            }
        }]

    # Split table while preserving header
    chunks = []
    effective_max = max_words - header_words - 5  # Reserve space for header + "[continued]"

    # Split content into rows
    rows = lines[1:] if caption else lines  # Skip caption if separate
    current_rows = []
    current_word_count = 0
    part_num = 1

    # First pass: count total parts
    temp_rows = []
    temp_count = 0
    total_parts = 0
    for row in rows:
        row_words = len(row.split())
        if temp_count + row_words > effective_max and temp_rows:
            total_parts += 1
            temp_rows = []
            temp_count = 0
        temp_rows.append(row)
        temp_count += row_words
    if temp_rows:
        total_parts += 1

    # Second pass: create chunks
    for row in rows:
        row_words = len(row.split())

        if current_word_count + row_words > effective_max and current_rows:
            # Create chunk with header
            chunk_content = '\n'.join(current_rows)
            if part_num > 1:
                chunk_text = f"{header} [Part {part_num}/{total_parts}]\n{chunk_content}"
            else:
                chunk_text = f"{header}\n{chunk_content}"

            chunks.append({
                "chunk_id": deterministic_chunk_id(pdf_name, page_no, chunk_text, "table"),
                "content_type": "table",
                "text": chunk_text,
                "pdf_name": pdf_name,
                "page_no": page_no,
                "position": position + ((part_num - 1) * 0.001),
                "chunk_number": chunk_number + len(chunks),
                "image_link": "",
                "table_link": table_link,
                "context_before_id": "",
                "context_after_id": "",
                "metadata": {
                    "chunking_method": "table_summary",
                    "caption": caption,
                    "is_table": True,
                    "is_split": True,
                    "part": part_num,
                    "total_parts": total_parts,
                    "original_header": header
                }
            })
            part_num += 1
            current_rows = []
            current_word_count = 0

        current_rows.append(row)
        current_word_count += row_words

    # Final chunk
    if current_rows:
        chunk_content = '\n'.join(current_rows)
        if total_parts > 1:
            chunk_text = f"{header} [Part {part_num}/{total_parts}]\n{chunk_content}"
        else:
            chunk_text = f"{header}\n{chunk_content}" if header not in chunk_content else chunk_content

        chunks.append({
            "chunk_id": deterministic_chunk_id(pdf_name, page_no, chunk_text, "table"),
            "content_type": "table",
            "text": chunk_text,
            "pdf_name": pdf_name,
            "page_no": page_no,
            "position": position + ((part_num - 1) * 0.001),
            "chunk_number": chunk_number + len(chunks),
            "image_link": "",
            "table_link": table_link,
            "context_before_id": "",
            "context_after_id": "",
            "metadata": {
                "chunking_method": "table_summary",
                "caption": caption,
                "is_table": True,
                "is_split": total_parts > 1,
                "part": part_num,
                "total_parts": total_parts,
                "original_header": header
            }
        })

    return chunks


def chunk_figure_with_structure(
    content: str,
    pdf_name: str,
    page_no: int,
    position: float,
    chunk_number: int,
    max_words: int = 500,
    image_link: str = "",
    caption: str = ""
) -> List[Dict[str, Any]]:
    """
    Chunk figure/image content while preserving image link and caption in all split chunks.

    If figure description > 500 words, split but keep caption and image_link in each chunk.
    """
    total_words = len(content.split())

    # Extract caption (usually first line)
    lines = content.strip().split('\n')
    header = caption if caption else (lines[0] if lines else "")
    header_words = len(header.split())

    # If fits in one chunk
    if total_words <= max_words:
        return [{
            "chunk_id": deterministic_chunk_id(pdf_name, page_no, content, "figure"),
            "content_type": "figure",
            "text": content,
            "pdf_name": pdf_name,
            "page_no": page_no,
            "position": position,
            "chunk_number": chunk_number,
            "image_link": image_link,
            "table_link": "",
            "context_before_id": "",
            "context_after_id": "",
            "metadata": {
                "chunking_method": "image_summary",
                "caption": caption,
                "is_figure": True,
                "is_split": False,
                "part": 1,
                "total_parts": 1
            }
        }]

    # Split figure description while preserving caption
    chunks = []
    sentences = sentence_split(content)
    effective_max = max_words - header_words - 5

    current_sentences = []
    current_word_count = 0
    part_num = 1

    # Count total parts
    temp_sentences = []
    temp_count = 0
    total_parts = 0
    for sentence in sentences:
        s_words = len(sentence.split())
        if temp_count + s_words > effective_max and temp_sentences:
            total_parts += 1
            temp_sentences = []
            temp_count = 0
        temp_sentences.append(sentence)
        temp_count += s_words
    if temp_sentences:
        total_parts += 1

    # Create chunks
    for sentence in sentences:
        s_words = len(sentence.split())

        if current_word_count + s_words > effective_max and current_sentences:
            chunk_content = ' '.join(current_sentences)
            if part_num > 1:
                chunk_text = f"{header} [Part {part_num}/{total_parts}]\n{chunk_content}"
            else:
                chunk_text = f"{header}\n{chunk_content}" if header not in chunk_content else chunk_content

            chunks.append({
                "chunk_id": deterministic_chunk_id(pdf_name, page_no, chunk_text, "figure"),
                "content_type": "figure",
                "text": chunk_text,
                "pdf_name": pdf_name,
                "page_no": page_no,
                "position": position + ((part_num - 1) * 0.001),
                "chunk_number": chunk_number + len(chunks),
                "image_link": image_link,
                "table_link": "",
                "context_before_id": "",
                "context_after_id": "",
                "metadata": {
                    "chunking_method": "image_summary",
                    "caption": caption,
                    "is_figure": True,
                    "is_split": True,
                    "part": part_num,
                    "total_parts": total_parts,
                    "original_caption": header
                }
            })
            part_num += 1
            current_sentences = []
            current_word_count = 0

        current_sentences.append(sentence)
        current_word_count += s_words

    # Final chunk
    if current_sentences:
        chunk_content = ' '.join(current_sentences)
        if total_parts > 1:
            chunk_text = f"{header} [Part {part_num}/{total_parts}]\n{chunk_content}"
        else:
            chunk_text = content

        chunks.append({
            "chunk_id": deterministic_chunk_id(pdf_name, page_no, chunk_text, "figure"),
            "content_type": "figure",
            "text": chunk_text,
            "pdf_name": pdf_name,
            "page_no": page_no,
            "position": position + ((part_num - 1) * 0.001),
            "chunk_number": chunk_number + len(chunks),
            "image_link": image_link,
            "table_link": "",
            "context_before_id": "",
            "context_after_id": "",
            "metadata": {
                "chunking_method": "image_summary",
                "caption": caption,
                "is_figure": True,
                "is_split": total_parts > 1,
                "part": part_num,
                "total_parts": total_parts,
                "original_caption": header
            }
        })

    return chunks


def chunk_toc_with_structure(
    content: str,
    pdf_name: str,
    page_no: int,
    chunk_number: int,
    max_words: int = 500
) -> List[Dict[str, Any]]:
    """
    Chunk Table of Contents while preserving header in all split chunks.

    If TOC > 500 words, split but keep "Table of Contents" header in each chunk.
    """
    lines = content.strip().split('\n')
    total_words = len(content.split())

    # Find TOC header
    header = "Table of Contents"
    for line in lines[:3]:
        if is_table_of_contents_header(line.strip()):
            header = line.strip()
            break

    header_words = len(header.split())

    # If fits in one chunk
    if total_words <= max_words:
        return [{
            "chunk_id": deterministic_chunk_id(pdf_name, page_no, content, "toc"),
            "content_type": "toc",
            "text": content,
            "pdf_name": pdf_name,
            "page_no": page_no,
            "position": 0,
            "chunk_number": chunk_number,
            "image_link": "",
            "table_link": "",
            "context_before_id": "",
            "context_after_id": "",
            "metadata": {
                "chunking_method": "toc_separate",
                "is_table_of_contents": True,
                "is_split": False,
                "part": 1,
                "total_parts": 1
            }
        }]

    # Split TOC while preserving header
    chunks = []
    effective_max = max_words - header_words - 5
    entries = [l for l in lines if l.strip() and not is_table_of_contents_header(l.strip())]

    current_entries = []
    current_word_count = 0
    part_num = 1

    # Count total parts
    total_parts = 0
    temp_entries = []
    temp_count = 0
    for entry in entries:
        e_words = len(entry.split())
        if temp_count + e_words > effective_max and temp_entries:
            total_parts += 1
            temp_entries = []
            temp_count = 0
        temp_entries.append(entry)
        temp_count += e_words
    if temp_entries:
        total_parts += 1

    # Create chunks
    for entry in entries:
        e_words = len(entry.split())

        if current_word_count + e_words > effective_max and current_entries:
            chunk_content = '\n'.join(current_entries)
            chunk_text = f"{header} [Part {part_num}/{total_parts}]\n{chunk_content}"

            chunks.append({
                "chunk_id": deterministic_chunk_id(pdf_name, page_no, chunk_text, "toc"),
                "content_type": "toc",
                "text": chunk_text,
                "pdf_name": pdf_name,
                "page_no": page_no,
                "position": (part_num - 1) * 0.001,
                "chunk_number": chunk_number + len(chunks),
                "image_link": "",
                "table_link": "",
                "context_before_id": "",
                "context_after_id": "",
                "metadata": {
                    "chunking_method": "toc_separate",
                    "is_table_of_contents": True,
                    "is_split": True,
                    "part": part_num,
                    "total_parts": total_parts,
                    "original_header": header
                }
            })
            part_num += 1
            current_entries = []
            current_word_count = 0

        current_entries.append(entry)
        current_word_count += e_words

    # Final chunk
    if current_entries:
        chunk_content = '\n'.join(current_entries)
        if total_parts > 1:
            chunk_text = f"{header} [Part {part_num}/{total_parts}]\n{chunk_content}"
        else:
            chunk_text = content

        chunks.append({
            "chunk_id": deterministic_chunk_id(pdf_name, page_no, chunk_text, "toc"),
            "content_type": "toc",
            "text": chunk_text,
            "pdf_name": pdf_name,
            "page_no": page_no,
            "position": (part_num - 1) * 0.001,
            "chunk_number": chunk_number + len(chunks),
            "image_link": "",
            "table_link": "",
            "context_before_id": "",
            "context_after_id": "",
            "metadata": {
                "chunking_method": "toc_separate",
                "is_table_of_contents": True,
                "is_split": total_parts > 1,
                "part": part_num,
                "total_parts": total_parts,
                "original_header": header
            }
        })

    return chunks


def chunk_list_with_heading(
    list_text: str,
    pdf_name: str,
    page_no: int,
    position: float,
    chunk_number: int,
    max_words: int = 500,
    heading: str = None
) -> List[Dict[str, Any]]:
    """
    Chunk list while preserving heading in all split chunks.

    If list has a heading and > 500 words, keep heading in each split chunk.
    """
    lines = list_text.strip().split('\n')
    items = [l for l in lines if l.strip()]
    total_words = len(list_text.split())

    # Calculate effective max (reserve space for heading)
    heading_words = len(heading.split()) if heading else 0
    effective_max = max_words - heading_words - 2

    # If fits in one chunk
    if total_words + heading_words <= max_words:
        full_text = f"{heading}\n{list_text}" if heading else list_text
        return [{
            "chunk_id": deterministic_chunk_id(pdf_name, page_no, full_text, "list"),
            "content_type": "list",
            "text": full_text,
            "pdf_name": pdf_name,
            "page_no": page_no,
            "position": position,
            "chunk_number": chunk_number,
            "image_link": "",
            "table_link": "",
            "context_before_id": "",
            "context_after_id": "",
            "metadata": {
                "chunking_method": "list_with_heading",
                "item_count": len(items),
                "heading": heading,
                "is_complete_list": True,
                "is_split": False,
                "part": 1,
                "total_parts": 1
            }
        }]

    # Split list while preserving heading
    chunks = []
    current_items = []
    current_word_count = 0
    part_num = 1

    # Count total parts
    total_parts = 0
    temp_items = []
    temp_count = 0
    for item in items:
        i_words = len(item.split())
        if temp_count + i_words > effective_max and temp_items:
            total_parts += 1
            temp_items = []
            temp_count = 0
        temp_items.append(item)
        temp_count += i_words
    if temp_items:
        total_parts += 1

    # Create chunks
    for item in items:
        i_words = len(item.split())

        if current_word_count + i_words > effective_max and current_items:
            chunk_content = '\n'.join(current_items)
            if heading:
                if total_parts > 1:
                    chunk_text = f"{heading} [Part {part_num}/{total_parts}]\n{chunk_content}"
                else:
                    chunk_text = f"{heading}\n{chunk_content}"
            else:
                chunk_text = chunk_content

            chunks.append({
                "chunk_id": deterministic_chunk_id(pdf_name, page_no, chunk_text, "list"),
                "content_type": "list",
                "text": chunk_text,
                "pdf_name": pdf_name,
                "page_no": page_no,
                "position": position + ((part_num - 1) * 0.001),
                "chunk_number": chunk_number + len(chunks),
                "image_link": "",
                "table_link": "",
                "context_before_id": "",
                "context_after_id": "",
                "metadata": {
                    "chunking_method": "list_with_heading",
                    "item_count": len(current_items),
                    "heading": heading,
                    "is_complete_list": False,
                    "is_split": True,
                    "part": part_num,
                    "total_parts": total_parts
                }
            })
            part_num += 1
            current_items = []
            current_word_count = 0

        current_items.append(item)
        current_word_count += i_words

    # Final chunk
    if current_items:
        chunk_content = '\n'.join(current_items)
        if heading:
            if total_parts > 1:
                chunk_text = f"{heading} [Part {part_num}/{total_parts}]\n{chunk_content}"
            else:
                chunk_text = f"{heading}\n{chunk_content}"
        else:
            chunk_text = chunk_content

        chunks.append({
            "chunk_id": deterministic_chunk_id(pdf_name, page_no, chunk_text, "list"),
            "content_type": "list",
            "text": chunk_text,
            "pdf_name": pdf_name,
            "page_no": page_no,
            "position": position + ((part_num - 1) * 0.001),
            "chunk_number": chunk_number + len(chunks),
            "image_link": "",
            "table_link": "",
            "context_before_id": "",
            "context_after_id": "",
            "metadata": {
                "chunking_method": "list_with_heading",
                "item_count": len(current_items),
                "heading": heading,
                "is_complete_list": False,
                "is_split": total_parts > 1,
                "part": part_num,
                "total_parts": total_parts
            }
        })

    return chunks


def chunk_list_separate(
    list_text: str,
    pdf_name: str,
    page_no: int,
    position: float,
    chunk_number: int,
    max_words: int = 500
) -> List[Dict[str, Any]]:
    """
    Create list as a separate chunk.
    If list exceeds max_words, split at item boundaries.
    Wrapper for backward compatibility - calls chunk_list_with_heading.
    """
    return chunk_list_with_heading(
        list_text, pdf_name, page_no, position, chunk_number, max_words, heading=None
    )


def chunk_heading_with_subpoints(
    heading: str,
    subpoints: List[str],
    pdf_name: str,
    page_no: int,
    position: float,
    chunk_number: int,
    max_words: int = 500
) -> List[Dict[str, Any]]:
    """
    Chunk content with heading and subpoints.

    If total > 500 words, split subpoints but keep heading common in all chunks.
    """
    all_content = heading + '\n' + '\n'.join(subpoints)
    total_words = len(all_content.split())
    heading_words = len(heading.split())

    # If fits in one chunk
    if total_words <= max_words:
        return [{
            "chunk_id": deterministic_chunk_id(pdf_name, page_no, all_content, "paragraph"),
            "content_type": "paragraph",
            "text": all_content,
            "pdf_name": pdf_name,
            "page_no": page_no,
            "position": position,
            "chunk_number": chunk_number,
            "image_link": "",
            "table_link": "",
            "context_before_id": "",
            "context_after_id": "",
            "metadata": {
                "chunking_method": "heading_with_subpoints",
                "heading": heading,
                "subpoint_count": len(subpoints),
                "is_split": False,
                "part": 1,
                "total_parts": 1
            }
        }]

    # Split subpoints while keeping heading
    chunks = []
    effective_max = max_words - heading_words - 5
    current_subpoints = []
    current_word_count = 0
    part_num = 1

    # Count total parts
    total_parts = 0
    temp_sp = []
    temp_count = 0
    for sp in subpoints:
        sp_words = len(sp.split())
        if temp_count + sp_words > effective_max and temp_sp:
            total_parts += 1
            temp_sp = []
            temp_count = 0
        temp_sp.append(sp)
        temp_count += sp_words
    if temp_sp:
        total_parts += 1

    # Create chunks
    for subpoint in subpoints:
        sp_words = len(subpoint.split())

        if current_word_count + sp_words > effective_max and current_subpoints:
            chunk_content = '\n'.join(current_subpoints)
            chunk_text = f"{heading} [Part {part_num}/{total_parts}]\n{chunk_content}"

            chunks.append({
                "chunk_id": deterministic_chunk_id(pdf_name, page_no, chunk_text, "paragraph"),
                "content_type": "paragraph",
                "text": chunk_text,
                "pdf_name": pdf_name,
                "page_no": page_no,
                "position": position + ((part_num - 1) * 0.001),
                "chunk_number": chunk_number + len(chunks),
                "image_link": "",
                "table_link": "",
                "context_before_id": "",
                "context_after_id": "",
                "metadata": {
                    "chunking_method": "heading_with_subpoints",
                    "heading": heading,
                    "subpoint_count": len(current_subpoints),
                    "is_split": True,
                    "part": part_num,
                    "total_parts": total_parts
                }
            })
            part_num += 1
            current_subpoints = []
            current_word_count = 0

        current_subpoints.append(subpoint)
        current_word_count += sp_words

    # Final chunk
    if current_subpoints:
        chunk_content = '\n'.join(current_subpoints)
        if total_parts > 1:
            chunk_text = f"{heading} [Part {part_num}/{total_parts}]\n{chunk_content}"
        else:
            chunk_text = all_content

        chunks.append({
            "chunk_id": deterministic_chunk_id(pdf_name, page_no, chunk_text, "paragraph"),
            "content_type": "paragraph",
            "text": chunk_text,
            "pdf_name": pdf_name,
            "page_no": page_no,
            "position": position + ((part_num - 1) * 0.001),
            "chunk_number": chunk_number + len(chunks),
            "image_link": "",
            "table_link": "",
            "context_before_id": "",
            "context_after_id": "",
            "metadata": {
                "chunking_method": "heading_with_subpoints",
                "heading": heading,
                "subpoint_count": len(current_subpoints),
                "is_split": total_parts > 1,
                "part": part_num,
                "total_parts": total_parts
            }
        })

    return chunks


# Legacy function - kept for backward compatibility
def _chunk_list_separate_legacy(
    list_text: str,
    pdf_name: str,
    page_no: int,
    position: float,
    chunk_number: int,
    max_words: int = 500
) -> List[Dict[str, Any]]:
    """Legacy list chunking - use chunk_list_with_heading instead."""
    lines = list_text.strip().split('\n')
    items = [l for l in lines if l.strip()]

    total_words = len(list_text.split())

    # If fits in one chunk, keep together
    if total_words <= max_words:
        return [{
            "chunk_id": deterministic_chunk_id(pdf_name, page_no, list_text, "list"),
            "content_type": "list",
            "text": list_text,
            "pdf_name": pdf_name,
            "page_no": page_no,
            "position": position,
            "chunk_number": chunk_number,
            "image_link": "",
            "table_link": "",
            "context_before_id": "",
            "context_after_id": "",
            "metadata": {
                "chunking_method": "list_separate",
                "item_count": len(items),
                "is_complete_list": True
            }
        }]

    # Split large list
    chunks = []
    current_items = []
    current_word_count = 0
    sub_position = 0

    for item in items:
        item_words = len(item.split())

        if current_word_count + item_words > max_words and current_items:
            chunk_text = '\n'.join(current_items)
            chunks.append({
                "chunk_id": deterministic_chunk_id(pdf_name, page_no, chunk_text, "list"),
                "content_type": "list",
                "text": chunk_text,
                "pdf_name": pdf_name,
                "page_no": page_no,
                "position": position + (sub_position * 0.001),
                "chunk_number": chunk_number + len(chunks),
                "image_link": "",
                "table_link": "",
                "context_before_id": "",
                "context_after_id": "",
                "metadata": {
                    "chunking_method": "list_separate",
                    "item_count": len(current_items),
                    "is_complete_list": False,
                    "list_part": sub_position + 1
                }
            })
            sub_position += 1
            current_items = []
            current_word_count = 0

        current_items.append(item)
        current_word_count += item_words

    # Final chunk
    if current_items:
        chunk_text = '\n'.join(current_items)
        chunks.append({
            "chunk_id": deterministic_chunk_id(pdf_name, page_no, chunk_text, "list"),
            "content_type": "list",
            "text": chunk_text,
            "pdf_name": pdf_name,
            "page_no": page_no,
            "position": position + (sub_position * 0.001),
            "chunk_number": chunk_number + len(chunks),
            "image_link": "",
            "table_link": "",
            "context_before_id": "",
            "context_after_id": "",
            "metadata": {
                "chunking_method": "list_separate",
                "item_count": len(current_items),
                "is_complete_list": False,
                "list_part": sub_position + 1
            }
        })

    return chunks


def chunk_toc_separate(
    toc_block: Dict[str, Any],
    pdf_name: str,
    page_no: int,
    chunk_number: int
) -> Dict[str, Any]:
    """Create Table of Contents as a separate chunk."""
    content = toc_block.get("content", "")

    return {
        "chunk_id": deterministic_chunk_id(pdf_name, page_no, content, "toc"),
        "content_type": "toc",
        "text": content,
        "pdf_name": pdf_name,
        "page_no": page_no,
        "position": 0,
        "chunk_number": chunk_number,
        "image_link": "",
        "table_link": "",
        "context_before_id": "",
        "context_after_id": "",
        "metadata": {
            "chunking_method": "toc_separate",
            "is_table_of_contents": True,
            **toc_block.get("metadata", {})
        }
    }


def detect_heading_for_content(text: str) -> Tuple[Optional[str], str]:
    """
    Detect if text starts with a heading and return (heading, remaining_text).

    Headings are detected as:
    - Lines that are short (< 100 chars) and end without period
    - Lines matching heading patterns (Roman numerals, numbers, etc.)
    - All-caps short lines
    """
    if not text:
        return None, ""

    lines = text.strip().split('\n')
    if not lines:
        return None, ""

    first_line = lines[0].strip()

    # Check if first line looks like a heading
    is_heading_line = False

    # Short line without period at end
    if len(first_line) < 100 and first_line and first_line[-1] not in '.!?':
        # Check if it's a heading pattern
        if is_heading(first_line):
            is_heading_line = True
        # All caps short line
        elif first_line.isupper() and len(first_line.split()) <= 10:
            is_heading_line = True
        # Title case short line
        elif first_line.istitle() and len(first_line.split()) <= 8:
            is_heading_line = True
        # Ends with colon
        elif first_line.endswith(':'):
            is_heading_line = True

    if is_heading_line:
        remaining = '\n'.join(lines[1:]).strip()
        return first_line, remaining

    return None, text


def extract_heading_with_subpoints(text: str) -> Tuple[Optional[str], List[str], str]:
    """
    Extract heading with its subpoints (list items or numbered points).

    Returns:
        (heading, subpoints, remaining_text)
    """
    if not text:
        return None, [], ""

    lines = text.strip().split('\n')
    heading = None
    subpoints = []
    remaining_start = 0

    # Find heading
    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            continue

        # Check if this is a heading
        if is_heading(stripped) or (len(stripped) < 100 and stripped[-1] not in '.!?' if stripped else False):
            heading = stripped
            remaining_start = i + 1
            break

    if not heading:
        return None, [], text

    # Collect subpoints (list items following the heading)
    for i in range(remaining_start, len(lines)):
        stripped = lines[i].strip()
        if not stripped:
            continue

        if is_list_item(stripped):
            subpoints.append(stripped)
        elif is_heading(stripped):
            # New heading - stop collecting
            remaining_start = i
            break
        else:
            # Regular text - stop collecting subpoints
            remaining_start = i
            break
    else:
        remaining_start = len(lines)

    remaining = '\n'.join(lines[remaining_start:]).strip()
    return heading, subpoints, remaining


def process_document_with_strategy(
    pages: List[Dict[str, Any]],
    pdf_name: str,
    max_words: int = 500,
    min_words: int = 50
) -> List[Dict[str, Any]]:
    """
    Process document with the new chunking strategy.

    Strategy:
    - Paragraph chunks (max 500 words, min 50 words)
    - Table summary as separate chunk (with header preserved in splits)
    - Image summary as separate chunk (with caption preserved in splits)
    - List as separate chunk (with heading preserved in splits)
    - Table of Contents as separate chunk (with header preserved in splits)
    - Remove page numbers (don't keep)
    - Keep cross-page references
    - Preserve PDF structure by keeping headers/links in all split chunks

    Args:
        pages: List of page dictionaries with 'text', 'page_no', 'blocks', etc.
        pdf_name: Name of the PDF document
        max_words: Maximum words per chunk (default 500)
        min_words: Minimum words per chunk (default 50)

    Returns:
        List of chunk dictionaries ready for embedding and indexing
    """
    # Step 1: Cross-page merging
    pages = merge_cross_page_content(pages)
    pages = merge_split_lists(pages)

    all_chunks = []
    chunk_number = 0
    toc_extracted = False
    current_heading = None  # Track current section heading

    for page in pages:
        page_no = page.get("page_no", page.get("metadata", {}).get("page_no", 0))
        text = page.get("text", "")
        blocks = page.get("blocks", [])

        # Step 2: Remove page numbers from text
        if text:
            text = remove_page_numbers(text)

        # Step 3: Extract Table of Contents (only once, usually on early pages)
        if not toc_extracted and text:
            text, toc_block = extract_table_of_contents(text)
            if toc_block:
                # Use structure-preserving TOC chunking
                toc_content = toc_block.get("content", "")
                toc_chunks = chunk_toc_with_structure(
                    toc_content, pdf_name, page_no, chunk_number, max_words
                )
                all_chunks.extend(toc_chunks)
                chunk_number += len(toc_chunks)
                toc_extracted = True

        # Step 4: Extract tables and figures
        remaining_text = text
        table_figure_blocks = []
        if text:
            remaining_text, table_figure_blocks = extract_tables_and_figures(text)

        # Step 5: Process content with heading awareness
        position = 0

        # First, try to extract heading+subpoints patterns
        while remaining_text and remaining_text.strip():
            # Check for heading with subpoints
            heading, subpoints, after_subpoints = extract_heading_with_subpoints(remaining_text)

            if heading and subpoints:
                # Use chunk_heading_with_subpoints for this pattern
                heading_chunks = chunk_heading_with_subpoints(
                    heading, subpoints, pdf_name, page_no, position, chunk_number, max_words
                )
                all_chunks.extend(heading_chunks)
                chunk_number += len(heading_chunks)
                position += 1
                current_heading = heading  # Update current heading
                remaining_text = after_subpoints
                continue

            # Check for heading followed by content
            detected_heading, content_after_heading = detect_heading_for_content(remaining_text)
            if detected_heading:
                current_heading = detected_heading
                remaining_text = content_after_heading

            # Extract lists from remaining text
            content_blocks = extract_lists_from_text(remaining_text) if remaining_text else []

            if not content_blocks:
                break

            # Process each content block
            for block in content_blocks:
                block_type = block.get("type", "text")
                content = block.get("content", "").strip()

                if not content:
                    continue

                # Remove page numbers from block content
                content = remove_page_numbers(content)
                if not content.strip():
                    continue

                if block_type == "list":
                    # List with heading preserved in splits
                    list_chunks = chunk_list_with_heading(
                        content, pdf_name, page_no, position, chunk_number,
                        max_words, heading=current_heading
                    )
                    all_chunks.extend(list_chunks)
                    chunk_number += len(list_chunks)
                else:
                    # Paragraph chunk with heading context
                    para_chunks = chunk_paragraph(
                        content, pdf_name, page_no, position, chunk_number,
                        max_words, min_words, heading=current_heading
                    )
                    all_chunks.extend(para_chunks)
                    chunk_number += len(para_chunks)

                position += 1

            # We've processed all content blocks, exit the while loop
            break

        # Step 6: Add table/figure chunks with structure-preserving splits
        for tf_block in table_figure_blocks:
            tf_type = tf_block.get("type", "table")
            content = tf_block.get("content", "")
            caption = tf_block.get("caption", "")

            # Remove page numbers from table/figure content
            content = remove_page_numbers(content)
            if not content.strip():
                continue

            if tf_type == "table":
                # Use structure-preserving table chunking
                table_chunks = chunk_table_with_structure(
                    content, pdf_name, page_no, position, chunk_number,
                    max_words, table_link=tf_block.get("table_link", ""), caption=caption
                )
                all_chunks.extend(table_chunks)
                chunk_number += len(table_chunks)
            else:
                # Use structure-preserving figure chunking
                figure_chunks = chunk_figure_with_structure(
                    content, pdf_name, page_no, position, chunk_number,
                    max_words, image_link=tf_block.get("image_link", ""), caption=caption
                )
                all_chunks.extend(figure_chunks)
                chunk_number += len(figure_chunks)

            position += 1

        # Step 7: Process any pre-defined blocks (from PDF extraction)
        if blocks:
            for block in blocks:
                block_type = block.get("type", "text")
                content = block.get("content", "")
                caption = block.get("caption", "")

                # Remove page numbers
                content = remove_page_numbers(content)
                if not content.strip():
                    continue

                if block_type == "table":
                    # Use structure-preserving table chunking
                    table_chunks = chunk_table_with_structure(
                        content, pdf_name, page_no, position, chunk_number,
                        max_words, table_link=block.get("table_link", ""), caption=caption
                    )
                    all_chunks.extend(table_chunks)
                    chunk_number += len(table_chunks)

                elif block_type == "image":
                    # Use structure-preserving figure chunking
                    figure_chunks = chunk_figure_with_structure(
                        content, pdf_name, page_no, position, chunk_number,
                        max_words, image_link=block.get("image_link", ""), caption=caption
                    )
                    all_chunks.extend(figure_chunks)
                    chunk_number += len(figure_chunks)

                position += 1

    # Step 8: Merge small chunks (below min_words)
    all_chunks = merge_small_chunks_final(all_chunks, min_words)

    # Step 9: Link cross-page context references
    all_chunks = _link_chunk_context(all_chunks)

    # Add cross-page reference metadata
    for i, chunk in enumerate(all_chunks):
        chunk["metadata"]["has_cross_page_reference"] = False
        if i > 0:
            prev_chunk = all_chunks[i - 1]
            if prev_chunk.get("page_no") != chunk.get("page_no"):
                chunk["metadata"]["has_cross_page_reference"] = True
                chunk["metadata"]["previous_page"] = prev_chunk.get("page_no")

    return all_chunks


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
                "section_hierarchy": block.get("section_hierarchy", []),
                "heading_level": block.get("heading_level", 0),
                "table_summary": block.get("table_summary", ""),
                "image_caption": "",
                "image_summary": "",
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
                "section_hierarchy": block.get("section_hierarchy", []),
                "heading_level": block.get("heading_level", 0),
                "table_summary": "",
                "image_caption": block.get("image_caption", ""),
                "image_summary": block.get("image_summary", ""),
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
                "section_hierarchy": block.get("section_hierarchy", []),
                "heading_level": block.get("heading_level", 0),
                "table_summary": block.get("table_summary", ""),
                "image_caption": "",
                "image_summary": "",
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
                "section_hierarchy": block.get("section_hierarchy", []),
                "heading_level": block.get("heading_level", 0),
                "table_summary": "",
                "image_caption": block.get("image_caption", ""),
                "image_summary": block.get("image_summary", ""),
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


# =============================================================================
# STRUCTURE-BASED CHUNKING (Section Hierarchy Aware)
# =============================================================================
# These functions create chunks that know their position in the document
# hierarchy, with section context in every chunk.
# =============================================================================

def get_section_prefix(section_hierarchy: List[str]) -> str:
    """
    Generate section prefix string for chunk text.

    Args:
        section_hierarchy: List of section titles from root to current

    Returns:
        String like "[Section: Chapter 1 > Introduction > Background]"
    """
    if not section_hierarchy:
        return ""

    path_str = " > ".join(section_hierarchy)
    return f"[Section: {path_str}]"


def chunk_paragraph_with_section(
    text: str,
    section_path: List[str],
    pdf_name: str,
    page_no: int,
    position: float,
    chunk_number: int,
    max_words: int = 500,
    min_words: int = 50,
    heading_level: int = 0
) -> List[Dict[str, Any]]:
    """
    Chunk paragraph with [Section: path] prefix.

    Every chunk includes the section hierarchy context in its text.

    Args:
        text: Paragraph text to chunk
        section_path: List of section titles (hierarchy path)
        pdf_name: Source PDF name
        page_no: Page number
        position: Position on page
        chunk_number: Starting chunk number
        max_words: Maximum words per chunk (default 500)
        min_words: Minimum words per chunk (default 50)
        heading_level: Heading level if this is a heading (0=body)

    Returns:
        List of chunk dictionaries with section_hierarchy field
    """
    sentences = sentence_split(text)
    chunks = []

    if not sentences:
        return chunks

    # Generate section prefix
    section_prefix = get_section_prefix(section_path)
    prefix_words = len(section_prefix.split()) if section_prefix else 0

    # Reserve space for prefix in each chunk
    effective_max = max_words - prefix_words - 2  # 2 for newlines

    current_sentences = []
    current_word_count = 0
    sub_position = 0

    # First pass: count total parts
    temp_sentences = []
    temp_word_count = 0
    total_parts = 0
    for sentence in sentences:
        word_count = len(sentence.split())
        if temp_word_count + word_count > effective_max and temp_sentences:
            total_parts += 1
            temp_sentences = []
            temp_word_count = 0
        temp_sentences.append(sentence)
        temp_word_count += word_count
    if temp_sentences:
        total_parts += 1

    # Second pass: create chunks
    part_num = 1
    for sentence in sentences:
        word_count = len(sentence.split())

        # If adding this sentence exceeds max, save current chunk
        if current_word_count + word_count > effective_max and current_sentences:
            chunk_text = ' '.join(current_sentences)

            # Add section prefix
            if section_prefix:
                chunk_text = f"{section_prefix}\n\n{chunk_text}"

            chunks.append({
                "chunk_id": deterministic_chunk_id(pdf_name, page_no, chunk_text, "paragraph"),
                "content_type": "paragraph",
                "text": chunk_text,
                "pdf_name": pdf_name,
                "page_no": page_no,
                "position": position + (sub_position * 0.001),
                "chunk_number": chunk_number + len(chunks),
                "image_link": "",
                "table_link": "",
                "context_before_id": "",
                "context_after_id": "",
                "section_hierarchy": section_path.copy(),
                "heading_level": heading_level,
                "table_summary": "",
                "image_caption": "",
                "image_summary": "",
                "metadata": {
                    "chunking_method": "structure_based",
                    "word_count": current_word_count + prefix_words,
                    "sentence_count": len(current_sentences),
                    "part": part_num,
                    "total_parts": total_parts,
                    "is_split": total_parts > 1
                }
            })
            part_num += 1
            sub_position += 1
            current_sentences = []
            current_word_count = 0

        current_sentences.append(sentence)
        current_word_count += word_count

    # Save final chunk
    if current_sentences:
        chunk_text = ' '.join(current_sentences)

        # Add section prefix
        if section_prefix:
            chunk_text = f"{section_prefix}\n\n{chunk_text}"

        chunks.append({
            "chunk_id": deterministic_chunk_id(pdf_name, page_no, chunk_text, "paragraph"),
            "content_type": "paragraph",
            "text": chunk_text,
            "pdf_name": pdf_name,
            "page_no": page_no,
            "position": position + (sub_position * 0.001),
            "chunk_number": chunk_number + len(chunks),
            "image_link": "",
            "table_link": "",
            "context_before_id": "",
            "context_after_id": "",
            "section_hierarchy": section_path.copy(),
            "heading_level": heading_level,
            "table_summary": "",
            "image_caption": "",
            "image_summary": "",
            "metadata": {
                "chunking_method": "structure_based",
                "word_count": current_word_count + prefix_words,
                "sentence_count": len(current_sentences),
                "part": part_num,
                "total_parts": total_parts,
                "is_split": total_parts > 1
            }
        })

    return chunks


def chunk_table_with_section(
    content: str,
    section_path: List[str],
    pdf_name: str,
    page_no: int,
    position: float,
    chunk_number: int,
    max_words: int = 500,
    table_link: str = "",
    caption: str = "",
    table_summary: str = ""
) -> List[Dict[str, Any]]:
    """
    Chunk table preserving header row in all splits.

    Format: "Table N: Caption [Part X/Y]"

    Args:
        content: Table content (markdown or text)
        section_path: Section hierarchy path
        pdf_name: Source PDF name
        page_no: Page number
        position: Position on page
        chunk_number: Starting chunk number
        max_words: Maximum words per chunk
        table_link: Link to table image
        caption: Table caption
        table_summary: Vision model generated table summary

    Returns:
        List of chunk dictionaries
    """
    total_words = len(content.split())

    # Extract header (first line or caption)
    lines = content.strip().split('\n')
    header = caption if caption else (lines[0] if lines else "")

    # Add section context to header
    section_prefix = get_section_prefix(section_path)
    header_with_context = f"{section_prefix}\n{header}" if section_prefix and header else (section_prefix or header)
    header_words = len(header_with_context.split()) if header_with_context else 0

    # If fits in one chunk
    if total_words + header_words <= max_words:
        full_text = content
        if header_with_context and header_with_context not in content:
            full_text = f"{header_with_context}\n{content}" if header else f"{section_prefix}\n\n{content}"
        elif section_prefix and section_prefix not in content:
            full_text = f"{section_prefix}\n\n{content}"

        return [{
            "chunk_id": deterministic_chunk_id(pdf_name, page_no, full_text, "table"),
            "content_type": "table",
            "text": full_text,
            "pdf_name": pdf_name,
            "page_no": page_no,
            "position": position,
            "chunk_number": chunk_number,
            "image_link": "",
            "table_link": table_link,
            "context_before_id": "",
            "context_after_id": "",
            "section_hierarchy": section_path.copy(),
            "heading_level": 0,
            "table_summary": table_summary,
            "image_caption": "",
            "image_summary": "",
            "metadata": {
                "chunking_method": "structure_based",
                "caption": caption,
                "is_table": True,
                "is_split": False,
                "part": 1,
                "total_parts": 1
            }
        }]

    # Split table while preserving header
    chunks = []
    effective_max = max_words - header_words - 10

    rows = lines[1:] if caption else lines
    current_rows = []
    current_word_count = 0
    part_num = 1

    # Count total parts
    total_parts = 0
    temp_rows = []
    temp_count = 0
    for row in rows:
        row_words = len(row.split())
        if temp_count + row_words > effective_max and temp_rows:
            total_parts += 1
            temp_rows = []
            temp_count = 0
        temp_rows.append(row)
        temp_count += row_words
    if temp_rows:
        total_parts += 1

    for row in rows:
        row_words = len(row.split())

        if current_word_count + row_words > effective_max and current_rows:
            chunk_content = '\n'.join(current_rows)
            part_header = f"{header} [Part {part_num}/{total_parts}]"
            if section_prefix:
                chunk_text = f"{section_prefix}\n{part_header}\n{chunk_content}"
            else:
                chunk_text = f"{part_header}\n{chunk_content}"

            chunks.append({
                "chunk_id": deterministic_chunk_id(pdf_name, page_no, chunk_text, "table"),
                "content_type": "table",
                "text": chunk_text,
                "pdf_name": pdf_name,
                "page_no": page_no,
                "position": position + ((part_num - 1) * 0.001),
                "chunk_number": chunk_number + len(chunks),
                "image_link": "",
                "table_link": table_link,
                "context_before_id": "",
                "context_after_id": "",
                "section_hierarchy": section_path.copy(),
                "heading_level": 0,
                "table_summary": table_summary,
                "image_caption": "",
                "image_summary": "",
                "metadata": {
                    "chunking_method": "structure_based",
                    "caption": caption,
                    "is_table": True,
                    "is_split": True,
                    "part": part_num,
                    "total_parts": total_parts,
                    "original_header": header
                }
            })
            part_num += 1
            current_rows = []
            current_word_count = 0

        current_rows.append(row)
        current_word_count += row_words

    if current_rows:
        chunk_content = '\n'.join(current_rows)
        if total_parts > 1:
            part_header = f"{header} [Part {part_num}/{total_parts}]"
            if section_prefix:
                chunk_text = f"{section_prefix}\n{part_header}\n{chunk_content}"
            else:
                chunk_text = f"{part_header}\n{chunk_content}"
        else:
            chunk_text = content

        chunks.append({
            "chunk_id": deterministic_chunk_id(pdf_name, page_no, chunk_text, "table"),
            "content_type": "table",
            "text": chunk_text,
            "pdf_name": pdf_name,
            "page_no": page_no,
            "position": position + ((part_num - 1) * 0.001),
            "chunk_number": chunk_number + len(chunks),
            "image_link": "",
            "table_link": table_link,
            "context_before_id": "",
            "context_after_id": "",
            "section_hierarchy": section_path.copy(),
            "heading_level": 0,
            "table_summary": table_summary,
            "image_caption": "",
            "image_summary": "",
            "metadata": {
                "chunking_method": "structure_based",
                "caption": caption,
                "is_table": True,
                "is_split": total_parts > 1,
                "part": part_num,
                "total_parts": total_parts,
                "original_header": header
            }
        })

    return chunks


def chunk_list_with_section(
    items: List[str],
    section_path: List[str],
    pdf_name: str,
    page_no: int,
    position: float,
    chunk_number: int,
    max_words: int = 500,
    list_heading: str = None
) -> List[Dict[str, Any]]:
    """
    Chunk list at item boundaries with section context.

    Args:
        items: List of item strings
        section_path: Section hierarchy path
        pdf_name: Source PDF name
        page_no: Page number
        position: Position on page
        chunk_number: Starting chunk number
        max_words: Maximum words per chunk
        list_heading: Optional heading for the list

    Returns:
        List of chunk dictionaries
    """
    list_text = '\n'.join(items)
    total_words = len(list_text.split())

    section_prefix = get_section_prefix(section_path)
    prefix_words = len(section_prefix.split()) if section_prefix else 0
    heading_words = len(list_heading.split()) if list_heading else 0

    effective_max = max_words - prefix_words - heading_words - 2

    if total_words + prefix_words + heading_words <= max_words:
        parts = []
        if section_prefix:
            parts.append(section_prefix)
        if list_heading:
            parts.append(list_heading)
        parts.append(list_text)
        full_text = '\n\n'.join(parts) if parts else list_text

        return [{
            "chunk_id": deterministic_chunk_id(pdf_name, page_no, full_text, "list"),
            "content_type": "list",
            "text": full_text,
            "pdf_name": pdf_name,
            "page_no": page_no,
            "position": position,
            "chunk_number": chunk_number,
            "image_link": "",
            "table_link": "",
            "context_before_id": "",
            "context_after_id": "",
            "section_hierarchy": section_path.copy(),
            "heading_level": 0,
            "table_summary": "",
            "image_caption": "",
            "image_summary": "",
            "metadata": {
                "chunking_method": "structure_based",
                "item_count": len(items),
                "list_heading": list_heading,
                "is_complete_list": True,
                "is_split": False,
                "part": 1,
                "total_parts": 1
            }
        }]

    chunks = []
    current_items = []
    current_word_count = 0
    part_num = 1

    total_parts = 0
    temp_items = []
    temp_count = 0
    for item in items:
        i_words = len(item.split())
        if temp_count + i_words > effective_max and temp_items:
            total_parts += 1
            temp_items = []
            temp_count = 0
        temp_items.append(item)
        temp_count += i_words
    if temp_items:
        total_parts += 1

    for item in items:
        item_words = len(item.split())

        if current_word_count + item_words > effective_max and current_items:
            chunk_content = '\n'.join(current_items)
            parts = []
            if section_prefix:
                parts.append(section_prefix)
            if list_heading:
                parts.append(f"{list_heading} [Part {part_num}/{total_parts}]")
            parts.append(chunk_content)
            chunk_text = '\n\n'.join(parts)

            chunks.append({
                "chunk_id": deterministic_chunk_id(pdf_name, page_no, chunk_text, "list"),
                "content_type": "list",
                "text": chunk_text,
                "pdf_name": pdf_name,
                "page_no": page_no,
                "position": position + ((part_num - 1) * 0.001),
                "chunk_number": chunk_number + len(chunks),
                "image_link": "",
                "table_link": "",
                "context_before_id": "",
                "context_after_id": "",
                "section_hierarchy": section_path.copy(),
                "heading_level": 0,
                "table_summary": "",
                "image_caption": "",
                "image_summary": "",
                "metadata": {
                    "chunking_method": "structure_based",
                    "item_count": len(current_items),
                    "list_heading": list_heading,
                    "is_complete_list": False,
                    "is_split": True,
                    "part": part_num,
                    "total_parts": total_parts
                }
            })
            part_num += 1
            current_items = []
            current_word_count = 0

        current_items.append(item)
        current_word_count += item_words

    if current_items:
        chunk_content = '\n'.join(current_items)
        parts = []
        if section_prefix:
            parts.append(section_prefix)
        if list_heading and total_parts > 1:
            parts.append(f"{list_heading} [Part {part_num}/{total_parts}]")
        elif list_heading:
            parts.append(list_heading)
        parts.append(chunk_content)
        chunk_text = '\n\n'.join(parts)

        chunks.append({
            "chunk_id": deterministic_chunk_id(pdf_name, page_no, chunk_text, "list"),
            "content_type": "list",
            "text": chunk_text,
            "pdf_name": pdf_name,
            "page_no": page_no,
            "position": position + ((part_num - 1) * 0.001),
            "chunk_number": chunk_number + len(chunks),
            "image_link": "",
            "table_link": "",
            "context_before_id": "",
            "context_after_id": "",
            "section_hierarchy": section_path.copy(),
            "heading_level": 0,
            "table_summary": "",
            "image_caption": "",
            "image_summary": "",
            "metadata": {
                "chunking_method": "structure_based",
                "item_count": len(current_items),
                "list_heading": list_heading,
                "is_complete_list": False,
                "is_split": total_parts > 1,
                "part": part_num,
                "total_parts": total_parts
            }
        })

    return chunks


def chunk_figure_with_section(
    content: str,
    section_path: List[str],
    pdf_name: str,
    page_no: int,
    position: float,
    chunk_number: int,
    max_words: int = 500,
    image_link: str = "",
    caption: str = "",
    image_caption: str = "",
    image_summary: str = ""
) -> List[Dict[str, Any]]:
    """
    Chunk figure with caption preserved.

    Args:
        content: Figure description/content
        section_path: Section hierarchy path
        pdf_name: Source PDF name
        page_no: Page number
        position: Position on page
        chunk_number: Starting chunk number
        max_words: Maximum words per chunk
        image_link: Link to figure image
        caption: Figure caption
        image_caption: Vision model generated short caption
        image_summary: Vision model generated detailed summary

    Returns:
        List of chunk dictionaries
    """
    total_words = len(content.split())

    section_prefix = get_section_prefix(section_path)
    header = caption if caption else ""
    header_with_context = f"{section_prefix}\n{header}" if section_prefix and header else (section_prefix or header)
    header_words = len(header_with_context.split()) if header_with_context else 0

    if total_words + header_words <= max_words:
        full_text = content
        if header_with_context and header_with_context not in content:
            full_text = f"{header_with_context}\n\n{content}"
        elif section_prefix and section_prefix not in content:
            full_text = f"{section_prefix}\n\n{content}"

        return [{
            "chunk_id": deterministic_chunk_id(pdf_name, page_no, full_text, "figure"),
            "content_type": "figure",
            "text": full_text,
            "pdf_name": pdf_name,
            "page_no": page_no,
            "position": position,
            "chunk_number": chunk_number,
            "image_link": image_link,
            "table_link": "",
            "context_before_id": "",
            "context_after_id": "",
            "section_hierarchy": section_path.copy(),
            "heading_level": 0,
            "table_summary": "",
            "image_caption": image_caption,
            "image_summary": image_summary,
            "metadata": {
                "chunking_method": "structure_based",
                "caption": caption,
                "is_figure": True,
                "is_split": False,
                "part": 1,
                "total_parts": 1
            }
        }]

    chunks = []
    sentences = sentence_split(content)
    effective_max = max_words - header_words - 10

    current_sentences = []
    current_word_count = 0
    part_num = 1

    total_parts = 0
    temp_sentences = []
    temp_count = 0
    for sentence in sentences:
        s_words = len(sentence.split())
        if temp_count + s_words > effective_max and temp_sentences:
            total_parts += 1
            temp_sentences = []
            temp_count = 0
        temp_sentences.append(sentence)
        temp_count += s_words
    if temp_sentences:
        total_parts += 1

    for sentence in sentences:
        s_words = len(sentence.split())

        if current_word_count + s_words > effective_max and current_sentences:
            chunk_content = ' '.join(current_sentences)
            part_header = f"{header} [Part {part_num}/{total_parts}]" if header else f"[Part {part_num}/{total_parts}]"
            if section_prefix:
                chunk_text = f"{section_prefix}\n{part_header}\n\n{chunk_content}"
            else:
                chunk_text = f"{part_header}\n\n{chunk_content}"

            chunks.append({
                "chunk_id": deterministic_chunk_id(pdf_name, page_no, chunk_text, "figure"),
                "content_type": "figure",
                "text": chunk_text,
                "pdf_name": pdf_name,
                "page_no": page_no,
                "position": position + ((part_num - 1) * 0.001),
                "chunk_number": chunk_number + len(chunks),
                "image_link": image_link,
                "table_link": "",
                "context_before_id": "",
                "context_after_id": "",
                "section_hierarchy": section_path.copy(),
                "heading_level": 0,
                "table_summary": "",
                "image_caption": image_caption,
                "image_summary": image_summary,
                "metadata": {
                    "chunking_method": "structure_based",
                    "caption": caption,
                    "is_figure": True,
                    "is_split": True,
                    "part": part_num,
                    "total_parts": total_parts,
                    "original_caption": header
                }
            })
            part_num += 1
            current_sentences = []
            current_word_count = 0

        current_sentences.append(sentence)
        current_word_count += s_words

    if current_sentences:
        chunk_content = ' '.join(current_sentences)
        if total_parts > 1:
            part_header = f"{header} [Part {part_num}/{total_parts}]" if header else f"[Part {part_num}/{total_parts}]"
            if section_prefix:
                chunk_text = f"{section_prefix}\n{part_header}\n\n{chunk_content}"
            else:
                chunk_text = f"{part_header}\n\n{chunk_content}"
        else:
            chunk_text = content

        chunks.append({
            "chunk_id": deterministic_chunk_id(pdf_name, page_no, chunk_text, "figure"),
            "content_type": "figure",
            "text": chunk_text,
            "pdf_name": pdf_name,
            "page_no": page_no,
            "position": position + ((part_num - 1) * 0.001),
            "chunk_number": chunk_number + len(chunks),
            "image_link": image_link,
            "table_link": "",
            "context_before_id": "",
            "context_after_id": "",
            "section_hierarchy": section_path.copy(),
            "heading_level": 0,
            "table_summary": "",
            "image_caption": image_caption,
            "image_summary": image_summary,
            "metadata": {
                "chunking_method": "structure_based",
                "caption": caption,
                "is_figure": True,
                "is_split": total_parts > 1,
                "part": part_num,
                "total_parts": total_parts,
                "original_caption": header
            }
        })

    return chunks


def process_document_with_structure(
    pages: List[Dict[str, Any]],
    pdf_name: str,
    max_words: int = 500,
    min_words: int = 50
) -> List[Dict[str, Any]]:
    """
    Main entry point for structure-based chunking.

    1. Detect headings and build hierarchy from blocks
    2. Create chunks with section context

    Args:
        pages: List of page dictionaries with 'blocks' containing
               section_hierarchy and heading_level fields
        pdf_name: Name of the PDF document
        max_words: Maximum words per chunk (default 500)
        min_words: Minimum words per chunk (default 50)

    Returns:
        List of chunk dictionaries with section_hierarchy field
    """
    pages = merge_cross_page_content(pages)
    pages = merge_split_lists(pages)

    all_chunks = []
    chunk_number = 0
    current_section_path = []

    for page in pages:
        page_no = page.get("page_no", page.get("metadata", {}).get("page_no", 0))
        blocks = page.get("blocks", [])
        text = page.get("text", "")

        if text:
            text = remove_page_numbers(text)

        if blocks:
            position = 0

            for block in blocks:
                block_type = block.get("type", "text")
                content = block.get("content", "")
                heading_level = block.get("heading_level", 0) or 0
                block_section_path = block.get("section_hierarchy", [])

                if block_section_path:
                    current_section_path = block_section_path.copy()
                elif heading_level > 0 and content:
                    heading_text = content.strip()[:200]
                    if heading_level == 1:
                        current_section_path = [heading_text]
                    elif heading_level == 2:
                        current_section_path = current_section_path[:1] + [heading_text]
                    elif heading_level == 3:
                        current_section_path = current_section_path[:2] + [heading_text]

                section_path = current_section_path.copy()

                if not content or not content.strip():
                    continue

                content = remove_page_numbers(content)
                if not content.strip():
                    continue

                if block_type == "text":
                    content_blocks = extract_lists_from_text(content)

                    for cb in content_blocks:
                        cb_type = cb.get("type", "text")
                        cb_content = cb.get("content", "").strip()

                        if not cb_content:
                            continue

                        if cb_type == "list":
                            items = cb.get("items", cb_content.split('\n'))
                            list_chunks = chunk_list_with_section(
                                items=items,
                                section_path=section_path,
                                pdf_name=pdf_name,
                                page_no=page_no,
                                position=position,
                                chunk_number=chunk_number,
                                max_words=max_words
                            )
                            all_chunks.extend(list_chunks)
                            chunk_number += len(list_chunks)
                        else:
                            para_chunks = chunk_paragraph_with_section(
                                text=cb_content,
                                section_path=section_path,
                                pdf_name=pdf_name,
                                page_no=page_no,
                                position=position,
                                chunk_number=chunk_number,
                                max_words=max_words,
                                min_words=min_words,
                                heading_level=heading_level
                            )
                            all_chunks.extend(para_chunks)
                            chunk_number += len(para_chunks)

                        position += 1

                elif block_type == "table":
                    table_chunks = chunk_table_with_section(
                        content=content,
                        section_path=section_path,
                        pdf_name=pdf_name,
                        page_no=page_no,
                        position=position,
                        chunk_number=chunk_number,
                        max_words=max_words,
                        table_link=block.get("table_link", ""),
                        caption=block.get("caption", ""),
                        table_summary=block.get("table_summary", "")
                    )
                    all_chunks.extend(table_chunks)
                    chunk_number += len(table_chunks)
                    position += 1

                elif block_type == "image":
                    figure_chunks = chunk_figure_with_section(
                        content=content or "Image",
                        section_path=section_path,
                        pdf_name=pdf_name,
                        page_no=page_no,
                        position=position,
                        chunk_number=chunk_number,
                        max_words=max_words,
                        image_link=block.get("image_link", ""),
                        caption=block.get("caption", ""),
                        image_caption=block.get("image_caption", ""),
                        image_summary=block.get("image_summary", "")
                    )
                    all_chunks.extend(figure_chunks)
                    chunk_number += len(figure_chunks)
                    position += 1

        elif text:
            text, toc_block = extract_table_of_contents(text)
            if toc_block:
                toc_chunks = chunk_toc_with_structure(
                    toc_block.get("content", ""),
                    pdf_name, page_no, chunk_number, max_words
                )
                for tc in toc_chunks:
                    tc["section_hierarchy"] = []
                    tc["heading_level"] = 0
                all_chunks.extend(toc_chunks)
                chunk_number += len(toc_chunks)

            remaining_text, table_figure_blocks = extract_tables_and_figures(text)

            if remaining_text.strip():
                content_blocks = extract_lists_from_text(remaining_text)
                position = 0

                for block in content_blocks:
                    block_type = block.get("type", "text")
                    content = block.get("content", "").strip()

                    if not content:
                        continue

                    content = remove_page_numbers(content)
                    if not content.strip():
                        continue

                    if block_type == "list":
                        items = block.get("items", content.split('\n'))
                        list_chunks = chunk_list_with_section(
                            items=items,
                            section_path=current_section_path,
                            pdf_name=pdf_name,
                            page_no=page_no,
                            position=position,
                            chunk_number=chunk_number,
                            max_words=max_words
                        )
                        all_chunks.extend(list_chunks)
                        chunk_number += len(list_chunks)
                    else:
                        para_chunks = chunk_paragraph_with_section(
                            text=content,
                            section_path=current_section_path,
                            pdf_name=pdf_name,
                            page_no=page_no,
                            position=position,
                            chunk_number=chunk_number,
                            max_words=max_words,
                            min_words=min_words
                        )
                        all_chunks.extend(para_chunks)
                        chunk_number += len(para_chunks)

                    position += 1

            for tf_block in table_figure_blocks:
                tf_type = tf_block.get("type", "table")
                content = tf_block.get("content", "")
                content = remove_page_numbers(content)

                if not content.strip():
                    continue

                if tf_type == "table":
                    table_chunks = chunk_table_with_section(
                        content=content,
                        section_path=current_section_path,
                        pdf_name=pdf_name,
                        page_no=page_no,
                        position=position,
                        chunk_number=chunk_number,
                        max_words=max_words,
                        caption=tf_block.get("caption", "")
                    )
                    all_chunks.extend(table_chunks)
                    chunk_number += len(table_chunks)
                else:
                    figure_chunks = chunk_figure_with_section(
                        content=content,
                        section_path=current_section_path,
                        pdf_name=pdf_name,
                        page_no=page_no,
                        position=position,
                        chunk_number=chunk_number,
                        max_words=max_words,
                        caption=tf_block.get("caption", "")
                    )
                    all_chunks.extend(figure_chunks)
                    chunk_number += len(figure_chunks)

                position += 1

    all_chunks = merge_small_chunks_final(all_chunks, min_words)
    all_chunks = _link_chunk_context(all_chunks)

    for i, chunk in enumerate(all_chunks):
        if "metadata" not in chunk:
            chunk["metadata"] = {}
        chunk["metadata"]["has_cross_page_reference"] = False
        if i > 0:
            prev_chunk = all_chunks[i - 1]
            if prev_chunk.get("page_no") != chunk.get("page_no"):
                chunk["metadata"]["has_cross_page_reference"] = True
                chunk["metadata"]["previous_page"] = prev_chunk.get("page_no")

    return all_chunks
