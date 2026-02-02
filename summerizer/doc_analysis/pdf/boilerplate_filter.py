"""
Header/Footer and Boilerplate Detection.

Detects and filters repeating elements across pages:
- Page numbers
- Document headers (repeating title, date, etc.)
- Document footers (confidentiality notices, etc.)
- Watermarks

Detection strategy:
1. Group blocks by normalized text content
2. Check if text appears on threshold % of pages
3. Check if position is consistent (top/bottom of page)
"""

import re
from collections import defaultdict
from typing import List, Dict, Set, Tuple

from doc_analysis.config import ChunkingConfig, DEFAULT_CHUNKING_CONFIG


# Common boilerplate patterns
PAGE_NUMBER_PATTERNS = [
    re.compile(r'^[\s]*\d+[\s]*$'),                    # Just number: "42"
    re.compile(r'^[\s]*Page\s+\d+[\s]*$', re.I),       # "Page 42"
    re.compile(r'^[\s]*\d+\s+of\s+\d+[\s]*$', re.I),   # "42 of 100"
    re.compile(r'^[\s]*-\s*\d+\s*-[\s]*$'),            # "- 42 -"
    re.compile(r'^[\s]*\[\d+\][\s]*$'),                # "[42]"
]

COMMON_FOOTER_PATTERNS = [
    re.compile(r'confidential', re.I),
    re.compile(r'proprietary', re.I),
    re.compile(r'all rights reserved', re.I),
    re.compile(r'copyright\s+©?\s*\d{4}', re.I),
    re.compile(r'^\s*©\s*\d{4}', re.I),
    re.compile(r'draft|internal use only', re.I),
]


def _normalize_text(text: str) -> str:
    """Normalize text for comparison (remove numbers, lowercase)."""
    # Remove page numbers that might vary
    normalized = re.sub(r'\d+', '#', text.lower().strip())
    # Remove extra whitespace
    normalized = re.sub(r'\s+', ' ', normalized)
    return normalized


def _is_page_number(text: str) -> bool:
    """Check if text is a page number."""
    text = text.strip()
    for pattern in PAGE_NUMBER_PATTERNS:
        if pattern.match(text):
            return True
    return False


def _is_common_footer(text: str) -> bool:
    """Check if text matches common footer patterns."""
    for pattern in COMMON_FOOTER_PATTERNS:
        if pattern.search(text):
            return True
    return False


def _get_vertical_position(block: Dict) -> str:
    """
    Determine vertical position of block on page.

    Returns: 'header', 'footer', or 'body'
    """
    bbox = block.get("bbox", [])
    page_height = block.get("page_height", 792)  # Default letter size

    if not bbox or len(bbox) < 4:
        return "body"

    y_top = bbox[1]
    y_bottom = bbox[3]

    # Header zone: top 10% of page
    header_threshold = page_height * 0.10

    # Footer zone: bottom 10% of page
    footer_threshold = page_height * 0.90

    if y_top < header_threshold:
        return "header"
    elif y_bottom > footer_threshold:
        return "footer"
    else:
        return "body"


def detect_boilerplate(
    blocks: List[Dict],
    config: ChunkingConfig = None,
) -> Tuple[Set[str], Dict[str, str]]:
    """
    Detect boilerplate text across pages.

    Args:
        blocks: List of text blocks with page_no, text, bbox
        config: Chunking configuration

    Returns:
        Tuple of:
        - Set of normalized text patterns that are boilerplate
        - Dict mapping block text to boilerplate type ('header', 'footer', 'page_number')
    """
    if config is None:
        config = DEFAULT_CHUNKING_CONFIG

    if not config.filter_headers_footers:
        return set(), {}

    # Count pages
    pages = set(b.get("page_no", 1) for b in blocks)
    total_pages = len(pages)

    if total_pages < 3:
        # Not enough pages to detect patterns
        return set(), {}

    # Group blocks by normalized text
    text_occurrences: Dict[str, List[Dict]] = defaultdict(list)

    for block in blocks:
        text = block.get("text", "").strip()
        if not text or len(text) > 200:  # Skip empty or very long text
            continue

        normalized = _normalize_text(text)
        text_occurrences[normalized].append(block)

    # Identify boilerplate
    boilerplate_patterns: Set[str] = set()
    boilerplate_map: Dict[str, str] = {}

    threshold = config.header_footer_threshold

    for normalized, occurrences in text_occurrences.items():
        # Check if appears on enough pages
        occurrence_pages = set(b.get("page_no", 1) for b in occurrences)
        occurrence_ratio = len(occurrence_pages) / total_pages

        if occurrence_ratio < threshold:
            continue

        # Determine type based on position and content
        positions = [_get_vertical_position(b) for b in occurrences]
        position_counts = {
            "header": positions.count("header"),
            "footer": positions.count("footer"),
            "body": positions.count("body"),
        }

        # Check for page numbers
        sample_text = occurrences[0].get("text", "")
        if _is_page_number(sample_text):
            boilerplate_patterns.add(normalized)
            boilerplate_map[normalized] = "page_number"
            continue

        # Check position consistency
        if position_counts["header"] >= len(occurrences) * 0.7:
            boilerplate_patterns.add(normalized)
            boilerplate_map[normalized] = "header"
        elif position_counts["footer"] >= len(occurrences) * 0.7:
            boilerplate_patterns.add(normalized)
            boilerplate_map[normalized] = "footer"
        elif _is_common_footer(sample_text):
            boilerplate_patterns.add(normalized)
            boilerplate_map[normalized] = "footer"

    return boilerplate_patterns, boilerplate_map


def filter_boilerplate(
    blocks: List[Dict],
    config: ChunkingConfig = None,
    tag_only: bool = False,
) -> List[Dict]:
    """
    Filter or tag boilerplate blocks.

    Args:
        blocks: List of text blocks
        config: Chunking configuration
        tag_only: If True, tag blocks instead of removing them

    Returns:
        List of blocks with boilerplate filtered/tagged
    """
    if config is None:
        config = DEFAULT_CHUNKING_CONFIG

    if not config.filter_headers_footers:
        return blocks

    boilerplate_patterns, boilerplate_map = detect_boilerplate(blocks, config)

    if not boilerplate_patterns:
        return blocks

    filtered = []

    for block in blocks:
        text = block.get("text", "").strip()
        normalized = _normalize_text(text)

        if normalized in boilerplate_patterns:
            if tag_only:
                # Tag the block instead of removing
                block = block.copy()
                block["is_boilerplate"] = True
                block["boilerplate_type"] = boilerplate_map.get(normalized, "unknown")
                filtered.append(block)
            # else: skip the block (don't add to filtered)
        else:
            filtered.append(block)

    return filtered


def extract_page_numbers(blocks: List[Dict]) -> Dict[int, int]:
    """
    Extract page number mapping from blocks.

    Returns:
        Dict mapping physical page_no to logical page number
    """
    page_numbers = {}

    for block in blocks:
        text = block.get("text", "").strip()
        page_no = block.get("page_no", 1)

        if _is_page_number(text):
            # Extract the number
            match = re.search(r'\d+', text)
            if match:
                logical_page = int(match.group())
                page_numbers[page_no] = logical_page

    return page_numbers
