# structure_detector.py
"""
Structure detection module for PDF documents.
Analyzes font sizes and styles to detect document hierarchy (headings, sections).
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from collections import Counter
import logging

logger = logging.getLogger(__name__)


@dataclass
class FontProfile:
    """Profile of fonts used in a document."""
    body_size: float                           # Most common font size (body text)
    heading_sizes: List[float] = field(default_factory=list)  # Sizes larger than body (sorted desc)
    size_to_level: Dict[float, int] = field(default_factory=dict)  # Map size -> heading level (1-3)
    bold_at_body_is_h3: bool = True           # Whether bold at body size is H3


@dataclass
class SectionNode:
    """A node in the section hierarchy tree."""
    title: str
    level: int  # 1=H1, 2=H2, 3=H3, 0=body
    block_index: int
    children: List['SectionNode'] = field(default_factory=list)
    parent: Optional['SectionNode'] = None


def analyze_document_fonts(doc) -> FontProfile:
    """
    First pass: collect font statistics across all pages.

    Analyzes font sizes to determine:
    - Body text size (most frequent)
    - Heading sizes (larger than body)
    - Size-to-level mapping (H1, H2, H3)

    Args:
        doc: PyMuPDF document object

    Returns:
        FontProfile with font analysis results
    """
    font_sizes = Counter()
    bold_sizes = Counter()

    # Collect font statistics from all pages
    for page in doc:
        text_dict = page.get_text("dict")

        for block in text_dict.get("blocks", []):
            if block.get("type") != 0:  # Skip non-text blocks
                continue

            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    size = round(span.get("size", 0), 1)
                    flags = span.get("flags", 0)
                    is_bold = bool(flags & 2**4)  # Bold flag
                    text = span.get("text", "").strip()

                    # Only count non-empty spans with reasonable sizes
                    if text and 6 <= size <= 72:
                        font_sizes[size] += len(text)
                        if is_bold:
                            bold_sizes[size] += len(text)

    if not font_sizes:
        # Default profile if no fonts found
        return FontProfile(body_size=12.0)

    # Body size is the most common font size
    body_size = font_sizes.most_common(1)[0][0]

    # Find heading sizes (larger than body)
    heading_sizes = sorted(
        [size for size in font_sizes.keys() if size > body_size],
        reverse=True
    )

    # Create size-to-level mapping
    size_to_level = {}

    # Calculate thresholds relative to body size
    h1_threshold = body_size * 1.5  # H1 >= 1.5x body
    h2_threshold = body_size * 1.2  # H2 >= 1.2x body
    h3_threshold = body_size * 1.1  # H3 >= 1.1x body

    for size in heading_sizes:
        if size >= h1_threshold:
            size_to_level[size] = 1  # H1
        elif size >= h2_threshold:
            size_to_level[size] = 2  # H2
        elif size >= h3_threshold:
            size_to_level[size] = 3  # H3

    # Check if bold at body size should be H3
    # (only if body size bold text is relatively common but not majority)
    bold_body_ratio = bold_sizes.get(body_size, 0) / font_sizes.get(body_size, 1)
    bold_at_body_is_h3 = 0.01 < bold_body_ratio < 0.3

    logger.debug(f"Font analysis: body={body_size}, headings={heading_sizes[:5]}")
    logger.debug(f"Size-to-level mapping: {size_to_level}")

    return FontProfile(
        body_size=body_size,
        heading_sizes=heading_sizes[:10],  # Keep top 10 heading sizes
        size_to_level=size_to_level,
        bold_at_body_is_h3=bold_at_body_is_h3
    )


def classify_heading_level(
    font_size: float,
    is_bold: bool,
    font_profile: FontProfile
) -> Optional[int]:
    """
    Classify text as H1/H2/H3 or body based on font characteristics.

    Args:
        font_size: The font size of the text
        is_bold: Whether the text is bold
        font_profile: Document font profile

    Returns:
        Heading level (1, 2, 3) or None for body text
    """
    # Check if size is in the heading mapping
    if font_size in font_profile.size_to_level:
        return font_profile.size_to_level[font_size]

    # Check thresholds directly
    body_size = font_profile.body_size

    if font_size >= body_size * 1.5:
        return 1  # H1
    elif font_size >= body_size * 1.2:
        return 2  # H2
    elif font_size >= body_size * 1.1:
        return 3  # H3
    elif is_bold and abs(font_size - body_size) < 0.5 and font_profile.bold_at_body_is_h3:
        return 3  # Bold body text as H3

    return None  # Body text


def extract_block_heading_info(block: Dict[str, Any], font_profile: FontProfile) -> Dict[str, Any]:
    """
    Extract heading information from a text block.

    Analyzes the first line/span to determine if the block is a heading.

    Args:
        block: Text block from page.get_text('dict')
        font_profile: Document font profile

    Returns:
        Dict with heading_level, is_heading, and dominant_size
    """
    if block.get("type") != 0:  # Not a text block
        return {"heading_level": None, "is_heading": False, "dominant_size": None}

    lines = block.get("lines", [])
    if not lines:
        return {"heading_level": None, "is_heading": False, "dominant_size": None}

    # Analyze first line to determine heading status
    first_line = lines[0]
    spans = first_line.get("spans", [])

    if not spans:
        return {"heading_level": None, "is_heading": False, "dominant_size": None}

    # Get dominant font characteristics from first line
    total_chars = 0
    weighted_size = 0
    is_mostly_bold = False
    bold_chars = 0

    for span in spans:
        text = span.get("text", "")
        size = span.get("size", 0)
        flags = span.get("flags", 0)
        is_bold = bool(flags & 2**4)

        char_count = len(text)
        total_chars += char_count
        weighted_size += size * char_count
        if is_bold:
            bold_chars += char_count

    if total_chars == 0:
        return {"heading_level": None, "is_heading": False, "dominant_size": None}

    dominant_size = round(weighted_size / total_chars, 1)
    is_mostly_bold = bold_chars > total_chars * 0.5

    heading_level = classify_heading_level(dominant_size, is_mostly_bold, font_profile)

    return {
        "heading_level": heading_level,
        "is_heading": heading_level is not None,
        "dominant_size": dominant_size,
        "is_bold": is_mostly_bold
    }


def build_section_hierarchy(
    blocks: List[Dict[str, Any]],
    font_profile: FontProfile
) -> Dict[int, List[str]]:
    """
    Build hierarchy and map each block index to its section path.

    Traverses blocks to build a tree of sections, then generates
    a path (e.g., ["Chapter 1", "Introduction", "Background"]) for each block.

    Args:
        blocks: List of text blocks with heading info
        font_profile: Document font profile

    Returns:
        Dict mapping block_index -> list of section titles (the path)
    """
    # Current section at each level
    current_sections = {
        1: None,  # H1
        2: None,  # H2
        3: None,  # H3
    }

    block_to_path = {}

    for i, block in enumerate(blocks):
        # Get heading info for this block
        heading_info = extract_block_heading_info(block, font_profile)

        if heading_info.get("is_heading"):
            level = heading_info["heading_level"]

            # Extract heading text
            heading_text = ""
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    heading_text += span.get("text", "")
            heading_text = heading_text.strip()

            # Update current section at this level
            current_sections[level] = heading_text

            # Clear lower levels when a higher level heading is encountered
            for lower_level in range(level + 1, 4):
                current_sections[lower_level] = None

        # Build path for this block
        path = []
        for level in [1, 2, 3]:
            if current_sections[level]:
                path.append(current_sections[level])

        block_to_path[i] = path

    return block_to_path


def get_section_path_string(section_path: List[str]) -> str:
    """
    Convert section path to a string for display.

    Args:
        section_path: List of section titles

    Returns:
        String like "[Section: Chapter 1 > Introduction > Background]"
    """
    if not section_path:
        return ""

    path_str = " > ".join(section_path)
    return f"[Section: {path_str}]"


def detect_document_structure(doc) -> Tuple[FontProfile, Dict[int, Dict[int, List[str]]]]:
    """
    Detect document structure across all pages.

    Performs font analysis and builds section hierarchy for entire document.

    Args:
        doc: PyMuPDF document object

    Returns:
        Tuple of (FontProfile, Dict[page_no -> Dict[block_idx -> section_path]])
    """
    # Analyze fonts across document
    font_profile = analyze_document_fonts(doc)

    # Build section hierarchy for each page
    # Track current section state across pages
    current_sections = {1: None, 2: None, 3: None}
    page_block_paths = {}

    for page_no, page in enumerate(doc):
        text_dict = page.get_text("dict")
        blocks = text_dict.get("blocks", [])

        block_paths = {}

        for block_idx, block in enumerate(blocks):
            if block.get("type") != 0:  # Skip non-text blocks
                # Non-text blocks get current section path
                path = [s for s in [current_sections[1], current_sections[2], current_sections[3]] if s]
                block_paths[block_idx] = path
                continue

            heading_info = extract_block_heading_info(block, font_profile)

            if heading_info.get("is_heading"):
                level = heading_info["heading_level"]

                # Extract heading text
                heading_text = ""
                for line in block.get("lines", []):
                    for span in line.get("spans", []):
                        heading_text += span.get("text", "")
                heading_text = heading_text.strip()[:200]  # Limit length

                # Update current section
                current_sections[level] = heading_text

                # Clear lower levels
                for lower_level in range(level + 1, 4):
                    current_sections[lower_level] = None

            # Build path for this block
            path = [s for s in [current_sections[1], current_sections[2], current_sections[3]] if s]
            block_paths[block_idx] = path

        page_block_paths[page_no] = block_paths

    return font_profile, page_block_paths


def annotate_blocks_with_structure(
    blocks: List[Dict[str, Any]],
    font_profile: FontProfile,
    page_block_paths: Optional[Dict[int, List[str]]] = None
) -> List[Dict[str, Any]]:
    """
    Annotate blocks with section hierarchy and heading level.

    Args:
        blocks: List of content blocks
        font_profile: Document font profile
        page_block_paths: Pre-computed block paths (optional)

    Returns:
        Blocks with added section_hierarchy and heading_level fields
    """
    if page_block_paths is None:
        # Build paths for just these blocks
        raw_blocks = []  # Convert to dict format if needed
        for b in blocks:
            if isinstance(b, dict):
                raw_blocks.append(b)
            else:
                raw_blocks.append({"lines": [], "type": 0})

        page_block_paths = build_section_hierarchy(raw_blocks, font_profile)

    annotated_blocks = []

    for i, block in enumerate(blocks):
        block_copy = block.copy() if isinstance(block, dict) else {"content": str(block)}

        # Get section path
        section_path = page_block_paths.get(i, [])

        # Get heading level
        heading_info = extract_block_heading_info(block, font_profile) if isinstance(block, dict) else {}
        heading_level = heading_info.get("heading_level", 0) or 0

        block_copy["section_hierarchy"] = section_path
        block_copy["heading_level"] = heading_level

        annotated_blocks.append(block_copy)

    return annotated_blocks
