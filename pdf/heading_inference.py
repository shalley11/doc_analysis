"""
Enhanced heading detection with multi-signal approach.

Signals used:
1. Font size (larger = heading)
2. Bold weight (flags & 16)
3. ALL CAPS text
4. Numbered patterns (1., 1.1, Chapter X)
5. Short line length (< heading_max_words)
6. Position (first line after page break)
"""

import re
from collections import Counter
from typing import List, Dict, Tuple, Optional

from doc_analysis.config import ChunkingConfig, DEFAULT_CHUNKING_CONFIG


# Heading patterns
NUMBERED_HEADING = re.compile(
    r'^(\d+\.)+\s*\w|'           # 1.2.3 Style
    r'^Chapter\s+\d+|'           # Chapter 1
    r'^Section\s+\d+|'           # Section 1
    r'^Part\s+[IVXLCDM]+|'       # Part III (Roman numerals)
    r'^Appendix\s+[A-Z]|'        # Appendix A
    r'^Article\s+\d+|'           # Article 1
    r'^\d+\s+[A-Z]',             # 1 Introduction
    re.IGNORECASE
)

# Patterns that indicate NOT a heading
NON_HEADING_PATTERNS = re.compile(
    r'\.$|'                       # Ends with period (sentence)
    r'^\s*[\(\[]|'               # Starts with parenthesis/bracket
    r'\b(the|a|an|is|are|was|were|be|been|being)\b.*\b(the|a|an|is|are)\b',  # Contains multiple articles (sentence-like)
    re.IGNORECASE
)


def _is_all_caps(text: str) -> bool:
    """Check if text is ALL CAPS (excluding numbers/punctuation)."""
    letters = [c for c in text if c.isalpha()]
    if len(letters) < 3:
        return False
    return all(c.isupper() for c in letters)


def _is_bold(flags: int) -> bool:
    """Check if font flags indicate bold."""
    return bool(flags & 16)


def _is_italic(flags: int) -> bool:
    """Check if font flags indicate italic."""
    return bool(flags & 2)


def _word_count(text: str) -> int:
    """Count words in text."""
    return len(text.split())


def _has_numbered_pattern(text: str) -> bool:
    """Check if text matches numbered heading pattern."""
    return bool(NUMBERED_HEADING.match(text.strip()))


def _is_sentence_like(text: str) -> bool:
    """Check if text looks like a sentence rather than a heading."""
    text = text.strip()

    # Too many words for a heading
    if _word_count(text) > 20:
        return True

    # Ends with period and has multiple words
    if text.endswith('.') and _word_count(text) > 3:
        return True

    # Matches non-heading patterns
    if NON_HEADING_PATTERNS.search(text):
        return True

    return False


def _calculate_heading_score(
    block: Dict,
    font_stats: Dict,
    config: ChunkingConfig,
) -> Tuple[float, int]:
    """
    Calculate heading likelihood score and suggested level.

    Returns:
        (score, heading_level) where score is 0-1 and level is 1-6
    """
    text = block.get("text", "").strip()
    font_info = block.get("font_info", {})
    spans = block.get("spans", [])

    if not text:
        return 0.0, 0

    # Quick rejection for sentence-like text
    if _is_sentence_like(text):
        return 0.0, 0

    score = 0.0
    level_hints = []

    # --- Signal 1: Font Size ---
    font_size = font_info.get("size", 12)
    size_percentile = font_stats.get("percentiles", {})

    if font_size >= size_percentile.get("p95", 20):
        score += 0.35
        level_hints.append(1)
    elif font_size >= size_percentile.get("p90", 16):
        score += 0.30
        level_hints.append(2)
    elif font_size >= size_percentile.get("p80", 14):
        score += 0.20
        level_hints.append(3)
    elif font_size >= size_percentile.get("p70", 13):
        score += 0.10
        level_hints.append(4)

    # --- Signal 2: Bold Weight ---
    is_bold_text = font_info.get("bold", False)

    # Also check spans for bold
    if not is_bold_text and spans:
        bold_chars = 0
        total_chars = 0
        for line in spans:
            for span in line.get("spans", []):
                char_count = len(span.get("text", ""))
                total_chars += char_count
                if span.get("flags", 0) & 16:
                    bold_chars += char_count

        if total_chars > 0 and bold_chars / total_chars > 0.5:
            is_bold_text = True

    if is_bold_text and config.detect_bold_headings:
        score += 0.20

    # --- Signal 3: ALL CAPS ---
    if _is_all_caps(text) and config.detect_caps_headings:
        score += 0.15

    # --- Signal 4: Numbered Pattern ---
    if _has_numbered_pattern(text) and config.detect_numbered_headings:
        score += 0.25

        # Determine level from numbering depth
        match = re.match(r'^((\d+\.)+)', text)
        if match:
            depth = match.group(1).count('.')
            level_hints.append(min(depth, 6))
        elif re.match(r'^Chapter\s+\d+', text, re.IGNORECASE):
            level_hints.append(1)
        elif re.match(r'^Section\s+\d+', text, re.IGNORECASE):
            level_hints.append(2)

    # --- Signal 5: Short Line Length ---
    wc = _word_count(text)
    if wc <= config.heading_max_words:
        score += 0.10
    if wc <= 5:
        score += 0.05

    # --- Determine Level ---
    if level_hints:
        heading_level = min(level_hints)
    else:
        # Infer from font size ranking
        if font_size >= size_percentile.get("p95", 20):
            heading_level = 1
        elif font_size >= size_percentile.get("p90", 16):
            heading_level = 2
        elif font_size >= size_percentile.get("p80", 14):
            heading_level = 3
        elif font_size >= size_percentile.get("p70", 13):
            heading_level = 4
        else:
            heading_level = 5

    # Clamp to configured max levels
    heading_level = min(heading_level, config.heading_levels)

    return score, heading_level


def _compute_font_stats(blocks: List[Dict]) -> Dict:
    """
    Compute font size statistics across all blocks.

    Returns percentile thresholds for heading detection.
    """
    sizes = []

    for b in blocks:
        # From font_info
        font_info = b.get("font_info", {})
        if font_info.get("size"):
            sizes.append(font_info["size"])

        # From spans
        for line in b.get("spans", []):
            for span in line.get("spans", []):
                if span.get("size"):
                    sizes.append(span["size"])

    if not sizes:
        return {"percentiles": {}}

    sizes = sorted(sizes)
    n = len(sizes)

    def percentile(p):
        idx = int(n * p / 100)
        return sizes[min(idx, n - 1)]

    return {
        "percentiles": {
            "p50": percentile(50),
            "p70": percentile(70),
            "p80": percentile(80),
            "p90": percentile(90),
            "p95": percentile(95),
        },
        "min": sizes[0],
        "max": sizes[-1],
        "common": Counter(round(s, 1) for s in sizes).most_common(5),
    }


def infer_headings(
    blocks: List[Dict],
    config: ChunkingConfig = None,
    heading_threshold: float = 0.4,
) -> List[Dict]:
    """
    Infer headings from blocks using multi-signal approach.

    Args:
        blocks: List of text blocks with font info and spans
        config: Chunking configuration
        heading_threshold: Minimum score to classify as heading (0-1)

    Returns:
        List of heading elements with heading_level
    """
    if config is None:
        config = DEFAULT_CHUNKING_CONFIG

    # Compute font statistics
    font_stats = _compute_font_stats(blocks)

    headings = []

    for b in blocks:
        if b.get("type") != "paragraph":
            continue

        text = b.get("text", "").strip()
        if not text:
            continue

        # Calculate heading score
        score, level = _calculate_heading_score(b, font_stats, config)

        if score >= heading_threshold:
            headings.append({
                "type": "heading",
                "text": text,
                "page_no": b.get("page_no", 1),
                "heading_level": level,
                "heading_score": round(score, 2),
                "pdf_id": b.get("pdf_id", ""),
                "pdf_name": b.get("pdf_name", ""),
                "bbox": b.get("bbox", []),
                "font_info": b.get("font_info", {}),
            })

    return headings


def infer_headings_legacy(blocks: List[Dict]) -> List[Dict]:
    """
    Legacy heading inference using only font size.
    Kept for backwards compatibility.
    """
    sizes = []

    for b in blocks:
        for line in b.get("spans", []):
            for span in line.get("spans", []):
                sizes.append(round(span.get("size", 12), 1))

    common = sorted(
        [s for s, _ in Counter(sizes).most_common(4)],
        reverse=True
    )

    headings = []

    for b in blocks:
        if "spans" not in b:
            continue

        try:
            max_size = max(
                span.get("size", 12)
                for line in b["spans"]
                for span in line.get("spans", [])
            )
        except ValueError:
            continue

        if round(max_size, 1) in common[:2]:
            headings.append({
                "type": "heading",
                "text": b.get("text", ""),
                "page_no": b.get("page_no", 1),
                "heading_level": 1 if round(max_size, 1) == common[0] else 2,
                "pdf_id": b.get("pdf_id", ""),
                "pdf_name": b.get("pdf_name", ""),
            })

    return headings
