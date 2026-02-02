def merge_elements(text_tables, headings, layout_blocks=None):
    """
    Merge text/table elements with headings, sorted by reading order.

    Uses bbox coordinates for within-page ordering:
    - Primary: page_no
    - Secondary: y position (top to bottom)
    - Tertiary: x position (left to right)

    Args:
        text_tables: List of text/table elements from pdfplumber
        headings: List of heading elements from heading inference
        layout_blocks: Optional layout blocks with bbox for position lookup
    """
    elements = text_tables + headings

    # Build position lookup from layout blocks if available
    position_map = {}
    if layout_blocks:
        for block in layout_blocks:
            bbox = block.get("bbox", [])
            if bbox and len(bbox) >= 4:
                # Key by (page_no, text_prefix) for matching
                key = (block.get("page_no"), block.get("text", "")[:50])
                position_map[key] = {
                    "y": bbox[1],
                    "x": bbox[0],
                    "bbox": bbox,
                }

    def sort_key(el):
        page_no = el.get("page_no", 0)

        # Try to get position from element directly
        bbox = el.get("bbox", [])
        if bbox and len(bbox) >= 4:
            return (page_no, round(bbox[1], 1), round(bbox[0], 1))

        # Try to look up from position map
        key = (page_no, el.get("text", "")[:50])
        if key in position_map:
            pos = position_map[key]
            return (page_no, round(pos["y"], 1), round(pos["x"], 1))

        # Fallback to page order only
        return (page_no, float("inf"), float("inf"))

    elements.sort(key=sort_key)
    return elements
