import pdfplumber
import fitz
import os
import uuid
from typing import List, Dict, Optional

from doc_analysis.config import IMAGE_STORE_DIR


def _save_table_image(pdf_path: str, page_no: int, bbox: List[float], pdf_id: str, table_id: int) -> Optional[str]:
    """
    Save table region as image using PyMuPDF.

    Args:
        pdf_path: Path to PDF file
        page_no: Page number (1-indexed)
        bbox: Bounding box [x0, y0, x1, y1]
        pdf_id: PDF identifier
        table_id: Table identifier

    Returns:
        Path to saved image or None if failed
    """
    try:
        os.makedirs(IMAGE_STORE_DIR, exist_ok=True)
        doc = fitz.open(pdf_path)
        page = doc[page_no - 1]  # 0-indexed

        # Convert bbox to fitz.Rect
        rect = fitz.Rect(bbox)

        # Add padding around table
        rect.x0 = max(0, rect.x0 - 5)
        rect.y0 = max(0, rect.y0 - 5)
        rect.x1 = min(page.rect.width, rect.x1 + 5)
        rect.y1 = min(page.rect.height, rect.y1 + 5)

        # Extract pixmap at higher resolution
        mat = fitz.Matrix(2, 2)  # 2x zoom for better quality
        pix = page.get_pixmap(matrix=mat, clip=rect)

        image_filename = f"table_{pdf_id}_{table_id}_{uuid.uuid4().hex[:8]}.png"
        image_path = os.path.join(IMAGE_STORE_DIR, image_filename)
        pix.save(image_path)

        doc.close()
        return image_path
    except Exception as e:
        print(f"Failed to save table image: {e}")
        return None


def extract_text_and_tables(
    pdf_path: str,
    pdf_id: str,
    pdf_name: str,
    include_bbox: bool = True,
    start_page: Optional[int] = None,
    end_page: Optional[int] = None,
) -> List[Dict]:
    """
    Extract text lines and tables from PDF using pdfplumber.

    Args:
        pdf_path: Path to PDF file
        pdf_id: Unique PDF identifier
        pdf_name: Original PDF filename
        include_bbox: Whether to include bounding box coordinates
        start_page: Start page (1-indexed, inclusive). None means page 1.
        end_page: End page (1-indexed, inclusive). None means last page.

    Returns:
        List of elements with type, text, page_no, and optional bbox
    """
    elements = []
    table_counter = 0  # Track table ID across entire PDF

    with pdfplumber.open(pdf_path) as pdf:
        total_pages = len(pdf.pages)

        # Determine page range
        first_page = start_page if start_page else 1
        last_page = end_page if end_page else total_pages

        # Clamp to valid range
        first_page = max(1, min(first_page, total_pages))
        last_page = max(first_page, min(last_page, total_pages))

        for page_no, page in enumerate(pdf.pages, 1):
            # Skip pages outside the range
            if page_no < first_page or page_no > last_page:
                continue
            page_height = page.height
            page_width = page.width

            # Extract words with positions for bbox calculation
            words = page.extract_words() or []

            # Group words into lines by approximate y-position
            lines = _group_words_into_lines(words)

            for line_words in lines:
                if not line_words:
                    continue

                text = " ".join(w.get("text", "") for w in line_words)
                text = text.strip()

                if not text:
                    continue

                element = {
                    "type": "paragraph",
                    "text": text,
                    "page_no": page_no,
                    "pdf_id": pdf_id,
                    "pdf_name": pdf_name,
                    "page_height": page_height,
                    "page_width": page_width,
                }

                if include_bbox and line_words:
                    bbox = _compute_line_bbox(line_words)
                    if bbox:
                        element["bbox"] = bbox

                elements.append(element)

            # Extract tables
            tables = page.extract_tables() or []
            for table_idx, table in enumerate(tables):
                if not table:
                    continue

                # Filter empty rows and cells
                table_rows = []
                for row in table:
                    if row:
                        cleaned_row = [
                            str(cell).strip() if cell else ""
                            for cell in row
                        ]
                        if any(cleaned_row):
                            table_rows.append(cleaned_row)

                if not table_rows:
                    continue

                table_counter += 1  # Increment for each valid table

                table_text = "\n".join(
                    " | ".join(row) for row in table_rows
                )

                element = {
                    "type": "table",
                    "text": table_text,
                    "page_no": page_no,
                    "pdf_id": pdf_id,
                    "pdf_name": pdf_name,
                    "page_height": page_height,
                    "page_width": page_width,
                    "table_rows": len(table_rows),
                    "table_cols": len(table_rows[0]) if table_rows else 0,
                    "table_id": table_counter,  # Unique table ID within PDF
                }

                # Try to get table bbox from find_tables
                table_bbox = None
                try:
                    table_objects = page.find_tables()
                    if table_idx < len(table_objects):
                        table_obj = table_objects[table_idx]
                        table_bbox = list(table_obj.bbox)
                        element["bbox"] = table_bbox
                except Exception:
                    pass

                # Save table as image for vision model processing
                if table_bbox:
                    table_image_path = _save_table_image(
                        pdf_path, page_no, table_bbox, pdf_id, table_counter
                    )
                    if table_image_path:
                        element["table_image_path"] = table_image_path

                elements.append(element)

    return elements


def _group_words_into_lines(
    words: List[Dict],
    y_tolerance: float = 5.0,
) -> List[List[Dict]]:
    """
    Group words into lines based on y-position proximity.

    Args:
        words: List of word dictionaries from pdfplumber
        y_tolerance: Maximum y-distance to consider same line

    Returns:
        List of lines, where each line is a list of word dicts
    """
    if not words:
        return []

    # Sort by top position, then left position
    sorted_words = sorted(words, key=lambda w: (w.get("top", 0), w.get("x0", 0)))

    lines = []
    current_line = []
    current_y = None

    for word in sorted_words:
        word_y = word.get("top", 0)

        if current_y is None:
            current_y = word_y
            current_line.append(word)
        elif abs(word_y - current_y) <= y_tolerance:
            current_line.append(word)
        else:
            # New line
            if current_line:
                # Sort by x position within line
                current_line.sort(key=lambda w: w.get("x0", 0))
                lines.append(current_line)
            current_line = [word]
            current_y = word_y

    # Add last line
    if current_line:
        current_line.sort(key=lambda w: w.get("x0", 0))
        lines.append(current_line)

    return lines


def _compute_line_bbox(words: List[Dict]) -> Optional[List[float]]:
    """
    Compute bounding box for a line of words.

    Returns:
        [x0, y0, x1, y1] or None if cannot compute
    """
    if not words:
        return None

    try:
        x0 = min(w.get("x0", 0) for w in words)
        y0 = min(w.get("top", 0) for w in words)
        x1 = max(w.get("x1", 0) for w in words)
        y1 = max(w.get("bottom", 0) for w in words)
        return [x0, y0, x1, y1]
    except (ValueError, TypeError):
        return None
