# pdf_utils.py
from pathlib import Path
from typing import List, Dict, Any, Optional
import fitz  # PyMuPDF

from pdf.structure_detector import FontProfile, extract_block_heading_info, get_section_path_string


def save_full_page_image(page, page_no: int, out: str) -> str:
    """Save full page as image (for scanned pages)."""
    path = Path(out) / "images" / f"page_{page_no+1}_full.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    pix = page.get_pixmap(dpi=300)
    pix.save(path)
    return str(path)


def extract_image(page, img_info, page_no: int, img_index: int, out: str) -> Dict[str, Any]:
    """Extract a single image from page and save it."""
    xref = img_info[0]
    base = page.parent.extract_image(xref)
    path = Path(out) / "images" / f"page_{page_no+1}_img_{img_index+1}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write(base["image"])

    # Get image bounding box for position tracking
    img_rects = page.get_image_rects(xref)
    bbox = img_rects[0] if img_rects else None

    return {
        "image_path": str(path),
        "bbox": list(bbox) if bbox else None,
        "y_position": bbox.y0 if bbox else 0
    }


def extract_table_as_image(page, table, page_no: int, table_index: int, out: str) -> Dict[str, Any]:
    """Extract table and save as image for vision model processing."""
    # Save table as image
    bbox = table.bbox
    pix = page.get_pixmap(clip=bbox, dpi=300)
    path = Path(out) / "tables" / f"page_{page_no+1}_table_{table_index+1}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    pix.save(path)

    # bbox is a tuple (x0, y0, x1, y1)
    y_position = bbox[1] if isinstance(bbox, tuple) else bbox.y0

    return {
        "table_image_path": str(path),
        "table_markdown": table.to_markdown(),
        "row_count": table.row_count,
        "col_count": table.col_count,
        "bbox": list(bbox) if isinstance(bbox, tuple) else [bbox.x0, bbox.y0, bbox.x1, bbox.y1],
        "y_position": y_position
    }


def get_text_blocks(page) -> List[Dict[str, Any]]:
    """Extract text blocks with their positions."""
    blocks = []
    text_dict = page.get_text("dict")

    for block in text_dict.get("blocks", []):
        if block.get("type") == 0:  # Text block
            text_content = ""
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    text_content += span.get("text", "")
                text_content += "\n"

            text_content = text_content.strip()
            if text_content:
                blocks.append({
                    "type": "text",
                    "content": text_content,
                    "bbox": block.get("bbox"),
                    "y_position": block.get("bbox", [0, 0, 0, 0])[1]
                })

    return blocks


def process_page_with_positions(page, page_no: int, pdf_name: str, out: str) -> Dict[str, Any]:
    """
    Process page and extract all content with position tracking.
    Returns content blocks in reading order (top to bottom).
    """
    result = {
        "metadata": {
            "pdf_name": pdf_name,
            "page_no": page_no + 1,
            "page_count": page.parent.page_count
        },
        "scanned": False,
        "blocks": []  # Ordered list of content blocks
    }

    # Check if page is scanned (no extractable text)
    text = page.get_text("text").strip()
    if not text:
        result["scanned"] = True
        full_page_path = save_full_page_image(page, page_no, out)
        result["blocks"].append({
            "type": "image",
            "content": None,
            "image_link": full_page_path,
            "table_link": None,
            "y_position": 0,
            "metadata": {"is_full_page": True}
        })
        return result

    # Extract all content types
    all_elements = []

    # 1. Extract text blocks
    text_blocks = get_text_blocks(page)
    for tb in text_blocks:
        all_elements.append({
            "type": "text",
            "content": tb["content"],
            "image_link": None,
            "table_link": None,
            "y_position": tb["y_position"],
            "metadata": {}
        })

    # 2. Extract tables (all saved as images)
    tables = page.find_tables()
    for i, table in enumerate(tables):
        table_data = extract_table_as_image(page, table, page_no, i, out)
        all_elements.append({
            "type": "table",
            "content": table_data["table_markdown"],  # Keep markdown for fallback
            "image_link": None,
            "table_link": table_data["table_image_path"],
            "y_position": table_data["y_position"],
            "metadata": {
                "row_count": table_data["row_count"],
                "col_count": table_data["col_count"]
            }
        })

    # 3. Extract images
    images = page.get_images(full=True)
    for i, img in enumerate(images):
        img_data = extract_image(page, img, page_no, i, out)
        all_elements.append({
            "type": "image",
            "content": None,  # Will be filled by vision model
            "image_link": img_data["image_path"],
            "table_link": None,
            "y_position": img_data["y_position"],
            "metadata": {}
        })

    # Sort by y_position to maintain reading order
    all_elements.sort(key=lambda x: x["y_position"])

    # Assign positions
    for i, element in enumerate(all_elements):
        element["position"] = i
        del element["y_position"]  # Remove temporary field

    result["blocks"] = all_elements
    return result


def get_text_blocks_with_fonts(page) -> List[Dict[str, Any]]:
    """
    Extract text blocks with font metadata from page.

    Returns list of blocks with:
    - content: text content
    - bbox: bounding box
    - y_position: vertical position
    - font_size: dominant font size
    - is_bold: whether text is predominantly bold
    - lines: raw line/span data for detailed analysis

    Args:
        page: PyMuPDF page object

    Returns:
        List of block dicts with font metadata
    """
    blocks = []
    text_dict = page.get_text("dict")

    for block in text_dict.get("blocks", []):
        if block.get("type") != 0:  # Skip non-text blocks
            continue

        # Extract text and font info
        text_content = ""
        total_chars = 0
        weighted_size = 0
        bold_chars = 0
        lines_data = []

        for line in block.get("lines", []):
            line_text = ""
            line_spans = []

            for span in line.get("spans", []):
                span_text = span.get("text", "")
                line_text += span_text
                text_content += span_text

                char_count = len(span_text)
                size = span.get("size", 12)
                flags = span.get("flags", 0)
                is_bold = bool(flags & 2**4)

                total_chars += char_count
                weighted_size += size * char_count
                if is_bold:
                    bold_chars += char_count

                line_spans.append({
                    "text": span_text,
                    "size": size,
                    "is_bold": is_bold,
                    "flags": flags,
                    "font": span.get("font", "")
                })

            text_content += "\n"
            lines_data.append({
                "text": line_text,
                "spans": line_spans
            })

        text_content = text_content.strip()
        if not text_content:
            continue

        # Calculate dominant font characteristics
        font_size = round(weighted_size / total_chars, 1) if total_chars > 0 else 12.0
        is_bold = bold_chars > total_chars * 0.5

        blocks.append({
            "type": "text",
            "content": text_content,
            "bbox": block.get("bbox"),
            "y_position": block.get("bbox", [0, 0, 0, 0])[1],
            "font_size": font_size,
            "is_bold": is_bold,
            "lines": lines_data,
            "raw_block": block  # Keep raw block for structure detection
        })

    return blocks


def process_page_with_structure(
    page,
    page_no: int,
    pdf_name: str,
    out: str,
    font_profile: Optional[FontProfile] = None,
    page_block_paths: Optional[Dict[int, List[str]]] = None
) -> Dict[str, Any]:
    """
    Process page with heading detection using font_profile.

    Extracts all content with position tracking and adds
    section hierarchy information to each block.

    Args:
        page: PyMuPDF page object
        page_no: Page number (0-indexed)
        pdf_name: Name of the PDF file
        out: Output directory for extracted images/tables
        font_profile: Document font profile (optional, for heading detection)
        page_block_paths: Pre-computed section paths for blocks on this page

    Returns:
        Page result dict with blocks annotated with section_hierarchy
    """
    result = {
        "metadata": {
            "pdf_name": pdf_name,
            "page_no": page_no + 1,
            "page_count": page.parent.page_count
        },
        "scanned": False,
        "blocks": []
    }

    # Check if page is scanned
    text = page.get_text("text").strip()
    if not text:
        result["scanned"] = True
        full_page_path = save_full_page_image(page, page_no, out)
        result["blocks"].append({
            "type": "image",
            "content": None,
            "image_link": full_page_path,
            "table_link": None,
            "y_position": 0,
            "section_hierarchy": [],
            "heading_level": 0,
            "metadata": {"is_full_page": True}
        })
        return result

    # Extract all content types
    all_elements = []

    # Track current section for non-text elements
    current_section_path = []

    # 1. Extract text blocks with font info
    text_blocks = get_text_blocks_with_fonts(page)
    for idx, tb in enumerate(text_blocks):
        # Get section path from pre-computed paths or use current
        if page_block_paths is not None:
            section_path = page_block_paths.get(idx, current_section_path)
        else:
            section_path = []

        # Detect heading level if font_profile available
        heading_level = 0
        if font_profile and tb.get("raw_block"):
            heading_info = extract_block_heading_info(tb["raw_block"], font_profile)
            heading_level = heading_info.get("heading_level", 0) or 0

            # Update current section if this is a heading
            if heading_level and heading_level > 0:
                # Heading text becomes part of the section path
                heading_text = tb["content"].strip()[:200]
                # Rebuild section path based on heading level
                if heading_level == 1:
                    current_section_path = [heading_text]
                elif heading_level == 2:
                    current_section_path = current_section_path[:1] + [heading_text]
                elif heading_level == 3:
                    current_section_path = current_section_path[:2] + [heading_text]
                section_path = current_section_path.copy()

        all_elements.append({
            "type": "text",
            "content": tb["content"],
            "image_link": None,
            "table_link": None,
            "y_position": tb["y_position"],
            "font_size": tb["font_size"],
            "is_bold": tb["is_bold"],
            "section_hierarchy": section_path,
            "heading_level": heading_level,
            "metadata": {}
        })

    # 2. Extract tables
    tables = page.find_tables()
    for i, table in enumerate(tables):
        table_data = extract_table_as_image(page, table, page_no, i, out)
        all_elements.append({
            "type": "table",
            "content": table_data["table_markdown"],
            "image_link": None,
            "table_link": table_data["table_image_path"],
            "y_position": table_data["y_position"],
            "section_hierarchy": current_section_path.copy(),
            "heading_level": 0,
            "metadata": {
                "row_count": table_data["row_count"],
                "col_count": table_data["col_count"]
            }
        })

    # 3. Extract images
    images = page.get_images(full=True)
    for i, img in enumerate(images):
        img_data = extract_image(page, img, page_no, i, out)
        all_elements.append({
            "type": "image",
            "content": None,
            "image_link": img_data["image_path"],
            "table_link": None,
            "y_position": img_data["y_position"],
            "section_hierarchy": current_section_path.copy(),
            "heading_level": 0,
            "metadata": {}
        })

    # Sort by y_position to maintain reading order
    all_elements.sort(key=lambda x: x["y_position"])

    # Assign positions and clean up
    for i, element in enumerate(all_elements):
        element["position"] = i
        del element["y_position"]

    result["blocks"] = all_elements
    return result


def process_page(page, page_no: int, pdf_name: str, out: str) -> Dict[str, Any]:
    """
    Legacy function for backward compatibility.
    Use process_page_with_positions for new multi-modal processing.
    """
    text = page.get_text("text").strip()
    result = {
        "metadata": {
            "pdf_name": pdf_name,
            "page_no": page_no + 1,
            "page_count": page.parent.page_count
        },
        "scanned": False,
        "text_markdown": None,
        "images": [],
        "tables": []
    }

    if not text:
        result["scanned"] = True
        result["images"].append({"image_path": save_full_page_image(page, page_no, out)})
        return result

    result["text_markdown"] = text

    # Extract images
    for i, img in enumerate(page.get_images(full=True)):
        img_data = extract_image(page, img, page_no, i, out)
        result["images"].append({"image_path": img_data["image_path"]})

    # Extract tables (all saved as images now)
    for i, table in enumerate(page.find_tables()):
        table_data = extract_table_as_image(page, table, page_no, i, out)
        result["tables"].append({
            "table_markdown": table_data["table_markdown"],
            "table_image_path": table_data["table_image_path"],
            "row_count": table_data["row_count"],
            "col_count": table_data["col_count"]
        })

    return result

def page_to_markdown(page):
    md = []
    meta = page["metadata"]
    md.append(f"## 📘 Page {meta['page_no']}")
    md.append(f"**Scanned:** {'Yes' if page['scanned'] else 'No'}\n")

    if page["text_markdown"]:
        md.append("### 📝 Text")
        md.append(page["text_markdown"])

    if page["images"]:
        md.append("\n### 🖼 Images")
        for img in page["images"]:
            md.append(f"- {img['image_path']}")

    if page["tables"]:
        md.append("\n### 📊 Tables")
        for t in page["tables"]:
            md.append(t["table_markdown"])
            if t["complex_table"]:
                md.append(f"\n🖼 Table Image: {t['table_image_path']}")

    return "\n\n".join(md)
