# pdf_utils.py
from pathlib import Path
from typing import List, Dict, Any
import fitz  # PyMuPDF


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
