import fitz
import os
import uuid
from doc_analysis.config import IMAGE_STORE_DIR, DEFAULT_CHUNKING_CONFIG


def extract_layout_blocks(pdf_path: str, pdf_id: str = "", pdf_name: str = ""):
    """
    Extract layout blocks from PDF with enhanced metadata.

    Returns blocks with:
    - bbox: [x0, y0, x1, y1] coordinates
    - font_info: dominant font name, size, flags (bold/italic)
    - spans: raw span data for heading inference
    """
    doc = fitz.open(pdf_path)
    blocks = []

    os.makedirs(IMAGE_STORE_DIR, exist_ok=True)

    for page_no, page in enumerate(doc, 1):
        page_dict = page.get_text("dict")
        page_height = page.rect.height
        page_width = page.rect.width

        for block in page_dict.get("blocks", []):
            bbox = block.get("bbox", [0, 0, 0, 0])

            # --------------------
            # TEXT BLOCK
            # --------------------
            if block.get("type") == 0:
                lines = block.get("lines", [])
                text = " ".join(
                    span.get("text", "")
                    for line in lines
                    for span in line.get("spans", [])
                ).strip()

                if not text:
                    continue

                # Extract font information from spans
                font_info = _extract_font_info(lines)

                # Preserve spans for heading inference
                spans_data = [
                    {
                        "spans": [
                            {
                                "text": span.get("text", ""),
                                "size": span.get("size", 12),
                                "flags": span.get("flags", 0),
                                "font": span.get("font", ""),
                            }
                            for span in line.get("spans", [])
                        ]
                    }
                    for line in lines
                ]

                blocks.append({
                    "type": "paragraph",
                    "text": text,
                    "page_no": page_no,
                    "pdf_id": pdf_id,
                    "pdf_name": pdf_name,
                    "bbox": list(bbox),
                    "font_info": font_info,
                    "spans": spans_data,
                    "page_height": page_height,
                    "page_width": page_width,
                })

            # --------------------
            # IMAGE BLOCK (DEFENSIVE)
            # --------------------
            elif block.get("type") == 1:
                if not bbox or len(bbox) < 4:
                    continue

                try:
                    pix = page.get_pixmap(clip=bbox)
                except Exception:
                    continue

                image_id = str(uuid.uuid4())
                image_path = os.path.join(
                    IMAGE_STORE_DIR,
                    f"{image_id}.png"
                )

                try:
                    pix.save(image_path)
                except Exception:
                    continue

                blocks.append({
                    "type": "image",
                    "image_path": image_path,
                    "page_no": page_no,
                    "pdf_id": pdf_id,
                    "pdf_name": pdf_name,
                    "bbox": list(bbox),
                    "page_height": page_height,
                    "page_width": page_width,
                })

    doc.close()
    return blocks


def _extract_font_info(lines: list) -> dict:
    """
    Extract dominant font information from block lines.

    Returns dict with:
    - name: font name
    - size: font size
    - bold: is bold (flags & 2^4)
    - italic: is italic (flags & 2^1)
    """
    fonts = []

    for line in lines:
        for span in line.get("spans", []):
            fonts.append({
                "name": span.get("font", ""),
                "size": span.get("size", 12),
                "flags": span.get("flags", 0),
                "length": len(span.get("text", "")),
            })

    if not fonts:
        return {"name": "", "size": 12, "bold": False, "italic": False}

    # Find dominant font by text length
    dominant = max(fonts, key=lambda f: f["length"])
    flags = dominant["flags"]

    return {
        "name": dominant["name"],
        "size": round(dominant["size"], 1),
        "bold": bool(flags & 16),      # bit 4 = bold
        "italic": bool(flags & 2),      # bit 1 = italic
    }
