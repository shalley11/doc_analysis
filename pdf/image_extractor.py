"""
Image extraction from PDF documents using PyMuPDF.

Extracts embedded images from PDF pages with metadata including
bounding box, dimensions, and saves images to disk for vision model processing.
"""
import fitz
import os
import uuid
from typing import List, Dict, Optional

from doc_analysis.config import IMAGE_STORE_DIR


# Minimum image dimensions to filter out icons/decorations
MIN_IMAGE_WIDTH = 50
MIN_IMAGE_HEIGHT = 50

# Minimum area (width * height) to consider as meaningful image
MIN_IMAGE_AREA = 5000


def extract_images(
    pdf_path: str,
    pdf_id: str,
    pdf_name: str,
    start_page: Optional[int] = None,
    end_page: Optional[int] = None,
    min_width: int = MIN_IMAGE_WIDTH,
    min_height: int = MIN_IMAGE_HEIGHT,
    min_area: int = MIN_IMAGE_AREA,
) -> List[Dict]:
    """
    Extract images from PDF using PyMuPDF.

    Args:
        pdf_path: Path to PDF file
        pdf_id: Unique PDF identifier
        pdf_name: Original PDF filename
        start_page: Start page (1-indexed, inclusive). None means page 1.
        end_page: End page (1-indexed, inclusive). None means last page.
        min_width: Minimum image width to extract
        min_height: Minimum image height to extract
        min_area: Minimum image area (width * height) to extract

    Returns:
        List of image elements with type, image_path, page_no, bbox, and metadata
    """
    elements = []
    image_counter = 0

    os.makedirs(IMAGE_STORE_DIR, exist_ok=True)

    doc = fitz.open(pdf_path)
    total_pages = len(doc)

    # Determine page range
    first_page = start_page if start_page else 1
    last_page = end_page if end_page else total_pages

    # Clamp to valid range
    first_page = max(1, min(first_page, total_pages))
    last_page = max(first_page, min(last_page, total_pages))

    for page_no in range(first_page, last_page + 1):
        page = doc[page_no - 1]  # 0-indexed
        page_height = page.rect.height
        page_width = page.rect.width

        # Get all images on the page
        image_list = page.get_images(full=True)

        for img_index, img_info in enumerate(image_list):
            xref = img_info[0]  # Image xref number

            try:
                # Get image bbox on the page
                img_rects = page.get_image_rects(xref)

                if not img_rects:
                    continue

                # Use the first occurrence of the image
                img_rect = img_rects[0]
                bbox = [img_rect.x0, img_rect.y0, img_rect.x1, img_rect.y1]

                # Calculate dimensions
                width = img_rect.width
                height = img_rect.height
                area = width * height

                # Filter small images (icons, bullets, etc.)
                if width < min_width or height < min_height or area < min_area:
                    continue

                image_counter += 1

                # Extract and save the image
                image_path = _save_image(
                    doc, xref, pdf_id, image_counter, page, img_rect
                )

                if not image_path:
                    continue

                element = {
                    "type": "image",
                    "text": f"[Image {image_counter}]",  # Placeholder text for chunking
                    "image_path": image_path,
                    "page_no": page_no,
                    "pdf_id": pdf_id,
                    "pdf_name": pdf_name,
                    "page_height": page_height,
                    "page_width": page_width,
                    "bbox": bbox,
                    "image_width": int(width),
                    "image_height": int(height),
                    "image_id": image_counter,
                }

                elements.append(element)

            except Exception as e:
                print(f"Failed to extract image {img_index} on page {page_no}: {e}")
                continue

    doc.close()
    return elements


def _save_image(
    doc: fitz.Document,
    xref: int,
    pdf_id: str,
    image_id: int,
    page: fitz.Page,
    rect: fitz.Rect,
) -> Optional[str]:
    """
    Save an image from the PDF to disk.

    Tries to extract the raw image first, falls back to rendering
    the region if raw extraction fails.

    Args:
        doc: PyMuPDF document
        xref: Image xref number
        pdf_id: PDF identifier
        image_id: Image counter
        page: Page containing the image
        rect: Image bounding rectangle

    Returns:
        Path to saved image or None if failed
    """
    image_filename = f"image_{pdf_id}_{image_id}_{uuid.uuid4().hex[:8]}.png"
    image_path = os.path.join(IMAGE_STORE_DIR, image_filename)

    try:
        # Try to extract raw image data
        base_image = doc.extract_image(xref)

        if base_image and base_image.get("image"):
            # Save raw image bytes
            image_bytes = base_image["image"]
            ext = base_image.get("ext", "png")

            # Update filename with correct extension
            if ext != "png":
                image_filename = f"image_{pdf_id}_{image_id}_{uuid.uuid4().hex[:8]}.{ext}"
                image_path = os.path.join(IMAGE_STORE_DIR, image_filename)

            with open(image_path, "wb") as f:
                f.write(image_bytes)

            return image_path

    except Exception:
        pass

    # Fallback: render the region as pixmap
    try:
        # Add small padding
        padded_rect = fitz.Rect(rect)
        padded_rect.x0 = max(0, padded_rect.x0 - 2)
        padded_rect.y0 = max(0, padded_rect.y0 - 2)
        padded_rect.x1 = min(page.rect.width, padded_rect.x1 + 2)
        padded_rect.y1 = min(page.rect.height, padded_rect.y1 + 2)

        # Render at 2x resolution for quality
        mat = fitz.Matrix(2, 2)
        pix = page.get_pixmap(matrix=mat, clip=padded_rect)
        pix.save(image_path)

        return image_path

    except Exception as e:
        print(f"Failed to save image: {e}")
        return None


def get_image_count(pdf_path: str, start_page: Optional[int] = None, end_page: Optional[int] = None) -> int:
    """
    Count images in a PDF without extracting them.

    Args:
        pdf_path: Path to PDF file
        start_page: Start page (1-indexed)
        end_page: End page (1-indexed)

    Returns:
        Total count of images in the specified page range
    """
    doc = fitz.open(pdf_path)
    total_pages = len(doc)

    first_page = start_page if start_page else 1
    last_page = end_page if end_page else total_pages

    first_page = max(1, min(first_page, total_pages))
    last_page = max(first_page, min(last_page, total_pages))

    count = 0
    for page_no in range(first_page, last_page + 1):
        page = doc[page_no - 1]
        count += len(page.get_images())

    doc.close()
    return count
