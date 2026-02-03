"""PDF processing module for extraction and vision processing."""

from .pdf_utils import (
    process_page,
    process_page_with_positions,
    process_page_with_structure,
    get_text_blocks_with_fonts
)
from .vision_utils import VisionProcessor
from .structure_detector import (
    FontProfile,
    analyze_document_fonts,
    classify_heading_level,
    build_section_hierarchy,
    detect_document_structure,
    get_section_path_string
)

__all__ = [
    "process_page",
    "process_page_with_positions",
    "process_page_with_structure",
    "get_text_blocks_with_fonts",
    "VisionProcessor",
    "FontProfile",
    "analyze_document_fonts",
    "classify_heading_level",
    "build_section_hierarchy",
    "detect_document_structure",
    "get_section_path_string"
]
