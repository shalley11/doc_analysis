#!/usr/bin/env python3
"""
Test script for structure-based PDF chunking.
Tests the new hierarchical section detection and structure-aware chunking.
"""

import json
import fitz
from pathlib import Path

from pdf.structure_detector import (
    analyze_document_fonts,
    detect_document_structure,
    get_section_path_string
)
from pdf.pdf_utils import process_page_with_structure
from chunking.chunking_utils import process_document_with_structure


def test_structure_chunking(pdf_path: str, output_dir: str = "./test_output"):
    """Test structure-based chunking on a PDF."""
    print(f"\n{'='*60}")
    print(f"Testing Structure-Based Chunking")
    print(f"PDF: {pdf_path}")
    print(f"{'='*60}\n")

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Open PDF
    doc = fitz.open(pdf_path)
    pdf_name = Path(pdf_path).name
    print(f"Document: {pdf_name}")
    print(f"Pages: {doc.page_count}")

    # Step 1: Analyze document fonts
    print("\n--- Step 1: Font Analysis ---")
    font_profile = analyze_document_fonts(doc)
    print(f"Body font size: {font_profile.body_size}")
    print(f"Heading sizes detected: {font_profile.heading_sizes[:5]}")
    print(f"Size-to-level mapping: {font_profile.size_to_level}")
    print(f"Bold at body is H3: {font_profile.bold_at_body_is_h3}")

    # Step 2: Detect document structure
    print("\n--- Step 2: Structure Detection ---")
    font_profile, page_block_paths = detect_document_structure(doc)

    # Show section paths for first few pages
    for page_no in range(min(3, doc.page_count)):
        block_paths = page_block_paths.get(page_no, {})
        print(f"\nPage {page_no + 1} section paths:")
        for block_idx, path in list(block_paths.items())[:5]:
            path_str = get_section_path_string(path) if path else "(no section)"
            print(f"  Block {block_idx}: {path_str}")

    # Step 3: Extract pages with structure
    print("\n--- Step 3: Page Extraction with Structure ---")
    pages_result = []
    for i, page in enumerate(doc):
        block_paths = page_block_paths.get(i, {})
        page_result = process_page_with_structure(
            page, i, pdf_name, output_path,
            font_profile=font_profile,
            page_block_paths=block_paths
        )
        pages_result.append(page_result)

        # Show block info for first page
        if i == 0:
            print(f"\nPage 1 blocks:")
            for j, block in enumerate(page_result.get("blocks", [])[:5]):
                section = block.get("section_hierarchy", [])
                heading_level = block.get("heading_level", 0)
                content = block.get("content") or ""
                content_preview = content[:50].replace("\n", " ") if content else "(no content)"
                print(f"  Block {j}: level={heading_level}, section={section}")
                print(f"    Content: {content_preview}...")

    doc.close()

    # Step 4: Create structure-aware chunks
    print("\n--- Step 4: Structure-Aware Chunking ---")
    chunks = process_document_with_structure(
        pages=pages_result,
        pdf_name=pdf_name,
        max_words=500,
        min_words=50
    )

    print(f"\nTotal chunks created: {len(chunks)}")

    # Analyze chunks
    chunks_with_sections = sum(1 for c in chunks if c.get("section_hierarchy"))
    content_types = {}
    heading_levels = {0: 0, 1: 0, 2: 0, 3: 0}
    word_counts = []

    for chunk in chunks:
        ct = chunk.get("content_type", "text")
        content_types[ct] = content_types.get(ct, 0) + 1

        hl = chunk.get("heading_level", 0)
        if hl in heading_levels:
            heading_levels[hl] += 1

        word_counts.append(len(chunk.get("text", "").split()))

    print(f"Chunks with section hierarchy: {chunks_with_sections}")
    print(f"Content types: {content_types}")
    print(f"Heading levels: {heading_levels}")
    print(f"Word count stats: min={min(word_counts)}, max={max(word_counts)}, avg={sum(word_counts)//len(word_counts)}")

    # Check max word limit
    over_limit = [wc for wc in word_counts if wc > 500]
    if over_limit:
        print(f"WARNING: {len(over_limit)} chunks exceed 500 words!")
    else:
        print("All chunks are within 500 word limit.")

    # Step 5: Show sample chunks
    print("\n--- Step 5: Sample Chunks ---")
    for i, chunk in enumerate(chunks[:5]):
        print(f"\n--- Chunk {i + 1} ---")
        print(f"Type: {chunk.get('content_type')}")
        print(f"Page: {chunk.get('page_no')}")
        print(f"Section: {chunk.get('section_hierarchy')}")
        print(f"Heading Level: {chunk.get('heading_level')}")
        print(f"Words: {len(chunk.get('text', '').split())}")
        text_preview = chunk.get("text", "")[:300].replace("\n", "\\n")
        print(f"Text: {text_preview}...")

    # Save chunks to file
    output_file = output_path / "structure_chunks.json"
    with open(output_file, "w", encoding="utf-8") as f:
        # Remove non-serializable metadata
        chunks_to_save = []
        for chunk in chunks:
            chunk_copy = {k: v for k, v in chunk.items()
                         if k != "metadata" or isinstance(v, (dict, list, str, int, float, bool, type(None)))}
            chunks_to_save.append(chunk_copy)
        json.dump(chunks_to_save, f, ensure_ascii=False, indent=2)
    print(f"\nChunks saved to: {output_file}")

    # Summary
    print(f"\n{'='*60}")
    print("TEST SUMMARY")
    print(f"{'='*60}")
    print(f"PDF: {pdf_name}")
    print(f"Pages: {len(pages_result)}")
    print(f"Total Chunks: {len(chunks)}")
    print(f"Chunks with Sections: {chunks_with_sections}")
    print(f"Content Types: {content_types}")
    print(f"All chunks <= 500 words: {'YES' if not over_limit else 'NO'}")
    print(f"{'='*60}\n")

    return chunks


if __name__ == "__main__":
    # Test with test3.pdf
    pdf_path = "/home/labuser/github-workspace/summerizer/test3.pdf"
    chunks = test_structure_chunking(pdf_path)
