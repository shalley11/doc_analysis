#!/usr/bin/env python3
"""
Test script for improved chunking functionality.
Tests cross-page merging, list-aware chunking, and table/figure extraction.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import fitz  # PyMuPDF
from chunking.chunking_utils import (
    sentence_split,
    is_list_item,
    is_heading,
    extract_lists_from_text,
    extract_tables_and_figures,
    process_document_pages,
    merge_cross_page_content,
    detect_content_structure,
    ContentType
)


def extract_text_from_pdf(pdf_path: str) -> list:
    """Extract text from PDF, page by page."""
    pages = []
    doc = fitz.open(pdf_path)

    for page_num, page in enumerate(doc):
        text = page.get_text("text")
        pages.append({
            "page_no": page_num + 1,
            "text": text,
            "blocks": [],
            "metadata": {"pdf_name": os.path.basename(pdf_path), "page_no": page_num + 1}
        })

    doc.close()
    return pages


def test_sentence_splitting():
    """Test improved sentence splitting."""
    print("\n" + "=" * 60)
    print("TEST: Sentence Splitting")
    print("=" * 60)

    test_cases = [
        # Test abbreviations
        "This is e.g. an example. This is another sentence.",
        "Dr. Smith went to the store. He bought milk.",
        "See Fig. 1 for details. The results are shown.",
        # Test footnotes
        "This is referenced.1 This is another point.",
        "Complex systems modeling is useful.2'3 More text here.",
        # Test URLs (conceptual)
        "Visit the site at example.com for more. Next sentence.",
        # Test decimals
        "The value was 3.14 units. Another measurement was 2.5.",
    ]

    for test in test_cases:
        sentences = sentence_split(test)
        print(f"\nInput: {test}")
        print(f"Sentences ({len(sentences)}):")
        for i, s in enumerate(sentences):
            print(f"  {i+1}. {s}")


def test_list_detection():
    """Test list item detection."""
    print("\n" + "=" * 60)
    print("TEST: List Item Detection")
    print("=" * 60)

    test_items = [
        "- This is a bullet point",
        "• Another bullet style",
        "1. Numbered item",
        "a) Letter item",
        "(i) Roman numeral",
        "Regular text without bullet",
        "   - Indented bullet",
        "* Asterisk bullet",
    ]

    for item in test_items:
        is_list = is_list_item(item)
        print(f"  {'[LIST]' if is_list else '[TEXT]'} {item}")


def test_heading_detection():
    """Test heading detection."""
    print("\n" + "=" * 60)
    print("TEST: Heading Detection")
    print("=" * 60)

    test_headings = [
        "I. What is a complex system?",
        "II. Why use complex systems modeling?",
        "III. When is complex systems modeling feasible?",
        "IV. How is a complex systems model developed?",
        "V. What's next for complex systems modeling?",
        "Executive Summary",
        "Table of Contents",
        "References",
        "Figure 1. Criteria for determining feasibility",
        "This is regular text, not a heading.",
        "1. Introduction to the topic",
    ]

    for heading in test_headings:
        is_head = is_heading(heading)
        print(f"  {'[HEADING]' if is_head else '[TEXT]   '} {heading}")


def test_list_extraction():
    """Test list extraction from text."""
    print("\n" + "=" * 60)
    print("TEST: List Extraction")
    print("=" * 60)

    test_text = """
The research yielded the following key findings:
- There is strong interest within the humanitarian community in the prospect of using complex systems modeling.
- To determine which problems are best suited to being modeled using complex systems techniques, humanitarians should consider the following criteria:
  - Feasibility: clear goals and output; clear target geographic area.
  - Viability: data availability; synergy with domain experts.
  - Desirability: complexity of the problem space; potential impact.

The main recommendations include:
1. The Centre should coordinate initial pilots.
2. The Centre should develop partnerships with actors from technical research environments.
3. The Centre should create medium-term research fellowships.
"""

    blocks = extract_lists_from_text(test_text)

    print(f"\nExtracted {len(blocks)} blocks:")
    for i, block in enumerate(blocks):
        block_type = block.get("type", "unknown")
        content = block.get("content", "")[:100]
        print(f"\n  Block {i+1} [{block_type.upper()}]:")
        print(f"    {content}...")
        if block_type == "list":
            print(f"    Items: {block.get('metadata', {}).get('item_count', 0)}")


def test_table_figure_extraction():
    """Test table and figure extraction."""
    print("\n" + "=" * 60)
    print("TEST: Table/Figure Extraction")
    print("=" * 60)

    test_text = """
Some introductory text here.

Figure 1. Criteria for determining the feasibility, viability and desirability of complex systems modeling in humanitarian settings

Feasibility | Viability | Desirability
Clear goals and output | Data availability | Complexity of the problem space
Clear target geographic area | Synergy with domain experts | Ability to inform response activities

More text after the table.

Table 2. Selected methods for complex systems modeling

Method | Description
Network models | Components represented as nodes
Agent-based modeling | Simulate behavior of agents
System Dynamics | Represent system at macroscale

Final paragraph of text.
"""

    remaining, extracted = extract_tables_and_figures(test_text)

    print(f"\nExtracted {len(extracted)} tables/figures:")
    for i, item in enumerate(extracted):
        print(f"\n  {i+1}. Type: {item.get('type', 'unknown').upper()}")
        print(f"     Caption: {item.get('caption', 'N/A')}")
        print(f"     Content preview: {item.get('content', '')[:80]}...")

    print(f"\nRemaining text preview:")
    print(f"  {remaining[:200]}...")


def test_full_pdf_chunking(pdf_path: str):
    """Test full PDF processing with improved chunking."""
    print("\n" + "=" * 60)
    print(f"TEST: Full PDF Chunking - {os.path.basename(pdf_path)}")
    print("=" * 60)

    if not os.path.exists(pdf_path):
        print(f"  ERROR: PDF not found at {pdf_path}")
        return

    # Extract pages
    pages = extract_text_from_pdf(pdf_path)
    print(f"\n  Extracted {len(pages)} pages from PDF")

    # Process with improved chunking
    pdf_name = os.path.basename(pdf_path)
    chunks = process_document_pages(
        pages=pages,
        pdf_name=pdf_name,
        max_words=500,
        enable_cross_page_merge=True,
        enable_list_aware=True,
        enable_table_figure_extraction=True
    )

    print(f"  Created {len(chunks)} chunks")

    # Analyze chunk types
    type_counts = {}
    word_counts = []
    for chunk in chunks:
        content_type = chunk.get("content_type", "unknown")
        type_counts[content_type] = type_counts.get(content_type, 0) + 1
        word_counts.append(len(chunk.get("text", "").split()))

    print(f"\n  Chunk type distribution:")
    for ctype, count in sorted(type_counts.items()):
        print(f"    {ctype}: {count}")

    print(f"\n  Word count statistics:")
    print(f"    Min: {min(word_counts)} words")
    print(f"    Max: {max(word_counts)} words")
    print(f"    Avg: {sum(word_counts) / len(word_counts):.1f} words")

    # Show sample chunks
    print(f"\n  Sample chunks:")
    for i, chunk in enumerate(chunks[:5]):
        print(f"\n  --- Chunk {i+1} ---")
        print(f"  Type: {chunk.get('content_type', 'unknown')}")
        print(f"  Page: {chunk.get('page_no', 'N/A')}")
        print(f"  Words: {len(chunk.get('text', '').split())}")
        print(f"  Method: {chunk.get('metadata', {}).get('chunking_method', 'N/A')}")
        text_preview = chunk.get("text", "")[:150].replace('\n', ' ')
        print(f"  Preview: {text_preview}...")

    # Check for any chunks exceeding max_words
    oversized = [c for c in chunks if len(c.get("text", "").split()) > 500]
    if oversized:
        print(f"\n  WARNING: {len(oversized)} chunks exceed 500 words!")
        for chunk in oversized[:3]:
            print(f"    - {len(chunk.get('text', '').split())} words on page {chunk.get('page_no')}")
    else:
        print(f"\n  All chunks are within 500 word limit")

    return chunks


def test_cross_page_merge():
    """Test cross-page content merging."""
    print("\n" + "=" * 60)
    print("TEST: Cross-Page Merging")
    print("=" * 60)

    # Simulate pages with split content
    pages = [
        {
            "page_no": 1,
            "text": "This is the first page with some content. The sentence continues on the",
            "blocks": [],
            "metadata": {"pdf_name": "test.pdf", "page_no": 1}
        },
        {
            "page_no": 2,
            "text": "next page seamlessly. Here is more complete content.",
            "blocks": [],
            "metadata": {"pdf_name": "test.pdf", "page_no": 2}
        }
    ]

    print("\n  Before merging:")
    for p in pages:
        print(f"    Page {p['page_no']}: {p['text'][:50]}...")

    merged = merge_cross_page_content(pages)

    print("\n  After merging:")
    for p in merged:
        text = p.get('text', '')
        print(f"    Page {p['page_no']}: {text[:50]}...")
        if p.get('metadata', {}).get('merged_from_previous'):
            print(f"      [Merged from previous page]")
        if p.get('metadata', {}).get('continues_to_next'):
            print(f"      [Continues to next page]")


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("  IMPROVED CHUNKING TEST SUITE")
    print("=" * 60)

    # Run unit tests
    test_sentence_splitting()
    test_list_detection()
    test_heading_detection()
    test_list_extraction()
    test_table_figure_extraction()
    test_cross_page_merge()

    # Test with actual PDF if available
    pdf_path = "test3.pdf"
    if os.path.exists(pdf_path):
        chunks = test_full_pdf_chunking(pdf_path)

        # Print summary of improvements
        print("\n" + "=" * 60)
        print("  CHUNKING IMPROVEMENTS SUMMARY")
        print("=" * 60)
        print("""
  Features Implemented:
  ---------------------
  1. Improved Sentence Splitting
     - Handles abbreviations (e.g., i.e., Dr., Fig.)
     - Handles footnote references (text.1)
     - Handles decimal numbers (3.14)
     - Handles URLs

  2. Cross-Page Merging
     - Merges incomplete sentences across pages
     - Merges split lists across pages

  3. List-Aware Chunking
     - Detects bullet points (-, *, etc.)
     - Detects numbered lists (1., a), (i))
     - Keeps list items together

  4. Table/Figure Extraction
     - Extracts figures as separate chunks
     - Extracts tables as separate chunks
     - Preserves captions

  5. Minimum Chunk Size Enforcement
     - Merges small chunks (< 20 words)
     - Removes artifacts (page numbers, headers)

  6. Maximum Chunk Size (500 words)
     - Splits large content at sentence boundaries
     - Preserves context within chunks
        """)
    else:
        print(f"\n  Skipping PDF test - {pdf_path} not found")

    print("\n" + "=" * 60)
    print("  ALL TESTS COMPLETED")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
