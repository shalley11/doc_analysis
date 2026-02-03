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
    is_page_number,
    is_table_of_contents_header,
    remove_page_numbers,
    extract_lists_from_text,
    extract_tables_and_figures,
    extract_table_of_contents,
    process_document_pages,
    process_document_with_strategy,
    merge_cross_page_content,
    ContentType,
    # Structure-preserving functions
    chunk_table_with_structure,
    chunk_figure_with_structure,
    chunk_toc_with_structure,
    chunk_list_with_heading,
    chunk_heading_with_subpoints
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


def test_page_number_removal():
    """Test page number detection and removal."""
    print("\n" + "=" * 60)
    print("TEST: Page Number Removal")
    print("=" * 60)

    test_cases = [
        "5",
        " 12 ",
        "- 5 -",
        "Page 5",
        "Page 5 of 10",
        "p. 5",
        "[5]",
        "This is regular text",
        "Section 5: Introduction",
        "123",  # Could be page number
        "1234",  # Probably not a page number
    ]

    print("\n  Page number detection:")
    for test in test_cases:
        is_pn = is_page_number(test)
        print(f"    {'[PAGE#]' if is_pn else '[TEXT] '} \"{test}\"")

    # Test removal
    test_text = """
This is some content.
5
More content here.
Page 10
Even more content.
- 15 -
Final content.
    """
    cleaned = remove_page_numbers(test_text)
    print(f"\n  Before removal:\n{test_text}")
    print(f"\n  After removal:\n{cleaned}")


def test_toc_detection():
    """Test Table of Contents detection."""
    print("\n" + "=" * 60)
    print("TEST: Table of Contents Detection")
    print("=" * 60)

    test_text = """
Some intro text.

Table of Contents
Executive Summary ............................. 3
I. What is a complex system? .................. 5
II. Why use complex systems? .................. 6
III. When is it feasible? ..................... 7
References .................................... 16

Main content starts here.
This is the actual document content.
    """

    remaining, toc_block = extract_table_of_contents(test_text)

    if toc_block:
        print(f"\n  TOC Extracted: Yes")
        print(f"  Entry count: {toc_block.get('metadata', {}).get('entry_count', 0)}")
        print(f"\n  TOC Content:\n{toc_block.get('content', '')[:200]}...")
    else:
        print(f"\n  TOC Extracted: No")

    print(f"\n  Remaining text:\n{remaining[:200]}...")


def test_new_chunking_strategy(pdf_path: str):
    """Test the new chunking strategy with all requirements."""
    print("\n" + "=" * 60)
    print(f"TEST: New Chunking Strategy - {os.path.basename(pdf_path)}")
    print("=" * 60)

    if not os.path.exists(pdf_path):
        print(f"  ERROR: PDF not found at {pdf_path}")
        return

    # Extract pages
    pages = extract_text_from_pdf(pdf_path)
    print(f"\n  Extracted {len(pages)} pages from PDF")

    # Process with new strategy
    pdf_name = os.path.basename(pdf_path)
    chunks = process_document_with_strategy(
        pages=pages,
        pdf_name=pdf_name,
        max_words=500,
        min_words=50
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

    # Check for cross-page references
    cross_page_refs = sum(1 for c in chunks if c.get("metadata", {}).get("has_cross_page_reference"))
    print(f"\n  Cross-page references: {cross_page_refs}")

    # Show sample chunks by type
    print(f"\n  Sample chunks by type:")
    shown_types = set()
    for chunk in chunks:
        ctype = chunk.get("content_type", "unknown")
        if ctype not in shown_types:
            shown_types.add(ctype)
            print(f"\n  --- {ctype.upper()} chunk ---")
            print(f"  Page: {chunk.get('page_no', 'N/A')}")
            print(f"  Words: {len(chunk.get('text', '').split())}")
            text_preview = chunk.get("text", "")[:150].replace('\n', ' ')
            print(f"  Preview: {text_preview}...")

    # Verify constraints
    oversized = [c for c in chunks if len(c.get("text", "").split()) > 500]
    undersized = [c for c in chunks if len(c.get("text", "").split()) < 50]

    print(f"\n  Constraint verification:")
    if oversized:
        print(f"    WARNING: {len(oversized)} chunks exceed 500 words")
    else:
        print(f"    OK: All chunks <= 500 words")

    if undersized:
        print(f"    NOTE: {len(undersized)} chunks < 50 words (may be tables/figures/toc)")
    else:
        print(f"    OK: All chunks >= 50 words")

    # Check for page numbers in chunks
    page_num_found = False
    for chunk in chunks:
        text = chunk.get("text", "")
        lines = text.split('\n')
        for line in lines:
            if is_page_number(line.strip()):
                page_num_found = True
                break

    if page_num_found:
        print(f"    WARNING: Some page numbers may still exist")
    else:
        print(f"    OK: No standalone page numbers found")

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


def test_structure_preserving_chunking():
    """Test structure-preserving chunking functions."""
    print("\n" + "=" * 60)
    print("TEST: Structure-Preserving Chunking")
    print("=" * 60)

    # Test 1: Table split with header preservation
    print("\n  1. Table split with header preservation:")
    long_table = 'Table 1: Sample Data\n' + '\n'.join([f'Row {i} | Data {i} | Value {i}' for i in range(30)])
    table_chunks = chunk_table_with_structure(long_table, 'test.pdf', 1, 0, 0, max_words=100, caption='Table 1: Sample Data')
    print(f"     Created {len(table_chunks)} table chunks")
    all_have_header = all('Table 1' in c['text'] for c in table_chunks)
    print(f"     Header preserved in all chunks: {'OK' if all_have_header else 'FAIL'}")

    # Test 2: List split with heading preservation
    print("\n  2. List split with heading preservation:")
    long_list = '\n'.join([f'- Item {i}: Description here' for i in range(25)])
    list_chunks = chunk_list_with_heading(long_list, 'test.pdf', 1, 0, 0, max_words=100, heading='Key Findings:')
    print(f"     Created {len(list_chunks)} list chunks")
    all_have_heading = all('Key Findings' in c['text'] for c in list_chunks)
    print(f"     Heading preserved in all chunks: {'OK' if all_have_heading else 'FAIL'}")

    # Test 3: Heading with subpoints
    print("\n  3. Heading with subpoints:")
    subpoints = [f'• Point {i}: Explanation for point {i}' for i in range(15)]
    heading_chunks = chunk_heading_with_subpoints('III. Recommendations', subpoints, 'test.pdf', 1, 0, 0, max_words=100)
    print(f"     Created {len(heading_chunks)} chunks")
    all_have_section_heading = all('III. Recommendations' in c['text'] for c in heading_chunks)
    print(f"     Section heading in all chunks: {'OK' if all_have_section_heading else 'FAIL'}")

    # Test 4: TOC split with header preservation
    print("\n  4. TOC split with header preservation:")
    toc = 'Table of Contents\n' + '\n'.join([f'Chapter {i} ............ {i*10}' for i in range(25)])
    toc_chunks = chunk_toc_with_structure(toc, 'test.pdf', 1, 0, max_words=100)
    print(f"     Created {len(toc_chunks)} TOC chunks")
    all_have_toc_header = all('Table of Contents' in c['text'] for c in toc_chunks)
    print(f"     TOC header in all chunks: {'OK' if all_have_toc_header else 'FAIL'}")

    # Test 5: Figure split with caption preservation
    print("\n  5. Figure split with caption preservation:")
    long_figure = 'Figure 5: System Architecture\n' + ' '.join(['Description word'] * 150)
    figure_chunks = chunk_figure_with_structure(long_figure, 'test.pdf', 1, 0, 0, max_words=100, caption='Figure 5: System Architecture')
    print(f"     Created {len(figure_chunks)} figure chunks")
    all_have_caption = all('Figure 5' in c['text'] for c in figure_chunks)
    all_have_image_link = all('image_link' in c for c in figure_chunks)
    print(f"     Caption preserved in all chunks: {'OK' if all_have_caption else 'FAIL'}")
    print(f"     image_link field in all chunks: {'OK' if all_have_image_link else 'FAIL'}")

    # Summary
    print("\n  Structure Preservation Summary:")
    print("  - Split chunks keep original header/caption")
    print("  - Part numbers indicated: [Part X/Y]")
    print("  - Links (table_link, image_link) preserved")
    print("  - Metadata tracks split status and part numbers")


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("  IMPROVED CHUNKING TEST SUITE")
    print("=" * 60)

    # Run unit tests
    test_sentence_splitting()
    test_list_detection()
    test_heading_detection()
    test_page_number_removal()
    test_toc_detection()
    test_list_extraction()
    test_table_figure_extraction()
    test_cross_page_merge()
    test_structure_preserving_chunking()

    # Test with actual PDF if available
    pdf_path = "test3.pdf"
    if os.path.exists(pdf_path):
        # Test new chunking strategy
        chunks = test_new_chunking_strategy(pdf_path)

        # Print summary of improvements
        print("\n" + "=" * 60)
        print("  NEW CHUNKING STRATEGY SUMMARY")
        print("=" * 60)
        print("""
  Chunking Strategy Implementation:
  ---------------------------------
  1. PARAGRAPH CHUNKS (max 500 words)
     - Regular text split at sentence boundaries
     - Section heading preserved in all split parts
     - Preserves semantic coherence

  2. TABLE SUMMARY (separate chunk)
     - Each table extracted as individual chunk
     - STRUCTURE PRESERVED: Header/caption in all split parts
     - table_link maintained in all chunks

  3. IMAGE SUMMARY (separate chunk)
     - Each figure/image as individual chunk
     - STRUCTURE PRESERVED: Caption in all split parts
     - image_link maintained in all chunks

  4. LIST (separate chunk)
     - Bullet and numbered lists as separate chunks
     - STRUCTURE PRESERVED: Heading in all split parts
     - Large lists split at item boundaries

  5. TABLE OF CONTENTS (separate chunk)
     - TOC detected and extracted separately
     - STRUCTURE PRESERVED: "Table of Contents" header in splits
     - Entry count tracked in metadata

  6. HEADING WITH SUBPOINTS
     - Section headings kept with subpoints
     - STRUCTURE PRESERVED: Heading in all split subpoint chunks
     - Part numbers indicated [Part X/Y]

  7. PAGE NUMBERS REMOVED
     - Standalone page numbers filtered out
     - Page artifacts (headers/footers) removed

  8. CROSS-PAGE REFERENCES KEPT
     - Incomplete sentences merged across pages
     - Cross-page metadata tracked
     - Context links between chunks maintained

  9. MINIMUM CHUNK SIZE (50 words)
     - Small chunks merged with neighbors
     - Ensures meaningful context in each chunk
        """)
    else:
        print(f"\n  Skipping PDF test - {pdf_path} not found")

    print("\n" + "=" * 60)
    print("  ALL TESTS COMPLETED")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
